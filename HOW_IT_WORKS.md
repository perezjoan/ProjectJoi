# How v3 works

A reference for the running system as of 10 September 2026: every mechanism, the formulas behind it, and the
knobs that control it. Where a value is a default it is written as `name = value` and can be changed in
`config.json`. Sections follow the order of one turn.

---

## 0. One turn, end to end

```
user text
  │
  ├─ 1. recall        memory RAG: adaptive retrieval of earlier interactions (§5)
  ├─ 2. prompt        persona + instructions + time line + mood line + body line + memories + schema (§3)
  ├─ 3. brain         Qwen3 → {"emotion", "move", "reply"} in that order (§3)
  ├─ 4. reading       emotion line → lexicon → (v, a, d) raw → standardised point (§1)
  ├─ 5. mood          point nudged toward the reading; sincerity gap logged (§2)
  ├─ 6. body          move → clip RAG → node; budget check; or drift to the node nearest the mood (§4)
  ├─ 7. play          the graph routes from the current node; the window plays the clips (§4)
  ├─ 8. record        episode log + state file (§7), then the memory record (§5)
  └─ 9. window        reply, reading, body decision, mood, gauge (§8)
```

Every second (`tick_seconds = 1.0`) the mood decays and the window is told where it is. Every 60 s of silence the
body may walk to the node nearest the mood (§4.6). After 180 s of silence the brain gets an internal prompt and may
say something to herself (§3.5). On start, if she was switched off for at least 60 s, she gets a wake-up event (§7.3).

---

## 1. The space and the lexicon

### 1.1 The space

Three axes, each in standardised units where the seed keyframes span about [-1, 1]:

| axis | low | high |
|---|---|---|
| valence `v` | unpleasant | pleasant |
| arousal `a` | calm | energised |
| dominance `d` | submissive, small | in control |

Two occupants share it: the **mood**, a point anywhere; and the **keyframes** (painted samples), nine points the
body can stand on. Every decision below is a distance in this space: `dist(p, q) = sqrt(Σ (p_k - q_k)²)`.

### 1.2 Scoring a text (`affect/lexicon.py`)

Input: an emotional text (the emotion line, a seed keyframe's text). Never a physical description.

1. **Tokenise**: lower-case, `n't → not`, split into words; sentence punctuation `. ; ! ?` and the conjunctions
   `because, since, but, although, though, while, whereas, yet, even if, even though, so that, as if` open a new
   clause (index `clause[i]`).
2. **Match** left to right, longest phrase first (up to 3 words), against the NRC VAD v2.1 lexicon (54 800 terms,
   values in [-1, 1]). A word not found is tried as a lemma (`-ies→y, -ily→y, -es, -s, -ed, -ing, -ly`).
   Words in the `STOPLIST` (function words, time words, mental-state verbs, narrative verbs the lexicon rates
   strongly, e.g. "still" has arousal -1.0) are skipped.
3. **Modifiers** (not scored themselves):
   - negators `not, no, never, ...`: valence of terms starting within the next `NEG_WINDOW = 3` tokens is flipped.
   - intensifiers `very, really, so, deeply, ...`: the next term is scaled by `INTENSIFY = (1.25, 1.5, 1.25)`.
   - diminishers `a bit, a little, slightly, ...`: the next term is scaled by `DIMINISH = (0.7, 0.7, 0.7)`.
   - hedges `trying to, want to, pretending to, hoping to, ...`: terms within the next `HEDGE_WINDOW = 4`
     tokens get weight × `HEDGE_FACTOR = 0.15` and never count as hits ("trying to stay calm" is not calm).
4. **Weight** of a matched term: its distance from neutral **before** modifiers,
   `w = max(‖(v, a, d)‖, 0.05)`; so a diminisher shrinks the value, not the say. Terms in a later clause get
   `w × LATER_CLAUSE_FACTOR = 0.4` (the feeling comes first; the "why" and the consolation after).
5. **Hit**: a term counts as a hit if its original norm is `≥ NEUTRAL_RADIUS = 0.15` and it is not hedged.
6. **Raw score**: weighted mean of the (modified, clipped to [-1, 1]) vectors,
   `raw = Σ w_i · vec_i / Σ w_i`.
7. **Failed reading**: `hits < min_hits = 2` → no coordinates from the lexicon; the placement fallback runs (§1.4).

Per-term output (shown in the window's reading table): term, vector, weight, hit, flags
(`negated, intensified, diminished, hedged, clause N`).

### 1.3 Standardisation against the seed (`affect/space.py`)

Each seed keyframe is scored from its emotional text: the `feeling` field of its JSON entry if present, else
`"label, aliases. <expression clause of the description>"` (symbols and hands are physical and excluded).
With `lo_k, hi_k` the min and max raw value of the seed on axis k:

```
center_k = (lo_k + hi_k) / 2
scale_k  = max((hi_k - lo_k) / 2, 1e-3)
std_k    = clip((raw_k - center_k) / scale_k, -mood_clip, +mood_clip)      mood_clip = 1.5
```

The seed therefore spans exactly [-1, 1] on each axis; a reading may exceed it up to ±1.5. The hand-set
valence/arousal in the JSON are only used for a sanity check (Pearson r ≈ 0.97 / 0.81 today) and never adopted.
The neutral keyframe carries `feeling: "calm, quiet, at rest, ..."` so its valence reads ≈ 0; it is by construction
the calmest sample (arousal -1) and is the rest point of the mood (§2.1).

### 1.4 Placement fallback (`affect/placement.py`)

When the lexicon fails, the text is embedded (bge-small, CPU) and compared with the nine seed texts:

```
s_i = cos(e(text), e(seed_i))
w_i = exp((s_i - max_j s_j) / 0.05)
p   = Σ w_i · coords_i / Σ w_i
```

The window marks such readings `via placement (top 3 candidates)`.

---

## 2. The mood (`affect/state.py`)

### 2.1 State

- `point` `p`: the mood now.
- `baseline` `b`: where it settles; slow state (temperament, §2.4). Starts at `rest`.
- `rest`: the fixed rest point = the neutral sample's coordinates (`mood_baseline = "n001"`).
- `trail`: the last 200 points after each nudge.

### 2.2 Decay, every tick

```
k = 0.5 ^ (dt / mood_half_life_s)           mood_half_life_s = 120
p = b + (p - b) · k
```

### 2.3 Nudge, on every successful reading `r`

```
intensity = min(1, ‖r‖ / 1.0)
w = mood_nudge · (0.5 + 0.5 · intensity)     mood_nudge = 0.5   (mood_nudge_by_intensity = true)
p = clip(p + w · (r - p), ±mood_clip)
```

A faint reading pulls at half strength, a strong one at the full nudge; the mood never jumps more than half the gap.
`sincerity_gap = dist(r, p_before)` is logged per turn: how far what she reports sits from what she holds.
Nothing acts on it yet.

### 2.4 Temperament (the baseline as slow state)

```
k2 = 0.5 ^ (min(dt, 600) / temperament_half_life_s)      temperament_half_life_s = 259200 (3 days)
b  = b · k2 + p · (1 - k2)
if dist(b, rest) > temperament_max_shift: b = rest + (b - rest) · temperament_max_shift / dist(b, rest)
                                                          temperament_max_shift = 0.6
```

The averaging step is capped at 600 s so a day switched off is not a day of feeling. The clamp is the persona's
guardrail. `reset` in the window puts the point back on the baseline but keeps the temperament.

### 2.5 The mood sentence (what the brain reads)

Each axis is worded by thresholds `-0.7, -0.25, 0.25, 0.7`:

| axis | < -0.7 | < -0.25 | < 0.25 | < 0.7 | else |
|---|---|---|---|---|---|
| v | your spirits are low | a bit flat | on an even keel | in good spirits | bright and light |
| a | very settled, almost drowsy | unhurried | neither restless nor sleepy | keyed up | buzzing with energy |
| d | you feel small | a little hesitant | on an even footing | sure of yourself | fully in charge |

The sentence: `Where you are starting from this turn: <v>, <a>, and <d>; <this just changed | it has been like this
for about N min>.` The clock counts since the **worded** mood last changed, not since the last nudge.

- **Trajectory**: the trail point of the last 900 s, at least 60 s old, farthest from `p`; if that distance is
  ≥ 0.5: `Not long ago (about N min ago) <past-tense words>; you have drifted from there since.`
- **Temperament**: per axis, if `b_k - rest_k` is beyond ±0.35: `Lately you have tended to be on the low side /
  on the sunny side, quiet / lively, unsure of yourself / self-assured.`
- It ends with `This is background, not something to repeat.` The vocabulary is deliberately unlike the emotion
  lines we want back, so the model does not echo it.

---

## 3. The brain (`brain/`)

### 3.1 The prompt, in order

```
system:   persona.md
          instructions.md
          Now: <weekday day month year HH:MM>.
          <mood sentence>
          Your face is currently showing '<label>' and has been for N s. Ask for a move only when there is a reason to change it.
          <memory block, if any memories were recalled>            (§5.4)
          SCHEMA (fixed in code)
history:  the last history_turns = 8 exchanges; older assistant turns as {"reply": ...} only,
          the most recent assistant turn as {"emotion", "move", "reply"}   (one line to remember, not eight to imitate)
user:     the new text
```

### 3.2 The schema (three channels, this order)

```json
{"emotion": "<one or two full sentences, first person, feelings not facts>",
 "move":    {"description": "<what the face and hands do, physical language; \"\" = leave the face>", "hold": "turn"},
 "reply":   "<two to four sentences>"}
```

Parsing is lenient (code fences, trailing commas, a string for `move`, the v1 `perform` key). Failure is soft:
missing emotion → the mood only decays; missing or empty move → the body drifts; output that is not JSON → the
whole text is the reply.

### 3.3 Generation (Qwen backend)

- 4-bit NF4 with double quantisation, bf16 compute; the output head and embeddings stay bf16
  (1.2 GB each on the 8B, the reason the 8B fills an 8 GB card; `model_id = Qwen/Qwen3-4B`).
- sampling `temperature = 0.7`, `top_k = 20`, `top_p = 0.8`, thinking off, `max_new_tokens = 400`;
  a fresh random seed per call (`seed = 0`), because a fixed seed reproduced a whole turn verbatim.
- on CUDA out-of-memory: free the cache and retry once with half the new tokens; counted in the gauge.
- stats per call: prompt tokens, new tokens, seconds, VRAM allocated / reserved / peak, and the KV cost
  `bytes_per_token = 2 · layers · kv_heads · head_dim · 2` (≈ 144 KB on Qwen3-8B; the gauge budget
  `context_budget_tokens = 8192` is display only).

### 3.4 The echo backend

No model: canned emotion + move + reply keyed on words in the user text. Used with `config.dev.json`, which points
memory, state and log at `data_dev/` so it never touches her real memories.

### 3.5 Idle prompt

After `idle_prompt_seconds = 180` of silence, at most every `idle_prompt_min_gap = 300` s, the brain receives an
`[internal event]` with the time, the session length and whether the window is visible; she usually leaves the
face alone and replies `""`. Internal turns (idle prompts and the wake-up) are shown as "(to herself)", are answered
with the history in view but are **not appended to it**, run **no recall**, and idle turns are not written to memory
(the wake-up is). They are state, not conversation: a model that sees its own wake-up line as the last turn repeats
it for the next ten minutes. The fact of a restart lives in the time line instead (§6).

---

## 4. The body (`library/`, `graph/`, the fetch policy in `agent/loop.py`)

### 4.1 The library

`data/videos/clip_descriptions.json`: keyframes (`n001`…, permanent ids, a `label`, `aliases`, a description with
`expression / symbols / hands` clauses, optional `feeling`) and clips. A clip is `loop__n003__v1`,
`edge__n001__n003__v1` or `act__n001__wave__v1` (kinds loop | edge | act): first frame = `start` still, last
frame = `end` still. Each clip has `index_docs`, the only texts embedded: motion + end expression, the end
expression alone, and an alias list.

### 4.2 The clip RAG (`library/index.py`)

Chroma collection `clips`, cosine space, bge-small on CPU. A query embeds the move text, fetches the 30 nearest
documents and keeps, per **end node**, the best document: `similarity = 1 - cosine_distance`.

### 4.3 The matcher (`library/matcher.py`)

```
hits = nodes ranked by best-document similarity
top < tau                      → miss (tau = 0.70 for bge-small; re-pick with scripts/check_retrieval.py)
pool = hits within spread of top   (spread = 0.02)
choice ~ softmax((sim - top) / sample_temperature)   (sample_temperature = 0.01; a near-tie can go either way)
```

### 4.4 The graph (`graph/graph.py`)

Nodes = stills, directed edges = clips, each node's coordinates = its seed point in the space. Routing is
Dijkstra on clip duration with `max_hops = 3`. A target with an idle loop is a place to **dwell**; without one the
plan is **perform and return** to the hub (`hub_node = n001`). `Plan.path()` lists the nodes visited. The window
reports its true position after every clip, so the graph never drifts from what is on screen.

### 4.5 The fetch policy (per turn)

```
near = node nearest the mood
if move given:
    m = matcher.match(move.description)
    if hit:
        cost = dist(mood, coords[m.node])
        cost <= move_budget (2.0) → go to m.node                       reason "hit"
        else                     → go to near, log requested node       reason "out_of_reach"
    elif miss:                   → go to near, log nearest candidate    reason "miss"
else:                            → go to near                           reason "drift"
already there → reason suffixed "/stay", nothing played
mask = dist(mood, coords[body node])        (a face that differs from the feeling, on purpose or not)
```

Nothing is generated on a miss in v3 (no wanted list yet): the miss is logged with the closest candidate and its
similarity, and the body follows the mood.

### 4.6 Silence drift

Every `drift_check_seconds = 60`, when not busy:
`dist(mood, pos) - dist(mood, near) > drift_hysteresis (0.2)` → walk to `near`, logged as `drift` with the mask.
The mood decays toward the rest sample, so after a reaction she visibly settles back to neutral over a few minutes.

### 4.7 The player (window)

Plays the plan's clips in order, then loops the idle clip of the end node; a plan arriving mid-clip queues after it;
a queued clip whose start is not the current node is skipped. The travelled path is drawn on the graph as fading
arrows; the body's node is ringed; a dashed line to the mood shows the mask.

---

## 5. The memory RAG (`memory/`)

### 5.1 The record (one per user turn and per wake-up)

```
id        m<epoch ms>
document  "User said: <user>\nShe replied: <reply>\nShe felt: <emotion line>"     (what is embedded)
metadata  t, when (text), user, reply, emotion, v a d (mood after the turn), rv ra rd (the reading),
          has_reading, body (node), source (user | wake), surprise (0, no perception yet), refs (0), session
```

Store: Chroma collection `memories` in `data/memory`, cosine, same embedder as the clip RAG, separate store on
purpose (different filters, writers and pruning; they link by time only).

### 5.2 Retrieval, adaptive rather than top-k

```
query  = the user's new text          (user turns only: never an idle prompt or the wake-up)
cands  = memory_candidates (8) nearest, minus the turns still in the verbatim history (history_memory_ids)
top    = best similarity
keep   = cands with  sim >= memory_floor (0.66)  and  sim >= top - margin
         margin = memory_margin (0.12), halved when mood_v > 0.3 and the memory's v < -0.4   (congruence gate)
         (bge-small scores unrelated short texts 0.62-0.66, so the floor sits just above that)
result = the best memory_max (5) of keep, ordered chronologically
```

A question about the past pulls several; small talk pulls none. Each recalled memory gets `refs += 1` and
`last_ref_t`, the usefulness signal the archival reward will train on later.

### 5.3 The congruence gate

When she is fine, an old hurt has to match twice as closely to come back, so recall does not re-inject feelings
that were not asked about (the report's mirror-loop trap). A direct question still brings it back.

### 5.4 The memory block (in the prompt)

```
Things you remember from earlier conversations (when, and how you felt at the time):
- <N h ago> (<Thu 10 Sep 2026 11:51>): The user said: "...". You answered along the lines of: <gist>. At the time
  you felt: "...". Your mood then was <words>; time has passed since.
These are memories, not the current conversation. What you felt then is history: it can inform you, it does not
decide how you feel now. Do not repeat the words of an earlier answer; this moment is new.
```

The gist is the first clause of her reply, cut at 90 characters: a quoted reply is an invitation to say it again.

```
```

Relative times: `just now`, `N min ago`, `an hour ago`, `N h ago`, `yesterday`, `N days ago`.

---

## 6. Time

- The prompt opens with `Now: <weekday day month year HH:MM>.` For 15 min after a restart it adds
  `You were switched back on N min ago, after being off for M.`
- Every episode record carries `t` (epoch) and `when` (local text).
- Memories carry both, and are recalled with the relative time.
- The mood sentence says how long the worded mood has been the same, and how long ago the last excursion was.
- On restart she is told when she was switched off and how long ago (§7.3).

---

## 7. Persistence (`agent/state.py`)

### 7.1 What is saved (`data/kb/state.json`)

History (with emotion and move per turn and the memory id of each turn), the mood point, the baseline
(temperament), the mood's clock (`last_t`, `since_change`), the last 50 trail points, the body's node, the arrival
time, the session id, `saved_at`. Not the KV cache: the prompt is rebuilt every turn, so a saved cache would be
invalid on the next turn anyway.

### 7.2 When

After every turn, twice (before and after the memory write, so a crash in the memory store loses at most one
memory), and at server shutdown. The write is atomic (temp file, then replace).

### 7.3 Restore and wake-up

On start the state is loaded; the mood's clock is set to the save time, so the first tick applies the decay for the
whole gap (a day away settles her on the baseline, and the temperament averaging step is capped at 10 min). If the
gap is at least `wake_min_gap_s = 60`, she receives:

```
[internal event] You were switched off on <when> and have just been switched back on; that was <N ago>.
It is now <when>. What you remember of before is in your history and your memories. React as yourself ...
```

and her reaction is the first thing in the window. `run_qwen.cmd` restarts the server if it dies, so a crash
becomes a short pause and a wake-up.

---

## 8. The episode log (`data/kb/episodes.jsonl`)

One JSON line per record, `t` + `when` + `kind`:

| kind | when | notable fields |
|---|---|---|
| `start` | server start | seed table with coordinates, hand-value correlation, memories count, restored?, away seconds, session |
| `turn` | every brain call | source (user / idle / wake), user, reply, emotion, move, parsed, method, coords, hits, nudge, mood before/after, mood line, sincerity_gap, baseline, body decision (reason, target, similarity, cost, plan, path), top 3 clip hits, pos, mask, recalled memories, gpu stats |
| `memory` | after the record is written | memory id |
| `mood` | every 60 s | point, nearest sample, distance, pos, mask |
| `drift` | silence drift | from, to, mood, mask |
| `probe` | window probe | text, coords, method, applied |
| `index_built` | clip index (re)built | documents, clips |
| `error` | any handler exception | event, error |

`data/kb/crash.log` receives every thread's stack if the process dies natively, and the restart lines from
`run_qwen.cmd`.

---

## 9. The window (`renderer/static/window.html`, port 8767)

- **Video** (left): the body. Under it: the node, the clip playing, the queue, why the body moved, and `go to`
  buttons per node.
- **3D graph** (middle): seed samples (plum), mood (teal, with trail), the last reading (hollow teal ring), the body's
  node (orange ring) with the dashed mask line, the travelled path (orange arrows, fading), the temperament rest
  point (teal diamond), a probe (grey ring). Drag to rotate, wheel to zoom, double-click to reset.
- **Chat** (right): header with `dev` toggle, sliders (mood half-life, nudge, temperature, tau, budget), `reset`,
  `rebuild index`; gauges (context tokens / KV estimate, VRAM reserved / peak); the log; the reading table; boxes
  `recall` (query her memories), `move` (try a physical text against the clip RAG), `probe` (score an emotional
  sentence, optionally apply it), and `say`.
- `dev` off hides everything but her lines, yours, the video and the graph. Remembered by the browser.

Messages from the server: `hello, status, reply, play, mood, gpu, probe, match, recall, log`. From the window:
`user`, `command` (`reset, set, probe, match, recall, goto, rebuild_index`), `window` (`position, visibility`).

---

## 10. Knobs (config.json), with defaults

| knob | default | meaning |
|---|---|---|
| `model_id` | Qwen/Qwen3-4B | the brain (8B fills the 8 GB card at ~2.5k-token prompts) |
| `temperature`, `max_new_tokens`, `history_turns`, `seed` | 0.7, 400, 8, 0 | generation and the verbatim history |
| `min_hits` | 2 | feeling words needed for a lexicon reading |
| `placement_fallback` | true | embed against the seed when the lexicon fails |
| `mood_baseline` | n001 | the rest sample, or "center" of the box |
| `mood_half_life_s`, `mood_nudge`, `mood_nudge_by_intensity`, `mood_clip` | 120, 0.5, true, 1.5 | mood dynamics |
| `temperament_half_life_s`, `temperament_max_shift` | 259200, 0.6 | the slow baseline and its clamp |
| `recall_congruence` | true | tighter margin for old hurts when she is fine |
| `memory_floor`, `memory_margin`, `memory_max`, `memory_candidates` | 0.66, 0.12, 5, 8 | adaptive recall (user turns only) |
| `wake_min_gap_s` | 60 | gap that triggers the wake-up event |
| `tau`, `spread`, `sample_temperature` | 0.70, 0.02, 0.01 | clip RAG hit threshold and near-tie sampling |
| `hub_node`, `move_budget` | n001, 2.0 | the graph's hub; cost limit of a move |
| `drift_check_seconds`, `drift_hysteresis` | 60, 0.2 | silence drift |
| `idle_prompt_seconds`, `idle_prompt_min_gap` | 180, 300 | idle self-prompt |
| `context_budget_tokens` | 8192 | gauge only |
| paths | `data/...` | relative to config.json: copy the folder, get a new agent |

---

## 11. Known limits and open questions

- The lexicon is bag-of-words: figurative lines ("my heart did a flip") read as nothing; narrative content words
  in the why-clause still carry lexicon valence (they count 0.4).
- There is no mild-surprise sample: "eyes widen slightly" lands on full shocked. That is the coverage gap the
  design paints later (nodes for coverage, never on request).
- The sincerity gap is logged, not used; the self VLM that would measure it from the face does not exist yet.
- Recalled memories are chosen by similarity to the user's text only; `refs`, surprise and the archival reward
  are recorded for later, not used for ranking.
- A silent process death happened once right after a memory write on the 4B run; cause unknown, see
  `data/kb/crash.log` next time. The restart loop and the save order make it a short pause.
- Qwen3-4B repeats its previous emotion line more than the 8B did.
