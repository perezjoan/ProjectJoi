# Adding a keyframe by hand (v3)

Everything lives in `data/`. The next free id is the next number after the highest `nNNN` in
`data/videos/clip_descriptions.json`; ids are never reused, and the word for the node is only its `label`.
Example below: `n010`, "mildly surprised".

## 1. The still

`data/keyframes/n010.png`, 512×512, same character, framing and background as the others. Make it with an image
edit starting from `n001.png` (or from the nearest existing keyframe). Describe the change in axis deltas:

| axis delta | face and body |
|---|---|
| valence up / down | mouth corners up / down, inner brow ends relaxed / raised, cheeks lifted / flat |
| arousal up / down | eyes and mouth more / less open, posture energised / settled, hands active / still |
| dominance up / down | head level and forward / tilted down and away, direct / averted gaze, shoulders open / drawn in |

Keep the exact pixels: the png you save is the first and last frame you hand to the video model, never a rescaled
copy. Otherwise the seams show.

## 2. Three clips

16 fps, both ends pinned to existing stills:

```
data/videos/edge__n001__n010__v1.mp4   first = n001.png, last = n010.png,  49 frames (3 s)
data/videos/loop__n010__v1.mp4         first = last = n010.png,            49 frames (almost still: breathing, one blink)
data/videos/edge__n010__n001__v1.mp4   first = n010.png, last = n001.png,  25 frames (1.5 s)
```

Without the loop she cannot dwell there; without the exit she cannot leave. Add `__v2`, `__v3` for variants of the
same pair; the renderer rotates through them.

## 3. The JSON entries

In `data/videos/clip_descriptions.json`. Two different readers use them, so keep two vocabularies:

- the **lexicon** reads the keyframe's `feeling` (emotional words) to place the node in the space;
- the **clip RAG** reads the clips' `index_docs` (physical words) to find the node from a move.

Keyframe entry:

```json
{"node": "n010", "file": "n010.png", "label": "mildly surprised",
 "description": "subject: a cartoon girl with black pigtails, pink hair clip and pink striped top, bust-up on a white background. expression: eyebrows lifted, eyes a little wider, lips parted. symbols: none. hands: down.",
 "aliases": "mildly surprised, taken aback, oh, blink, small surprise",
 "feeling": "surprised, curious, alert, a little taken aback, interested"}
```

Keep the `expression: ... . symbols: ... . hands: ...` shape in `description`; the scripts parse it. `valence` and
`arousal` are optional in v3 (the lexicon places the node); if you add them they are only used as a sanity check.

Clip entries (the loop and the return edge follow the same shape):

```json
{"id": "edge__n001__n010__v1", "file": "edge__n001__n010__v1.mp4", "kind": "edge", "start": "n001", "end": "n010",
 "duration_s": 3.06, "origin": "recorded",
 "description": "start expression: calm face, small flat mouth. motion: eyebrows lift, eyes widen a little, lips part. end expression: eyebrows lifted, eyes a little wider, lips parted.",
 "fields": {"start_expression": "calm face, small flat mouth", "motion": "eyebrows lift, eyes widen a little, lips part", "end_expression": "eyebrows lifted, eyes a little wider, lips parted"},
 "aliases": "eyebrows lift, eyes widen slightly, head tilts forward, oh, small surprise, taken aback",
 "index_docs": ["eyebrows lift, eyes widen a little, lips part; ends eyebrows lifted, eyes a little wider",
                "eyebrows lifted, eyes a little wider, lips parted",
                "eyebrows lift, eyes widen slightly, head tilts forward, oh, small surprise, taken aback"]}
```

```json
{"id": "loop__n010__v1", "file": "loop__n010__v1.mp4", "kind": "loop", "start": "n010", "end": "n010",
 "duration_s": 3.06, "origin": "recorded",
 "description": "start expression: eyebrows lifted, eyes a little wider, lips parted. motion: almost still; subtle breathing, a slow blink, tiny head sway; expression held. end expression: eyebrows lifted, eyes a little wider, lips parted.",
 "fields": {"start_expression": "eyebrows lifted, eyes a little wider, lips parted", "motion": "almost still; subtle breathing, a slow blink; expression held", "end_expression": "eyebrows lifted, eyes a little wider, lips parted"},
 "aliases": "stay mildly surprised, keep the surprised look, hold, remain",
 "index_docs": ["almost still, breathing, blink; ends eyebrows lifted, eyes a little wider, lips parted",
                "eyebrows lifted, eyes a little wider, lips parted",
                "stay mildly surprised, keep the surprised look, hold, remain, taken aback"]}
```

```json
{"id": "edge__n010__n001__v1", "file": "edge__n010__n001__v1.mp4", "kind": "edge", "start": "n010", "end": "n001",
 "duration_s": 1.56, "origin": "recorded",
 "description": "start expression: eyebrows lifted, eyes a little wider, lips parted. motion: features relax back to a calm neutral face, brows settle, lips close. end expression: calm face, small flat mouth.",
 "fields": {"start_expression": "eyebrows lifted, eyes a little wider, lips parted", "motion": "features relax back to a calm neutral face, brows settle, lips close", "end_expression": "calm face, small flat mouth"},
 "aliases": "calm down, relax, back to neutral, settle",
 "index_docs": ["features relax back to a calm neutral face, brows settle, lips close; ends calm face",
                "calm face, small flat mouth",
                "calm down, relax, back to neutral, settle, composed"]}
```

`index_docs` is the only thing embedded: three short texts per clip, no boilerplate about the character. The
alias document is what makes retrieval work, in the words the brain actually writes in moves.

## 4. Rebuild and check

From the v3 folder:

```
python scripts/build_index.py          # re-embeds clip_descriptions.json into data/kb
python scripts/seed_coords.py          # where the new sample landed in the space, next to the others
python scripts/check_retrieval.py      # does a mild-surprise move now beat the shocked clip; does tau still separate
python scripts/travelled_edges.py      # which node pairs she actually travels (for direct edges, §6)
```

The space is standardised to the seed's min and max per axis. A sample **inside** the current range changes
nothing else. A sample more extreme than any existing one on an axis stretches the whole frame: every coordinate
shifts a little, including the saved mood and temperament in `data/kb/state.json`. If that happens and you want a
clean start, delete `state.json`.

Add a test line for the new node in `scripts/check_retrieval.py` (`TESTS`) so the check keeps meaning something.

## 5. Try it

Restart with `run_qwen.cmd` (or `run_echo.cmd` for the UI only; it uses `data_dev/`). Use the `move` box to test
phrases against the clip RAG, and the `go to` button to watch the three clips. The `match` result shows the
similarity of every candidate node; tau (0.70) is the hit line.

## 6. Direct keyframe-to-keyframe edges

Yes, but only for the pairs she actually travels. Every move routes through neutral today: the router takes the
shortest path by duration with at most three hops. Via neutral costs about 4.5 s and reads as "reset, then react";
a direct edge costs 3 s and reads as one feeling turning into another, which is what a face does.

Clips are directed, so a pair needs two files (`edge__n004__n008__v1.mp4` and `edge__n008__n004__v1.mp4`), and
all pairs would be 72 clips for nine nodes. Add the few that matter: neighbours in the space that show up
consecutively in the log. `python scripts/travelled_edges.py` ranks them from `data/kb/episodes.jsonl`, marks
which already have a direct clip, and prints the file names to make. Nothing else changes: drop the file in, add
its entry (same shape as above, `start` and `end` being the two nodes), rebuild the index, and the router uses it.
