# Project Joi

![version](https://img.shields.io/badge/version-3.0.1-8e4b8e)
![python](https://img.shields.io/badge/python-3.10%2B-3776ab)
![brain](https://img.shields.io/badge/brain-Qwen3--4B%20%7C%204--bit-1a9c8c)
![gpu](https://img.shields.io/badge/runs%20on-8%20GB%20VRAM-e07b28)
![local](https://img.shields.io/badge/cloud-none-222)
![tests](https://img.shields.io/badge/tests-44%20passing-2ea44f)

A local companion with a **mood**, a **body** made of short video clips, and a **memory** with timestamps. She runs
as a long-lived process on your own machine, on one consumer GPU, with nothing sent anywhere.

Her persona is Joi, the companion from *Blade Runner 2049*. This is a research side project and a fan homage; the
likeness and the character belong to their owners, and nothing here is for sale.

## What it is

Most chatbots pick a face per message. Joi does not. Each turn her language model says three things, in this order:
how she **feels** about the exchange, what her **face** does, and what she **replies**. The feeling is turned into a
point in a three-dimensional emotional space by a lexicon, not by the model. That point moves her **mood**, a state
that decays over minutes and that she reads back at the start of every turn, so she reports against something she does
not control. Her face is a small **graph of video clips**: still frames (painted samples of the same space) joined by
transitions, and the body walks between them, or drifts toward whatever the mood is nearest to when she has nothing
in particular to show. Every interaction is written to a **memory store** with the time and the mood she was in, and
recalled when it is relevant. When you switch her off, her self is saved; when you switch her back on she is told
how long she was away, and reacts.

Feeling and face are kept apart on purpose. The distance between them is measured and logged: a brave face is a
valid move, and the gap is information, not an error.

## How it works, briefly

- **The space.** Three axes from the NRC VAD lexicon: valence (unpleasant to pleasant), arousal (calm to energised),
  dominance (submissive to in control). Every emotional text in the system is scored by one function: longest-match
  lookup of 54 800 terms, negation flips valence, intensifiers and diminishers scale, hedges ("trying to stay calm")
  are discounted, the first clause outweighs the "because" clause, and each word counts in proportion to how far it
  is from neutral. Coordinates are standardised so the nine seed expressions span the space.
- **The mood.** A point that decays toward a rest point with a two-minute half-life and is nudged toward each
  reading by up to half the gap, in proportion to how strong the reading is. The rest point itself is slow state:
  a three-day average of where the mood has been, clamped, so weeks of conversation shift her temperament without
  touching the persona file.
- **The brain.** Qwen3-4B in 4-bit through Hugging Face transformers and bitsandbytes. One JSON object per turn,
  `emotion`, `move`, `reply`, parsed leniently: a missing move means the body drifts, output that is not JSON is a
  plain reply. The prompt carries the persona, the rules, the clock, the mood in words, what her face is showing, and
  the memories that match what you just said. Everything so far runs on a laptop **RTX 5060 with 8 GB of VRAM**;
  the 8B model fits but runs out of memory on long prompts, the 4B leaves room. A larger brain is the plan.
- **The body.** Nine keyframes (neutral, angry, puzzled, happy, crying, shocked, worried, laughing, wink), a loop
  clip per keyframe and a transition each way to neutral, plus direct transitions between pairs she travels often.
  The move text is matched against clip descriptions in a vector index (bge-small, on the CPU); a hit above a
  similarity threshold sends her there if the node is within budget of her mood, otherwise she drifts. The clips
  are a cache, not a vocabulary: new expressions are added where the mood keeps going and no sample is near.
- **The memory.** One record per interaction in a separate vector store: what you said, what she answered, how she
  felt, the mood point, the body, the time. Retrieval is adaptive rather than top-k: only memories above an absolute
  floor and close to the best one come back, so small talk pulls nothing and a real question pulls several. Each
  recall bumps a counter, the ground truth for a future archival policy.
- **Persistence.** History, mood, temperament, body and timestamps are saved after every turn and at shutdown. The
  KV cache is not saved: the prompt is rebuilt each turn, so a cache would be stale anyway.
- **The window.** Her face on the left, the 3D space in the middle with the mood, its trail, the body's node and the
  existing clips drawn as edges, the chat on the right. A `dev` toggle shows the lexicon reading of every line, the
  body decision, the memories recalled, a KV and VRAM gauge, and boxes to probe the lexicon, the clip index and the
  memory store without waking the model.

`HOW_IT_WORKS.md` has every mechanism with its formulas and knobs. `ADDING_A_KEYFRAME.md` explains how to add an
expression by hand, and `VIDEO_PROMPTS.md` holds the prompts the clips were generated with.

## Installation

Windows is what this was built on; nothing is Windows-specific except the two `.cmd` launchers.

1. **Python environment** (conda, Python 3.11):

   ```
   conda create -n embodied python=3.11
   conda activate embodied
   ```

2. **PyTorch with CUDA**, for your card. This project uses CUDA 12.8 (Blackwell):

   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

3. **The rest**:

   ```
   pip install -r requirements.txt
   ```

4. **The lexicon** (not included, see below): copy `NRC-VAD-Lexicon-v2.1.txt` into `data/lexicon/`.

5. **First start**, from the project folder:

   ```
   python -m embodied3.server
   ```

   The clip index is built on the first start. The language model (about 2.5 GB) and the embedder are downloaded
   from Hugging Face on the first message. Then open http://127.0.0.1:8768.

`run_qwen.cmd` does step 5 and restarts the server if it ever dies. `run_echo.cmd` starts a model-free stand-in on
a scratch data folder, for working on the window without a GPU. `pytest -q` runs 44 model-free tests.

To use a different brain, change `model_id` in `config.json` (any Qwen3 chat model works; larger ones need more
VRAM). To make a different character, copy the folder: edit `data/persona.md` and `data/instructions.md`, replace
the keyframes and clips keeping the file naming, edit `data/videos/clip_descriptions.json`, delete `data/kb` and
`data/memory`, change the port.

## The lexicon

Joi needs the **NRC Valence, Arousal, and Dominance Lexicon v2.1** by Saif M. Mohammad. It is free for research and
non-commercial use under its own terms and is therefore not part of this repository.

- Download: https://saifmohammad.com/WebPages/nrc-vad.html
- Place the top-level file of the archive here: `data/lexicon/NRC-VAD-Lexicon-v2.1.txt`

The server refuses to start, with the same instructions, until the file is there. `data/lexicon/README.md` has the
citation.

## Layout

```
config.json          all paths relative to this file; model, thresholds, half-lives, port
data/
  persona.md         who she is             instructions.md   how she reports (emotion, move, reply)
  keyframes/         n001.png ... the nine stills
  videos/            loop__ / edge__ clips and clip_descriptions.json (the library, and the seed of the space)
  lexicon/           put the NRC VAD file here
  kb/, memory/       created at run time: clip index, episode log, her saved self, her memories (git-ignored)
embodied3/
  affect/            lexicon scorer, the space, the mood, the placement fallback
  brain/             schema, prompt assembly, Qwen and echo backends
  library/ graph/    clip records, clip index, matcher; the keyframe graph and its routing
  memory/            the memory store and how memories are written into the prompt
  agent/             the event loop, the body policy, persistence
  renderer/          the window (a single HTML file)
  server.py          FastAPI + websocket
scripts/             build_index.py, check_retrieval.py, seed_coords.py, score_text.py, travelled_edges.py
tests/               pytest, no models needed
```

## Credits

- Lexicon: Saif M. Mohammad, *NRC Valence, Arousal, and Dominance Lexicon* (ACL 2018, v2.1 2025).
- Brain: Qwen3 by the Qwen team, Alibaba Cloud. Embedder: BAAI bge-small-en-v1.5. Vector store: Chroma.
- Character: Joi, *Blade Runner 2049* (Warner Bros. / Alcon). Fan project, non-commercial.

## Roadmap

Painting new expressions where the mood keeps going and no sample exists; direct transitions grown from the
travelled pairs; a larger brain; perception (webcam, microphone) so her reactions are measured against the world
rather than her own words; consolidation while idle.
