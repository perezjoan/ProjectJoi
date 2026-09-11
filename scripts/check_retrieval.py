"""Run test move texts against the clip RAG and print correct vs wrong similarities, to pick tau.
   python scripts/check_retrieval.py --config config.json"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embodied3.config import Config
from embodied3.library import ClipIndex, load_library

TESTS = [
 ("one eyebrow up, head tilts, eyes narrow", "puzzled"), ("narrow your eyes and wonder", "puzzled"),
 ("eyes squeeze shut, laugh bursts out, head tips back", "laughing"), ("giggle, shoulders shaking", "laughing"),
 ("brows pull down, hard stare, jaw sets", "angry"), ("glare, lips pressed thin", "angry"),
 ("one eye closes, a knowing smile", "wink"), ("a warm smile spreads", "happy"), ("beam with a big smile", "happy"),
 ("eyes well up, tears stream", "crying"), ("sob loudly", "crying"), ("eyes widen sharply, lips part", "shocked"),
 ("startled, chin draws back", "shocked"), ("brows knit, gaze drops, uneasy", "worried"), ("frown nervously", "worried"),
 ("features relax to a calm face", "neutral"), ("stay calm and still", "neutral"),
 ("hold up a cartoon cat with both hands", None), ("nod slowly", None), ("wave goodbye", None),
]
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.json"); a = ap.parse_args()
cfg = Config.load(a.config)
idx = ClipIndex(cfg.index_dir, cfg.embedder)
lib = load_library(cfg.videos_dir, cfg.keyframes_dir, cfg.descriptions_file)
label = lambda n: (lib.keyframes[n].label or n) if n in lib.keyframes else n
good, bad, none = [], [], []
for req, want in TESTS:
    h = idx.query_nodes(req, k=2)
    top, second = h[0], (h[1] if len(h) > 1 else None)
    ok = want is None or label(top["node"]) == want
    margin = top["similarity"] - (second["similarity"] if second else 0)
    (none if want is None else good if ok else bad).append(top["similarity"])
    print(f"{'OK ' if ok else 'BAD'} {top['similarity']:.4f} (margin {margin:.4f})  {req:50s} -> {label(top['node']):10s} want {want}")
if good: print(f"\ncorrect: min {min(good):.4f}")
if bad:  print(f"wrong:   max {max(bad):.4f}")
if none: print(f"should-miss requests scored: {', '.join(f'{s:.4f}' for s in none)}  -> tau must sit above these")
print(f"tau now: {cfg.tau}")
