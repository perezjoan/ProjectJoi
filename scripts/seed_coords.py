"""Print the seed keyframes' lexicon coordinates next to the hand-set values, and the correlation.
   python scripts/seed_coords.py --config config.json"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embodied3.config import Config
from embodied3.affect import Lexicon, Space

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.json"); a = ap.parse_args()
cfg = Config.load(a.config)
lex = Lexicon.load(cfg.lexicon_file, cfg.min_hits)
sp = Space.from_descriptions(lex, cfg.descriptions_file, clip=cfg.mood_clip)
print(f"lexicon: {len(lex.entries)} terms (up to {lex.max_n} words). space center {tuple(round(c, 3) for c in sp.center)}, "
      f"half-range {tuple(round(s, 3) for s in sp.scale)}")
print(f"{'node':5s} {'label':10s} hits   valence  arousal  dominance   | hand v   hand a")
for s in sp.samples:
    c = s.coords or (float('nan'),) * 3
    print(f"{s.node:5s} {s.label:10s} {s.hits:4d}   {c[0]:+7.2f}  {c[1]:+7.2f}  {c[2]:+7.2f}     | {s.hand.get('v', float('nan')):+6.2f}  {s.hand.get('a', float('nan')):6.2f}")
corr = sp.hand_correlation()
print("Pearson r vs hand values: " + ", ".join(f"{k} {v:.3f}" for k, v in corr.items()))
print(f"baseline (n001): {tuple(round(x, 2) for x in sp.baseline)}")
