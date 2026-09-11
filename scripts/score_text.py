"""Score emotional sentences with the lexicon and place them in the standardised space.
   python scripts/score_text.py --config config.json "surprised and pleased" "not sad at all"
   python scripts/score_text.py            # interactive: one sentence per line, empty line to quit"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embodied3.config import Config
from embodied3.affect import Lexicon, Space

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.json"); ap.add_argument("texts", nargs="*")
a = ap.parse_args()
cfg = Config.load(a.config)
lex = Lexicon.load(cfg.lexicon_file, cfg.min_hits)
sp = Space.from_descriptions(lex, cfg.descriptions_file, clip=cfg.mood_clip)


def show(text):
    r, c = sp.read(text)
    near, d = sp.nearest(c) if c else (None, None)
    print(f"  ok={r.ok} hits={r.hits} raw={tuple(round(x, 2) for x in r.raw) if r.raw else None} "
          f"std={tuple(round(x, 2) for x in c) if c else None} nearest={near.label if near else None}{f' d={d:.2f}' if d is not None else ''}")
    for t in r.terms:
        v = t["vec"]
        print(f"     {t['term']:16s} {v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}  w={t['weight']:.2f} {'hit' if t['hit'] else '   '} {' '.join(t['flags'])}")


if a.texts:
    for t in a.texts:
        print(repr(t)); show(t)
else:
    for line in sys.stdin:
        if not line.strip():
            break
        show(line.strip())
