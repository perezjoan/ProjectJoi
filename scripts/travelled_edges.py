"""Which node pairs does she actually travel? Ranks (from -> to) transitions from the episode log so you know which
direct edges are worth recording (see ADDING_A_KEYFRAME.md, section 6).

    python scripts/travelled_edges.py --config config.json [--days 30] [--top 15]

Counts every body movement: brain turns (hit / out_of_reach / miss / drift) and silence drifts. Pairs that already
have a direct clip in either direction are marked; the rest are printed with the file names to make. Also shows how
often each move went through the hub, which is the cost a direct edge would save.
"""
import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embodied3.config import Config
from embodied3.library import load_library

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.json")
ap.add_argument("--days", type=float, default=0, help="only the last N days (0 = all)")
ap.add_argument("--top", type=int, default=15)
a = ap.parse_args()
cfg = Config.load(a.config)
lib = load_library(cfg.videos_dir, cfg.keyframes_dir, cfg.descriptions_file)
label = lambda n: (lib.keyframes[n].label or n) if n in lib.keyframes else n
direct = {(c.start, c.end) for c in lib.clips.values() if c.kind == "edge" and c.start != c.end}

since = time.time() - a.days * 86400 if a.days > 0 else 0
pairs: Counter = Counter()
reasons: dict = defaultdict(Counter)
hops: Counter = Counter()          # total clips played for the pair (brain turns only; drifts carry no path)
hop_moves: Counter = Counter()     # how many of the pair's movements had a path
n_moves = 0
for line in Path(cfg.episode_log).read_text(encoding="utf-8").splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("t", 0) < since:
        continue
    if r["kind"] == "turn":
        b = r.get("body") or {}
        plan = b.get("plan")
        if not plan or not plan.get("path") or len(plan["path"]) < 2:
            continue
        path = plan["path"]
        frm, to, reason = path[0], path[-1], b.get("reason", "?")
    elif r["kind"] == "drift":
        frm, to, reason, path = r.get("frm"), r.get("to"), "silence drift", None
    else:
        continue
    if not frm or not to or frm == to:
        continue
    n_moves += 1
    pairs[(frm, to)] += 1
    reasons[(frm, to)][reason] += 1
    if path:
        hops[(frm, to)] += len(path) - 1
        hop_moves[(frm, to)] += 1

if not pairs:
    print("no movements in the log yet")
    sys.exit(0)

print(f"{n_moves} movements in {cfg.episode_log}" + (f" (last {a.days:g} days)" if a.days else ""))
print(f"{'from':10s} {'to':10s} {'n':>4s}  {'direct?':8s} {'avg hops':>8s}  reasons")
for (frm, to), n in pairs.most_common(a.top):
    has = (frm, to) in direct
    back = (to, frm) in direct
    mark = "yes" if has else ("back only" if back else "no")
    avg = hops[(frm, to)] / hop_moves[(frm, to)] if hop_moves[(frm, to)] else 0
    why = ", ".join(f"{k} {v}" for k, v in reasons[(frm, to)].most_common())
    print(f"{label(frm):10s} {label(to):10s} {n:4d}  {mark:8s} {avg:8.1f}  {why}")

# unordered pairs, both directions summed; the hub's spokes exist by construction
both: Counter = Counter()
for (frm, to), n in pairs.items():
    if cfg.hub_node in (frm, to):
        continue
    both[frozenset((frm, to))] += n
todo = [(sorted(p), n) for p, n in both.most_common() if not ({(tuple(sorted(p))), (tuple(sorted(p))[::-1])} <= direct)]
if todo:
    print("\nDirect edges worth recording (both directions; first/last frames pinned to the stills), most travelled first:")
    for (x, y), n in todo[: a.top]:
        print(f"  {n:3d}x  {label(x)} <-> {label(y)}")
        for frm, to in ((x, y), (y, x)):
            if (frm, to) not in direct:
                print(f"         videos/edge__{frm}__{to}__v1.mp4   first = keyframes/{frm}.png, last = keyframes/{to}.png,  49 frames")
else:
    print("\nEvery travelled pair already has its direct edges, or goes through the hub by design.")
