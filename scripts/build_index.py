"""Build (or rebuild) the clip RAG from clip_descriptions.json.   python scripts/build_index.py --config config.json"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embodied3.config import Config
from embodied3.library import load_library, ClipIndex

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.json"); a = ap.parse_args()
cfg = Config.load(a.config)
lib = load_library(cfg.videos_dir, cfg.keyframes_dir, cfg.descriptions_file)
missing = lib.missing_files()
print(f"{len(lib.clips)} clips, {len(lib.keyframes)} keyframes" + (f", MISSING files: {missing}" if missing else ""))
n = ClipIndex(cfg.index_dir, cfg.embedder).build(lib)
print(f"indexed {n} documents into {cfg.index_dir}")
