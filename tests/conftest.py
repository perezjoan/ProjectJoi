import json, sys
from pathlib import Path
import numpy as np, pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAKE = {
    "happy": (0.9, 0.5, 0.4), "glad": (0.8, 0.4, 0.4), "sad": (-0.8, -0.3, -0.6), "angry": (-0.7, 0.7, 0.3),
    "calm": (0.6, -0.9, -0.3), "afraid": (-0.7, 0.6, -0.7), "tired": (-0.6, -0.5, -0.4), "curious": (0.3, 0.3, 0.0),
    "puzzled": (-0.5, 0.0, -0.4), "shocked": (-0.4, 0.8, -0.2), "still": (0.0, -1.0, -0.4), "and": (0.0, 0.0, 0.0),
    "in control": (0.4, -0.1, 0.5), "hurt": (-0.6, 0.3, -0.5), "proud": (0.7, 0.3, 0.7), "fine": (0.08, -0.08, 0.04),
    "pleased": (0.8, 0.2, 0.5), "warm": (0.7, 0.2, 0.3), "cheerful": (0.9, 0.5, 0.4), "eager": (0.6, 0.6, 0.3),
    "improving": (0.6, 0.3, 0.4), "scared": (-0.8, 0.7, -0.8), "startled": (-0.3, 0.7, 0.0), "small": (0.0, -0.5, -0.8),
    "lonely": (-0.7, -0.4, -0.6), "delighted": (0.9, 0.6, 0.5), "settled": (0.5, -0.7, 0.1), "content": (0.6, -0.4, 0.2),
}


def kf(node, label, aliases, expr, v, a):
    return {"node": node, "file": f"{node}.png", "label": label, "valence": v, "arousal": a, "aliases": aliases,
            "description": f"subject: x. expression: {expr}. symbols: none. hands: down."}


def clip(id_, kind, start, end, docs, dur=3.0):
    return {"id": id_, "file": f"{id_}.mp4", "kind": kind, "start": start, "end": end, "duration_s": dur,
            "description": docs[0], "index_docs": docs, "aliases": docs[-1], "origin": "recorded"}


SEED = {"keyframes": [
    kf("n001", "neutral", "neutral, calm, fine, settled", "calm face, still", 0.0, 0.3),
    kf("n002", "angry", "angry, hurt, shocked", "furrowed brows", -0.7, 0.9),
    kf("n004", "happy", "happy, glad, proud, pleased", "big smile", 0.7, 0.5),
    kf("n005", "crying", "sad, afraid, tired, lonely", "tears", -0.8, 0.7),
    kf("n006", "shocked", "shocked, startled, scared", "very wide eyes", 0.0, 0.9),
], "clips": [
    clip("loop__n001__v1", "loop", "n001", "n001", ["almost still, breathing, blink; ends calm face", "calm face, still", "stay neutral, calm, relaxed"]),
    clip("loop__n002__v1", "loop", "n002", "n002", ["almost still; ends furrowed brows", "furrowed brows, gritted teeth", "stay angry, furious"]),
    clip("loop__n004__v1", "loop", "n004", "n004", ["almost still; ends big smile", "big smile, bright eyes", "stay happy, smiling"]),
    clip("loop__n005__v1", "loop", "n005", "n005", ["almost still; ends tears", "tears streaming, eyes shut", "stay crying, sobbing"]),
    clip("loop__n006__v1", "loop", "n006", "n006", ["almost still; ends wide eyes", "very wide eyes, mouth open", "stay shocked, startled"]),
    clip("edge__n001__n002__v1", "edge", "n001", "n002", ["brows pull down, teeth clench, fists rise; ends furrowed brows", "furrowed brows, clenched fists, gritted teeth", "angry, furious, mad, grit teeth, clench fists"]),
    clip("edge__n002__n001__v1", "edge", "n002", "n001", ["features relax back to calm; ends calm face", "calm face", "calm down, relax, neutral"], 1.5),
    clip("edge__n001__n004__v1", "edge", "n001", "n004", ["mouth opens into a broad smile, eyes brighten; ends big smile", "big smile with bright eyes", "happy, smile broadly, smiling, grin, glad"]),
    clip("edge__n004__n001__v1", "edge", "n004", "n001", ["smile closes to a small flat mouth; ends calm face", "calm face", "calm down, relax, neutral"], 1.5),
    clip("edge__n001__n005__v1", "edge", "n001", "n005", ["eyes close and tears stream, mouth trembles, fists clench; ends tears", "tears, eyes shut, mouth trembles", "cry, crying, sob, tears, eyes well up"]),
    clip("edge__n005__n001__v1", "edge", "n005", "n001", ["tears stop, features relax; ends calm face", "calm face", "calm down, relax, neutral"], 1.5),
    clip("edge__n001__n006__v1", "edge", "n001", "n006", ["eyes go wide, mouth opens, hands come up; ends very wide eyes", "very wide eyes, mouth open, hands up", "shocked, startled, gasp, eyes wide, mouth opens"]),
    clip("edge__n006__n001__v1", "edge", "n006", "n001", ["eyes return to normal, mouth closes; ends calm face", "calm face", "calm down, relax, neutral"], 1.5),
]}


class HashEmbedder:
    """Deterministic bag-of-words embedder: enough to test indexing and matching without models."""
    dim = 256

    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            v = np.zeros(self.dim)
            for w in t.lower().replace(",", " ").replace(";", " ").split():
                v[hash(w) % self.dim] += 1
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out)


@pytest.fixture
def fake_lexicon():
    from embodied3.affect import Lexicon
    return Lexicon(dict(FAKE), min_hits=3)


@pytest.fixture
def workspace(tmp_path):
    lex = tmp_path / "lex.txt"
    lex.write_text("term\tvalence\tarousal\tdominance\n" + "".join(f"{k}\t{v[0]}\t{v[1]}\t{v[2]}\n" for k, v in FAKE.items()), encoding="utf-8")
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "clip_descriptions.json").write_text(json.dumps(SEED), encoding="utf-8")
    for c in SEED["clips"]:
        (tmp_path / "videos" / c["file"]).write_bytes(b"\x00")
    (tmp_path / "keyframes").mkdir()
    return tmp_path
