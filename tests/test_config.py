import json
from embodied3.config import Config


def test_relative_paths_resolve_against_the_config_file(tmp_path):
    (tmp_path / "agent").mkdir()
    cfgf = tmp_path / "agent" / "config.json"
    cfgf.write_text(json.dumps({"lexicon_file": "data/lex.txt", "videos_dir": "data/videos", "port": 9999,
                                "state_file": "C:/abs/state.json" if __import__("os").name == "nt" else "/abs/state.json"}), encoding="utf-8")
    cfg = Config.load(cfgf)
    assert cfg.lexicon_file == str((tmp_path / "agent" / "data" / "lex.txt").resolve())
    assert cfg.videos_dir == str((tmp_path / "agent" / "data" / "videos").resolve())
    assert cfg.port == 9999 and cfg.state_file.replace("\\", "/").endswith("abs/state.json")
    # a copied folder is a new agent: same relative config, different absolute data
    (tmp_path / "twin").mkdir()
    (tmp_path / "twin" / "config.json").write_text(cfgf.read_text(encoding="utf-8"), encoding="utf-8")
    assert Config.load(tmp_path / "twin" / "config.json").videos_dir == str((tmp_path / "twin" / "data" / "videos").resolve())
