from embodied3.library import load_library
from embodied3.graph import Graph

COORDS = {"n001": (0.0, -1.0, 0.0), "n002": (-1.0, 1.0, 0.2), "n004": (1.0, 0.5, 1.0), "n005": (-1.0, 0.6, -1.0), "n006": (-0.3, 0.7, -0.2)}


def test_star_routes_through_hub_and_dwells(workspace):
    lib = load_library(workspace / "videos", workspace / "keyframes")
    g = Graph(lib, coords=COORDS, hub="n001")
    p = g.plan_to("n004")
    assert p.clips == ["edge__n001__n004__v1"] and p.end_node == "n004" and p.idle == "loop__n004__v1" and p.note == "dwell"
    assert p.path(lib) == ["n001", "n004"]
    g.commit(p)
    p = g.plan_to("n005")
    assert p.clips == ["edge__n004__n001__v1", "edge__n001__n005__v1"] and p.path(lib) == ["n004", "n001", "n005"]
    assert g.plan_to("n004").clips == [] and g.plan_to("n004").note == "dwell"        # already there: nothing to play
    assert g.plan_to("n001").clips == ["edge__n004__n001__v1"]


def test_geometry(workspace):
    lib = load_library(workspace / "videos", workspace / "keyframes")
    g = Graph(lib, coords=COORDS, hub="n001")
    assert g.nearest((0.9, 0.4, 0.9)) == ("n004", g.dist((0.9, 0.4, 0.9), COORDS["n004"]))
    assert set(g.reach((0.0, -1.0, 0.0), 0.5)) == {"n001"}
    assert abs(g.distance_to((0.0, 0.0, 0.0), "n001") - 1.0) < 1e-9
    assert g.distance_to((0, 0, 0), "nope") is None


def test_descriptions_file_can_be_a_full_path(workspace):
    lib = load_library(workspace / "videos", workspace / "keyframes", str(workspace / "videos" / "clip_descriptions.json"))
    assert len(lib.clips) == 13 and lib.missing_files() == []
