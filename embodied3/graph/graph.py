"""The keyframe graph: stills are nodes, clips are directed edges (kinds: loop, edge, act). Position and routing live here.

An edge exists iff a clip exists whose first frame is A and last frame is B. Routing is shortest path on clip
duration with a hop limit. A node with an idle loop is a place the avatar can dwell; a node without one is left
immediately by the cheapest exit. Geometry (distance to the mood, nearest node) uses the coordinates the space
computed for each keyframe, so the graph and the mood live in one space (see the report: one space, two occupants).
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass
from ..library.records import Library, Clip

Vec = tuple[float, float, float]


@dataclass
class Plan:
    clips: list[str]            # clip ids to play in order
    idle: str | None            # idle clip to loop afterwards (None = hold last frame)
    end_node: str               # where the avatar will be looping/holding
    cost: float                 # seconds of clip
    note: str = ""

    def path(self, lib: Library) -> list[str]:
        """The nodes visited, starting node first."""
        if not self.clips:
            return [self.end_node]
        first = lib.clips[self.clips[0]].start
        return [first] + [lib.clips[c].end for c in self.clips]


class Graph:
    def __init__(self, lib: Library, coords: dict[str, Vec] | None = None, max_hops: int = 3, hub: str | None = None):
        self.lib = lib
        self.coords: dict[str, Vec] = {n: tuple(c) for n, c in (coords or {}).items() if c is not None}
        self.max_hops = max_hops
        self.out: dict[str, list[Clip]] = {n: [] for n in lib.keyframes}
        self.idles: dict[str, Clip] = {}
        for c in lib.clips.values():
            if c.kind == "loop":
                self.idles.setdefault(c.start, c)          # first variant wins for now
            elif c.kind in ("edge", "act"):
                self.out.setdefault(c.start, []).append(c)
        self.HUB = hub if hub in lib.keyframes else self._guess_hub()
        self.pos = self.HUB

    def _guess_hub(self) -> str:
        for n, k in self.lib.keyframes.items():
            if (k.label or "").lower() == "neutral" or n == "neutral":
                return n
        return max(self.lib.keyframes, key=lambda n: len(self.out.get(n, []))) if self.out else next(iter(self.lib.keyframes))

    def label(self, node: str) -> str:
        k = self.lib.keyframes.get(node)
        return (k.label or node) if k else node

    # ---- queries ----
    def nodes(self) -> list[str]:
        return list(self.lib.keyframes)

    def has_loop(self, node: str) -> bool:
        return node in self.idles

    def edges_from(self, node: str) -> list[Clip]:
        return [c for c in self.out.get(node, []) if c.kind != "act"]

    def shortest_path(self, src: str, dst: str) -> tuple[list[Clip], float] | None:
        """Dijkstra on duration with a hop limit. Returns (clips, cost) or None."""
        if src == dst:
            return [], 0.0
        best: dict[tuple[str, int], float] = {}
        heap = [(0.0, 0, src, [])]
        while heap:
            cost, hops, node, path = heapq.heappop(heap)
            if node == dst:
                return path, cost
            if hops >= self.max_hops:
                continue
            for c in self.edges_from(node):
                key = (c.end, hops + 1)
                nc = cost + c.duration_s
                if nc < best.get(key, float("inf")):
                    best[key] = nc
                    heapq.heappush(heap, (nc, hops + 1, c.end, path + [c]))
        return None

    # ---- geometry ----
    @staticmethod
    def dist(a: Vec, b: Vec) -> float:
        return sum((a[k] - b[k]) ** 2 for k in range(3)) ** 0.5

    def distance_to(self, point: Vec, node: str) -> float | None:
        c = self.coords.get(node)
        return self.dist(point, c) if c is not None else None

    def nearest(self, point: Vec) -> tuple[str | None, float]:
        best, bd = None, float("inf")
        for n, c in self.coords.items():
            d = self.dist(point, c)
            if d < bd:
                best, bd = n, d
        return best, bd

    def reach(self, point: Vec, budget: float) -> list[str]:
        """Nodes whose distance to `point` is within the budget."""
        return [n for n, c in self.coords.items() if self.dist(point, c) <= budget]

    # ---- planning ----
    def plan_to(self, target: str, dwell: bool | None = None) -> Plan | None:
        """Plan from the current position to `target`.
        dwell=True: stay on target (needs an idle loop); False: perform and come back to the hub; None: dwell if possible."""
        if target not in self.lib.keyframes:
            return None
        route = self.shortest_path(self.pos, target)
        if route is None:
            return None
        clips, cost = route
        if dwell is None:
            dwell = self.has_loop(target)
        if dwell and self.has_loop(target):
            return Plan([c.id for c in clips], self.idles[target].id, target, cost, "dwell")
        back = self.shortest_path(target, self.HUB)
        if back is None:
            return Plan([c.id for c in clips], None, target, cost, "no exit: hold")
        bclips, bcost = back
        idle = self.idles.get(self.HUB)
        return Plan([c.id for c in clips + bclips], idle.id if idle else None, self.HUB, cost + bcost, "perform and return")

    def commit(self, plan: Plan) -> None:
        self.pos = plan.end_node
