"""The v3 agent: a process that owns time, with a mood and a body.

Per user turn: brain (emotion + move + reply) -> lexicon reading of the emotion -> mood nudge -> body policy:
    move given and it matches a clip (similarity >= tau)  -> go there, if distance(mood, node) <= budget
    move given, over budget                                -> drift instead, logged as out of reach
    move given, nothing close enough                       -> miss, logged; the body drifts (no wanted list yet)
    no move                                                -> drift: the node nearest the mood (stay if already there)
The mood never picks the node when a move is given; the distance mood <-> body is kept as the mask signal.
During silence the mood decays, and every `drift_check_seconds` the body walks to the node nearest the mood if
that node changed. The clip library is a cache: nothing is generated on a miss in v3.
"""
from __future__ import annotations
import asyncio, time
from pathlib import Path
from typing import Awaitable, Callable
from ..config import Config
from ..affect import Lexicon, Space, Mood, Placer
from ..brain import Brain, EchoBackend, QwenBackend, Move
from ..library import load_library, ClipIndex, Matcher
from ..graph import Graph
from ..memory import MemoryStore, memory_block, time_line, wake_line, ago_str, when_str
from .events import Event, EventType
from .episode import EpisodeLog
from .state import save_state, load_state, restore_state

Sink = Callable[[dict], Awaitable[None]]


class Agent:
    def __init__(self, cfg: Config, sink: Sink, embedder=None):
        self.cfg = cfg
        self.sink = sink
        self.queue: asyncio.Queue[Event] = asyncio.Queue()
        # the space and the mood
        self.lexicon = Lexicon.load(cfg.lexicon_file, min_hits=cfg.min_hits)
        self.space = Space.from_descriptions(self.lexicon, cfg.descriptions_file, clip=cfg.mood_clip,
                                             baseline_node=None if cfg.mood_baseline == "center" else cfg.mood_baseline)
        self.mood = Mood(baseline=self.space.baseline, rest=self.space.baseline, half_life_s=cfg.mood_half_life_s, nudge=cfg.mood_nudge,
                         by_intensity=cfg.mood_nudge_by_intensity, clip=cfg.mood_clip,
                         temperament_half_life_s=cfg.temperament_half_life_s, temperament_max_shift=cfg.temperament_max_shift)
        # the body: library, clip RAG, graph in the same space
        self.lib = load_library(cfg.videos_dir, cfg.keyframes_dir, cfg.descriptions_file)
        self.index = ClipIndex(cfg.index_dir, cfg.embedder, embedder=embedder)
        self.matcher = Matcher(self.index, tau=cfg.tau, spread=cfg.spread, sample_temperature=cfg.sample_temperature)
        self.graph = Graph(self.lib, coords={n: s.coords for n, s in self.space.by_node.items()}, hub=cfg.hub_node)
        self.budget = cfg.move_budget
        self.placer = Placer(self.space, embedder=self.index.embedder) if cfg.placement_fallback else None
        # the brain
        backend = EchoBackend() if cfg.backend == "echo" else QwenBackend(cfg.model_id)
        self.brain = Brain(backend, cfg.persona_file, cfg.instructions_file, cfg.temperature, cfg.max_new_tokens, cfg.history_turns, cfg.seed)
        self.log = EpisodeLog(cfg.episode_log)
        self.started = time.time()
        self.last_interaction = time.time()
        self.last_idle_prompt = time.time()
        self.last_mood_sample = time.time()
        self.last_drift_check = time.time()
        self.arrived_at = time.time()
        self.window_visible = True
        self.busy = False
        self._running = False
        # memory and the self between runs
        self.memory = MemoryStore(cfg.memory_dir, self.index.embedder, floor=cfg.memory_floor, margin=cfg.memory_margin,
                                  max_items=cfg.memory_max, candidates=cfg.memory_candidates)
        self.session = time.strftime("%Y%m%d-%H%M%S")
        self.history_memory_ids: set[str] = set()      # memories of turns still in the verbatim history: not recalled twice
        self.previous_session = ""
        self.away_gap = 0.0
        self.off_at: float | None = None
        self.on_at: float = time.time()
        st = load_state(cfg.state_file)
        if st:
            self.away_gap = restore_state(self, st)
            self.off_at = float(st["saved_at"])
        self.log.write("start", seed=self.space.table(), correlation=self.space.hand_correlation(), baseline=self.space.baseline,
                       clips=len(self.lib.clips), nodes=len(self.lib.keyframes), hub=self.graph.HUB, memories=self.memory.count(),
                       restored=bool(st), away_s=self.away_gap, session=self.session)

    def save(self) -> None:
        try:
            save_state(self.cfg.state_file, self)
        except Exception as e:
            self.log.write("error", event="save_state", error=repr(e))

    # ---- lifecycle ----
    def ensure_index(self, rebuild: bool = False) -> None:
        try:
            has = self.index.db.get_collection(ClipIndex.COLLECTION).count() > 0
        except Exception:
            has = False
        if rebuild or not has:
            n = self.index.build(self.lib)
            self.log.write("index_built", documents=n, clips=len(self.lib.clips))

    async def run(self) -> None:
        self._running = True
        asyncio.create_task(self._ticker())
        if self.off_at is not None and self.away_gap >= self.cfg.wake_min_gap_s:
            await self.wake()
        while self._running:
            ev = await self.queue.get()
            try:
                await self.handle(ev)
            except Exception as e:  # never let one bad event kill the loop
                self.log.write("error", event=ev.type.value, error=repr(e))
                await self.sink({"type": "log", "text": f"error: {e!r}"})

    async def wake(self) -> None:
        """Back after a gap: apply the mood decay for the whole gap, then let her react to the time that passed."""
        self.mood.tick()
        await self.sink({"type": "log", "text": f"she was switched off {when_str(self.off_at)}, {ago_str(self.off_at)}; "
                                                f"{len(self.brain.history) // 2} turns of history and {self.memory.count()} memories restored"})
        await self.think(wake_line(self.off_at), "wake")

    def stop(self) -> None:
        self._running = False

    async def _ticker(self) -> None:
        while self._running:
            await asyncio.sleep(self.cfg.tick_seconds)
            await self.queue.put(Event(EventType.TIMER))

    def post(self, ev: Event) -> None:
        self.queue.put_nowait(ev)

    # ---- the reading ----
    def read(self, text: str) -> dict:
        if not text or not text.strip():
            return {"coords": None, "method": "none", "reading": None, "placement": None}
        reading, coords = self.space.read(text)
        out = {"coords": coords, "method": reading.method if coords else "none", "reading": reading.to_dict(), "placement": None}
        if coords is None and self.placer is not None:
            try:
                coords, ranked = self.placer.place(text)
                out.update(coords=coords, method="placement", placement=ranked)
            except Exception as e:
                out["placement_error"] = repr(e)
        return out

    # ---- the body ----
    def mask(self) -> float | None:
        """Distance between the mood and the node the body stands on: a face that differs from a feeling."""
        return self.graph.distance_to(self.mood.point, self.graph.pos)

    def time_line_now(self) -> str:
        return time_line(off_at=self.off_at if self.away_gap >= self.cfg.wake_min_gap_s else None, on_at=self.on_at)

    def body_line(self) -> str:
        here = self.graph.label(self.graph.pos)
        return (f"Your face is currently showing '{here}' and has been for {int(time.time() - self.arrived_at)} s. "
                "Ask for a move only when there is a reason to change it.")

    def decide_body(self, move: Move) -> dict:
        """The fetch policy. Returns what happened and, if the body moves, the plan."""
        d = {"reason": "stay", "target": None, "similarity": None, "hits": [], "cost": None, "budget": self.budget,
             "plan": None, "move": move.description, "pos": self.graph.pos}
        near, dnear = self.graph.nearest(self.mood.point)
        if move:
            m = self.matcher.match(move.description)
            d["hits"] = [{**h, "label": self.graph.label(h["node"])} for h in m.hits]
            d["similarity"] = m.similarity
            if m.status == "hit":
                cost = self.graph.distance_to(self.mood.point, m.node)
                d["cost"] = cost
                if cost is not None and cost > self.budget:
                    d.update(reason="out_of_reach", requested=m.node, target=near)
                else:
                    d.update(reason="hit", target=m.node)
            elif m.status == "miss":
                d.update(reason="miss", requested=m.node, target=near)
            else:
                d.update(reason="empty", target=near)
        else:
            d.update(reason="drift", target=near)
        if d["target"] and d["target"] != self.graph.pos:
            plan = self.graph.plan_to(d["target"])
            if plan:
                d["plan"] = {"clips": plan.clips, "idle": plan.idle, "end_node": plan.end_node, "cost_s": plan.cost,
                             "note": plan.note, "path": plan.path(self.lib)}
        elif d["target"] == self.graph.pos and d["reason"] in ("drift", "miss", "out_of_reach", "empty"):
            d["reason"] = d["reason"] + "/stay"
        return d

    async def enact(self, body: dict, source: str) -> None:
        """Send the plan to the window and commit the graph position."""
        p = body.get("plan")
        if not p:
            return
        plan_obj = self.graph.plan_to(p["end_node"])       # same plan; commit it
        if plan_obj:
            self.graph.commit(plan_obj)
        self.arrived_at = time.time()
        await self.sink({"type": "play", "plan": p["clips"], "idle": p["idle"], "end_node": p["end_node"],
                         "end_label": self.graph.label(p["end_node"]), "note": p["note"], "path": p["path"],
                         "reason": body["reason"], "source": source})

    def gpu_msg(self) -> dict | None:
        st = getattr(self.brain.backend, "stats", None)
        if not st:
            return None
        bpt = st.get("kv_bytes_per_token") or 0
        ctx = int(st.get("prompt_tokens", 0)) + int(st.get("new_tokens", 0))
        budget = int(self.cfg.context_budget_tokens)
        msg = {"type": "gpu", "prompt_tokens": st.get("prompt_tokens"), "new_tokens": st.get("new_tokens"), "context_tokens": ctx,
               "budget_tokens": budget, "kv_mb": ctx * bpt / 2**20, "kv_budget_mb": budget * bpt / 2**20,
               "kv_kb_per_token": bpt / 1024, "seconds": st.get("seconds"), "model": st.get("model"),
               "oom_retries": st.get("oom_retries", 0)}
        g = st.get("gpu")
        if g:
            msg.update({k + "_mb": g[k] / 2**20 for k in ("allocated", "reserved", "peak", "total", "free")})
        return msg

    def mood_msg(self) -> dict:
        near, d = self.space.nearest(self.mood.point)
        return {"type": "mood", "point": self.mood.point, "baseline": self.mood.baseline, "rest": self.mood.rest, "sentence": self.mood.sentence(),
                "nearest": near.node if near else None, "nearest_label": near.label if near else None,
                "nearest_file": near.file if near else None, "distance": d if near else None,
                "body": self.graph.pos, "body_label": self.graph.label(self.graph.pos), "mask": self.mask(), "t": time.time()}

    # ---- handlers ----
    async def handle(self, ev: Event) -> None:
        if ev.type == EventType.USER_MESSAGE:
            await self.on_user(ev.payload["text"])
        elif ev.type == EventType.TIMER:
            await self.on_tick()
        elif ev.type == EventType.WINDOW:
            self.on_window(ev.payload)
        elif ev.type == EventType.COMMAND:
            await self.on_command(ev.payload)

    async def think(self, text: str, source: str) -> None:
        self.busy = True
        try:
            await self._think(text, source)
        finally:
            self.busy = False                                   # never leave her stuck after an exception
            await self.sink({"type": "status", "text": "idle"})

    async def _think(self, text: str, source: str) -> None:
        await self.sink({"type": "status", "text": "thinking"})
        self.mood.tick()
        mood_before = self.mood.point
        mood_line, body_line, tline = self.mood.sentence(), self.body_line(), self.time_line_now()
        loop = asyncio.get_running_loop()
        user_turn = source == "user"
        # recall: what earlier conversations say about this, if anything (adaptive count; never the verbatim history).
        # Only for the user's words: an idle prompt is not a question, and recalling on it inflates `refs`.
        mv = self.mood.point[0] if self.cfg.recall_congruence else None
        memories = []
        if user_turn:
            try:
                memories = self.memory.retrieve(text, self.history_memory_ids, mood_valence=mv)
                if memories:
                    self.memory.bump_refs([m["id"] for m in memories])
            except Exception as e:
                self.log.write("error", event="memory_retrieve", error=repr(e))
        mblock = memory_block(memories)
        turn = await loop.run_in_executor(None, lambda: self.brain.turn(text, mood_line, body_line, mblock, tline, keep_in_history=user_turn))
        rd = await loop.run_in_executor(None, self.read, turn.emotion)
        # sincerity: how far what she reports sits from the mood she holds (the report's "reported vs held" gap)
        sincerity_gap = self.graph.dist(rd["coords"], mood_before) if rd["coords"] is not None else None
        if rd["coords"] is not None:
            self.mood.apply(rd["coords"])
        body = await loop.run_in_executor(None, self.decide_body, turn.move)
        tnear = self.space.nearest(rd["coords"])[0] if rd["coords"] else None
        gpu = self.gpu_msg()
        await self.sink({"type": "reply", "text": turn.reply, "emotion": turn.emotion, "move": turn.move.to_dict(), "parsed": turn.parsed,
                         "internal": source != "user", "coords": rd["coords"], "method": rd["method"], "reading": rd["reading"],
                         "placement": rd["placement"], "target_nearest": tnear.label if tnear else None,
                         "nudge": self.mood.last_nudge if rd["coords"] is not None else None, "mood_line": mood_line,
                         "sincerity_gap": sincerity_gap,
                         "body": {**body, "target_label": self.graph.label(body["target"]) if body["target"] else None,
                                  "requested_label": self.graph.label(body["requested"]) if body.get("requested") else None},
                         "memories": [{"id": m["id"], "ago": m["ago"], "when": m["when"], "similarity": m["similarity"], "user": m["user"],
                                       "reply": m["reply"], "emotion": m["emotion"], "refs": m.get("refs", 0)} for m in memories],
                         "mood": self.mood_msg()})
        await self.enact(body, source)
        await self.sink(self.mood_msg())
        if gpu:
            await self.sink(gpu)
        # the record and the self first: a crash in the memory store then loses at most one memory, not the turn
        self.log.write("turn", source=source, user=text, reply=turn.reply, emotion=turn.emotion, move=turn.move.to_dict(),
                       parsed=turn.parsed, method=rd["method"], coords=rd["coords"], hits=(rd["reading"] or {}).get("hits"),
                       nudge=self.mood.last_nudge if rd["coords"] is not None else None,
                       mood_before=mood_before, mood_after=self.mood.point, mood_line=mood_line, sincerity_gap=sincerity_gap,
                       baseline=self.mood.baseline,
                       body={k: v for k, v in body.items() if k != "hits"}, body_similarity_hits=body["hits"][:3],
                       pos=self.graph.pos, mask=self.mask(),
                       recalled=[{"id": m["id"], "similarity": m["similarity"], "when": m["when"]} for m in memories],
                       gpu={k: v for k, v in gpu.items() if k != "type"} if gpu else None)
        self.save()
        # remember this interaction, with when it happened and how she felt (on the loop thread: the vector store
        # and its native index are not asked to work from a pool thread)
        if source in ("user", "wake"):
            try:
                mid = self.memory.write(text, turn.reply, turn.emotion, self.mood.point, rd["coords"], self.graph.pos, source, session=self.session)
            except Exception as e:
                self.log.write("error", event="memory_write", error=repr(e))
                await self.sink({"type": "log", "text": f"memory not written: {e!r}"})
                return
            if user_turn:                                   # the wake-up is remembered but is not a history turn
                self.history_memory_ids.add(mid)
                keep = {m.get("memory_id") for m in self.brain.history[-2 * self.brain.history_turns:]} | {mid}
                self.brain.history[-1]["memory_id"] = mid
                self.history_memory_ids = {i for i in self.history_memory_ids if i in keep}
            self.log.write("memory", id=mid, source=source)
            self.save()

    async def on_user(self, text: str) -> None:
        self.last_interaction = time.time()
        await self.think(text, "user")

    async def drift_if_needed(self) -> None:
        """During silence the mood decays; the body follows it when another node has become clearly nearer."""
        near, dnear = self.graph.nearest(self.mood.point)
        here = self.graph.distance_to(self.mood.point, self.graph.pos)
        if near and near != self.graph.pos and here is not None and here - dnear > self.cfg.drift_hysteresis:
            body = {"reason": "drift", "target": near, "pos": self.graph.pos, "plan": None}
            plan = self.graph.plan_to(near)
            if plan:
                body["plan"] = {"clips": plan.clips, "idle": plan.idle, "end_node": plan.end_node, "cost_s": plan.cost,
                                "note": plan.note, "path": plan.path(self.lib)}
                await self.enact(body, "tick")
                await self.sink(self.mood_msg())
                self.log.write("drift", frm=body["pos"], to=near, mood=self.mood.point, mask=self.mask())

    async def on_tick(self) -> None:
        self.mood.tick()
        await self.sink(self.mood_msg())
        now = time.time()
        if now - self.last_mood_sample >= 60:
            self.last_mood_sample = now
            near, d = self.space.nearest(self.mood.point)
            self.log.write("mood", point=self.mood.point, nearest=near.node if near else None, distance=d if near else None,
                           pos=self.graph.pos, mask=self.mask())
        if self.busy:
            return
        if self.cfg.drift_check_seconds > 0 and now - self.last_drift_check >= self.cfg.drift_check_seconds:
            self.last_drift_check = now
            await self.drift_if_needed()
        if self.cfg.idle_prompt_seconds <= 0:
            return
        silent = now - self.last_interaction
        if silent < self.cfg.idle_prompt_seconds or now - self.last_idle_prompt < self.cfg.idle_prompt_min_gap:
            return
        self.last_idle_prompt = now
        prompt = (f"[internal event, not from the user] Nothing has happened for {int(silent // 60)} min. "
                  f"Local time {time.strftime('%H:%M, %A')}. Session length {int((now - self.started) // 60)} min. "
                  f"The window is {'visible' if self.window_visible else 'hidden'}. "
                  "React as yourself: say how you feel now, usually leave your face alone (move \"\") and stay quiet (reply \"\"); "
                  "occasionally say one short thing to yourself. Reply in the usual JSON.")
        await self.think(prompt, "idle")

    def on_window(self, payload: dict) -> None:
        # Renderer feedback keeps the graph position honest even if the window skipped a clip.
        if payload.get("event") == "position" and payload.get("node") in self.lib.keyframes:
            self.graph.pos = payload["node"]
        elif payload.get("event") == "visibility":
            self.window_visible = bool(payload.get("visible", True))

    async def on_command(self, p: dict) -> None:
        cmd = p.get("cmd")
        if cmd == "reset":
            self.brain.reset()
            self.mood.reset()
            self.history_memory_ids = set()
            self.save()
            await self.sink({"type": "log", "text": "history and mood reset (the body stays where it is; memories are kept)"})
            await self.sink(self.mood_msg())
        elif cmd == "recall":                                # try a query against the memory store without the brain
            mems = self.memory.retrieve(str(p.get("text", "")), set())
            await self.sink({"type": "recall", "text": p.get("text", ""), "memories": [
                {"id": m["id"], "ago": m["ago"], "when": m["when"], "similarity": m["similarity"], "user": m["user"], "reply": m["reply"],
                 "emotion": m["emotion"], "refs": m.get("refs", 0)} for m in mems], "count": self.memory.count()})
        elif cmd == "set":
            if "half_life" in p:
                self.mood.half_life_s = float(p["half_life"])
            if "nudge" in p:
                self.mood.nudge = float(p["nudge"])
            if "temperature" in p:
                self.brain.temperature = float(p["temperature"])
            if "tau" in p:
                self.matcher.tau = float(p["tau"])
            if "budget" in p:
                self.budget = float(p["budget"])
        elif cmd == "probe":
            text = str(p.get("text", ""))
            rd = self.read(text)
            if p.get("apply") and rd["coords"] is not None:
                self.mood.apply(rd["coords"])
            tnear = self.space.nearest(rd["coords"])[0] if rd["coords"] else None
            await self.sink({"type": "probe", "text": text, "coords": rd["coords"], "method": rd["method"], "reading": rd["reading"],
                             "placement": rd["placement"], "applied": bool(p.get("apply")), "target_nearest": tnear.label if tnear else None,
                             "mood": self.mood_msg()})
            self.log.write("probe", text=text, coords=rd["coords"], method=rd["method"], applied=bool(p.get("apply")))
        elif cmd == "goto" and p.get("node") in self.lib.keyframes:
            plan = self.graph.plan_to(p["node"])
            if plan:
                body = {"reason": "goto", "target": p["node"], "pos": self.graph.pos,
                        "plan": {"clips": plan.clips, "idle": plan.idle, "end_node": plan.end_node, "cost_s": plan.cost,
                                 "note": plan.note, "path": plan.path(self.lib)}}
                await self.enact(body, "command")
                await self.sink(self.mood_msg())
        elif cmd == "match":                                 # try a move text against the clip RAG without the brain
            body = self.decide_body(Move(description=str(p.get("text", ""))))
            await self.sink({"type": "match", "text": p.get("text", ""), "body": {**body, "target_label": self.graph.label(body["target"]) if body["target"] else None}})
        elif cmd == "rebuild_index":
            self.ensure_index(rebuild=True)
            await self.sink({"type": "log", "text": "index rebuilt"})

    # ---- for the window ----
    def node_table(self) -> dict:
        return {n: {"label": k.label or n, "file": k.file, "coords": self.graph.coords.get(n), "loop": self.graph.has_loop(n)}
                for n, k in self.lib.keyframes.items()}

    def clip_table(self) -> dict:
        return {c.id: {"file": c.file, "kind": c.kind, "start": c.start, "end": c.end, "duration_s": c.duration_s}
                for c in self.lib.clips.values() if (self.lib.videos_dir / c.file).exists()}

    def hello(self) -> dict:
        idle = self.graph.idles.get(self.graph.pos)
        return {"type": "hello", "seed": self.space.table(), "baseline": self.space.baseline, "clip": self.space.clip,
                "correlation": self.space.hand_correlation(), "mood": self.mood_msg(), "gpu": self.gpu_msg(),
                "nodes": self.node_table(), "clips": self.clip_table(), "hub": self.graph.HUB, "pos": self.graph.pos,
                "idle": idle.id if idle else None, "memories": self.memory.count(), "history_turns": len(self.brain.history) // 2,
                "off_at": self.off_at, "away": ago_str(self.off_at) if self.off_at else None, "session": self.session,
                "config": {"half_life": self.mood.half_life_s, "nudge": self.mood.nudge, "temperature": self.brain.temperature,
                           "tau": self.matcher.tau, "budget": self.budget, "backend": self.cfg.backend, "model": self.cfg.model_id}}
