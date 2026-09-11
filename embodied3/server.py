"""Local server for v3: serves the window, the clips, the keyframes, and a websocket to the running agent.

    python -m embodied3.server --config config.json [--backend echo]
"""
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from .config import Config
from .agent import Agent, Event, EventType

STATIC = Path(__file__).parent / "renderer" / "static"


def create_app(cfg: Config, embedder=None) -> FastAPI:
    app = FastAPI(title="embodied v3")
    clients: set[WebSocket] = set()

    async def broadcast(msg: dict) -> None:
        dead = []
        for ws in clients:
            try:
                await ws.send_text(json.dumps(msg, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    agent = Agent(cfg, broadcast, embedder=embedder)
    agent.ensure_index()
    app.state.agent = agent
    app.mount("/clips", StaticFiles(directory=str(agent.lib.videos_dir)), name="clips")

    @app.on_event("startup")
    async def _start():
        asyncio.create_task(agent.run())

    @app.on_event("shutdown")
    async def _stop():
        agent.stop()
        agent.save()                       # her self, for next time: history, mood + clock, body, timestamps

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse((STATIC / "window.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        hub = agent.lib.keyframes.get(agent.graph.HUB)
        path = agent.lib.keyframes_dir / hub.file if hub else None
        return FileResponse(path, media_type="image/png") if path and path.exists() else HTMLResponse(status_code=204)

    @app.get("/keyframes/{name}")
    async def keyframe(name: str):
        p = agent.lib.keyframes_dir / Path(name).name
        return FileResponse(p) if p.exists() else HTMLResponse(status_code=404)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        await ws.send_text(json.dumps(agent.hello(), default=str))
        try:
            while True:
                m = json.loads(await ws.receive_text())
                if m.get("type") == "user":
                    agent.post(Event(EventType.USER_MESSAGE, {"text": m["text"]}))
                elif m.get("type") == "command":
                    agent.post(Event(EventType.COMMAND, m))
                elif m.get("type") == "window":
                    agent.post(Event(EventType.WINDOW, m))
        except WebSocketDisconnect:
            clients.discard(ws)

    return app


def main():
    # a native crash (CUDA, bitsandbytes, the vector index) kills the process without a Python traceback;
    # faulthandler writes the Python stack of every thread to data/kb/crash.log so the next one leaves evidence
    import faulthandler
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--backend", choices=["qwen", "echo"])
    a = ap.parse_args()
    cfg = Config.load(a.config)
    if a.host:
        cfg.host = a.host
    if a.port:
        cfg.port = a.port
    if a.backend:
        cfg.backend = a.backend
    if not Path(cfg.lexicon_file).exists():
        print(f"\nThe NRC VAD lexicon is missing: {cfg.lexicon_file}\n"
              "It is under a research licence and is not shipped with the project. Download it from\n"
              "https://saifmohammad.com/WebPages/nrc-vad.html and copy NRC-VAD-Lexicon-v2.1.txt into data/lexicon/\n"
              "(see data/lexicon/README.md).\n")
        raise SystemExit(2)
    crash_log = Path(cfg.episode_log).parent / "crash.log"
    crash_log.parent.mkdir(parents=True, exist_ok=True)
    faulthandler.enable(file=open(crash_log, "a", encoding="utf-8"), all_threads=True)
    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
