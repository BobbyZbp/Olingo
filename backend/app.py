"""Olingo Live backend. Serves the page on localhost, receives camera frames over a WebSocket (kept in memory only),
runs one call at a time on a thread and pushes every state change to the page as it happens.
  scripts/run.sh        or        ../.venv/bin/python -m backend.app        -> http://localhost:8765"""
import asyncio
import io
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from PIL import Image

from . import agent as agentlink
from . import brain, privacy
from . import lang as L
from .call import Call, has_cjk
from .scene import SceneWatch
from .speech import BridgeSpeech, MockSpeech
from .vision import to_jpeg

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("OLINGO_PORT", "8765"))
VISION_W = int(os.environ.get("OLINGO_VISION_W", "512"))
MOCK = os.environ.get("OLINGO_SPEECH") == "mock"
DECK = ROOT / "data/deck.json"                       # the only thing a call leaves behind: words
RING_BUDGET_S = 25.0                                 # the longest the ringing screen may take
CSP = ("default-src 'self'; "
       f"connect-src 'self' ws://localhost:{PORT} ws://127.0.0.1:{PORT}; "
       "img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self' 'unsafe-inline'")


def load_deck() -> list:
    try:
        return json.loads(DECK.read_text()).get("cards", [])
    except (OSError, ValueError):
        return []


def save_deck(new: list) -> int:
    """Merge this call's words into the deck. Text only: no audio, no picture ever goes in here."""
    cards = {(c.get("lang"), c.get("l1")): c for c in load_deck()}
    day = time.strftime("%Y-%m-%d")
    for c in new:
        old = cards.get((c["lang"], c["l1"]), {})
        cards[(c["lang"], c["l1"])] = {**old, **c, "score": max(c["score"], old.get("score", 0)),
                                       "drilled": c["drilled"] or old.get("drilled", False),
                                       "phrases": c["phrases"] or old.get("phrases", []),
                                       "example": c["example"] or old.get("example", ""), "date": day}
    DECK.parent.mkdir(parents=True, exist_ok=True)
    DECK.write_text(json.dumps({"cards": list(cards.values())}, ensure_ascii=False, indent=1))
    return len(cards)


class Session:
    """Everything about the camera and the call lives here, in memory. Nothing here writes a frame or audio to disk."""
    def __init__(self):
        self.speech = None
        self.speech_ready = threading.Event()
        self.speech_error = ""
        self.speech_secs = None
        self.frame = None            # latest JPEG bytes from the browser; replaced by the next one
        self.frames_seen = 0
        self.scene = SceneWatch()
        self.scene_pushed = 0.0
        self.call = None
        self.thread = None
        self.socks = set()
        self.loop = None
        self.lock = threading.Lock()
        self.agent = None            # the Olingo device, when connected: its camera, microphone and speaker are used
        self.agent_mic = True        # False: the device is camera and speaker only, this Mac's microphone listens
        self.mirrors = set()         # pages opened with ?mirror: they are shown the caller's camera too (memory only)
        self.monitor = privacy.Monitor(self.push, lambda: self.speech.pid if self.speech else None,
                                       device=lambda: self.agent)

    # ----- to the page -----
    def push(self, kind, **d):
        if self.loop is None or not self.socks:
            return
        msg = json.dumps({"t": kind, "at": round(time.time() * 1000), **d}, ensure_ascii=False)
        for ws in list(self.socks):
            asyncio.run_coroutine_threadsafe(self._send(ws, msg), self.loop)

    @staticmethod
    async def _send(ws, msg):
        try:
            await ws.send_text(msg)
        except Exception:
            pass

    @staticmethod
    async def _send_bytes(ws, raw):
        try:
            await ws.send_bytes(raw)
        except Exception:
            pass

    # ----- the Olingo device: eyes, ears and mouth; this computer stays the brain -----
    def connect_agent(self, target: str | None):
        """Use the device's camera, microphone and speaker in place of this computer's."""
        self.disconnect_agent(quiet=True)
        if not target:
            cands = agentlink.candidates()
            target = cands[0]["target"] if cands else None
        if not target:
            self.push("agent", state="failed", message="No Olingo device found. Pair it over Bluetooth first, or plug it in.")
            return
        self.push("agent", state="connecting", target=target)
        link = agentlink.AgentLink(target, on_frame=self._agent_frame, on_mic=self._agent_mic,
                                   on_close=lambda: self.disconnect_agent(lost=True))
        try:
            info = link.connect()
        except Exception as e:
            self.push("agent", state="failed", target=target, message=f"{type(e).__name__}: {e}")
            return
        # Luma sends its microphone uncompressed (256 kbit/s). Measure whether this link carries it: if not, the
        # device stays the camera and the speaker and this Mac listens with its own microphone.
        self.agent_mic, heard = True, None
        if link.luma:
            time.sleep(1.0)                                # let it switch to compressed sound first
            a = link.stats["mic_s"]
            time.sleep(3.0)
            heard = round(link.stats["mic_s"] - a, 1)
            self.agent_mic = heard >= 2.6
        self.agent = link
        self.speech_ready.wait(timeout=120)
        if self.speech:
            self.speech.on_speaker, self.speech.on_flush = link.speaker, link.flush
            self.speech.set_agent_audio(True, mic=self.agent_mic)
        self.scene.reset()
        self.push("agent", state="connected", **link.summary(), camera=info.get("camera"),
                  mic="device" if self.agent_mic else "mac", mic_arrived=heard)
        if self.speech and not (self.call and self.call.running):
            try:                                           # said by the device itself: you hear that it works
                self.speech.speak(f"{link.info.get('name', 'The device')} is connected.", lang="en")
            except Exception:
                pass

    def disconnect_agent(self, lost: bool = False, quiet: bool = False):
        link, self.agent = self.agent, None
        if link is None:
            return
        link.close()
        if self.speech:
            try:
                self.speech.set_agent_audio(False)             # back to this computer's microphone and speakers
            except Exception:
                pass
            self.speech.on_speaker = self.speech.on_flush = None
        self.frame = None
        if not quiet:
            self.push("agent", state="lost" if lost else "disconnected", **link.summary())

    def _agent_frame(self, raw: bytes):
        self.on_frame(raw)                                     # same path as a laptop frame: memory only
        if self.loop:
            for ws in list(self.socks):                        # and onto the page, in place of the laptop camera
                asyncio.run_coroutine_threadsafe(self._send_bytes(ws, raw), self.loop)

    def _agent_mic(self, pcm16: bytes):
        if self.agent_mic and self.speech and self.speech_ready.is_set():
            self.speech.feed_mic(pcm16)

    # ----- speech bridge: started once at boot, kept for every call -----
    def boot_speech(self):
        self.speech_ready.clear()
        self.speech_error = ""
        try:
            if MOCK:
                self.speech = MockSpeech()
            else:
                self.speech = BridgeSpeech(
                    on_level=lambda v: self.push("vu", v=round(v, 3)),
                    on_mic=lambda v, sp: self.push("miclevel", v=round(v, 3), speech=sp),
                    on_error=lambda code, m: self.push("error", code=code, message=m))
            self.speech_secs = round(self.speech.start(), 1)
        except Exception as e:
            self.speech_error = str(e)
        if self.agent and self.speech and not self.speech_error:      # a restarted bridge must know about the device
            self.speech.on_speaker, self.speech.on_flush = self.agent.speaker, self.agent.flush
            self.speech.set_agent_audio(True, mic=self.agent_mic)
        self.speech_ready.set()
        self.push("hello", **self.hello())
        if not MOCK and not self.speech_error:        # the usual pair's fixed lines, quietly, until a call starts
            self.prerender("en", "zh", L.voice_for("en", None, "zh"), lambda: bool(self.thread and self.thread.is_alive()))

    def prerender(self, l1, l2, voice, stop=lambda: False, extra=(), progress=None) -> int:
        """Make the audio of every fixed line (and any extra words) ahead of time. Each is synthesised once, ever."""
        todo = [(t, lang, L.voice_for_text(lang, voice)) for t, lang in [*L.fixed_texts(l1, l2), *extra]]
        todo = [x for x in todo if not self.speech.key(*x).exists()]
        made = 0
        for i, (text, lang, who) in enumerate(todo):
            if stop():
                break
            if progress:
                progress(i, len(todo))
            try:
                made += bool(self.speech.render(text, lang, timeout=20.0, speaker=who))
            except Exception:
                break
        return made

    def hello(self):
        return {"speech_ready": self.speech_ready.is_set() and not self.speech_error, "speech_error": self.speech_error,
                "running": bool(self.call and self.call.running), "bonsai_9b": brain.available("9B"),
                "bonsai_4b": brain.available("4B"), "mock": MOCK, "deck": len(load_deck()),
                "voice_id": bool(self.speech and getattr(self.speech, "ready_info", {}).get("voice_id")),
                "agent": ({**self.agent.summary(), "mic": "device" if self.agent_mic else "mac"} if self.agent else None),
                "langs": [{"code": c, "name": L.name(c, "en")} for c in L.LANGS],
                "voices": [{"id": v, "label": L.VOICE_LABELS.get(v, v)} for v in L.VOICE_CHOICES]}

    # ----- camera -----
    def model_frame(self):
        """The frame Bonsai gets: the latest one, downsized to VISION_W. None before the camera has sent anything."""
        raw = self.frame
        if raw is None:
            return None
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            w = min(VISION_W, im.width)                    # a small picture from the device is not blown up
            return to_jpeg(im.resize((w, round(im.height * w / im.width))))
        except Exception:
            return None

    def on_frame(self, raw: bytes):
        self.frame = raw
        self.frames_seen += 1
        now = time.time()
        fired = self.scene.update(raw, now)
        c = self.call
        took = bool(fired and c and c.notify_scene(self.model_frame(), self.scene.chg, self.scene.still_for(now)))
        if took:
            self.scene.absorb()
        if fired or now - self.scene_pushed > 0.5:
            self.scene_pushed = now
            self.push("scene", chg=round(self.scene.chg, 3), mot=round(self.scene.mot, 3), fired=fired, took=took)

    # ----- the call -----
    def start_call(self, opts: dict):
        with self.lock:
            old = self.thread
            if old and old.is_alive():                 # a page reload or a second click: end the old call first
                if self.call:
                    self.call.stop()
                old.join(timeout=6)
            self.thread = threading.Thread(target=self._run, args=(opts,), daemon=True)
            self.thread.start()

    def _preflight(self) -> str:
        """-> "" when a call can start, otherwise the reason, in words a person can act on."""
        self.speech_ready.wait(timeout=600)
        if self.speech_error:
            return "The speech models did not load: " + self.speech_error
        if not brain.available("9B"):
            return "Bonsai 9B is not running. Start it with scripts/run.sh."
        pong = self.speech.alive()
        local_audio = self.agent is None                  # with the device connected, this computer's audio is not used
        if pong is None or (local_audio and not (pong.get("audio_out", True) and pong.get("audio_in", True))):
            # dead, or macOS refused the audio devices (it does that when memory is nearly full): one fresh start
            self.push("ring", pct=8)
            try:
                self.speech.proc.kill()
            except Exception:
                pass
            self.boot_speech()
            pong = None if self.speech_error else self.speech.alive()
        if pong is None:
            return "The speech bridge is not answering: " + (self.speech_error or "see /tmp/olingo_bridge.log")
        if local_audio and not (pong.get("audio_out", True) and pong.get("audio_in", True)):
            m = privacy.memory()
            hog = (privacy.hogs(set(self.monitor.pids())) or [{"name": "other apps", "gb": 0}])[0]
            return (f"macOS refused to open the microphone or speakers. Memory is nearly full "
                    f"({m['wired_gb']} of {m['limit_gb']} GB locked). Close {hog['name']} ({hog['gb']} GB) and press Start again.")
        return ""

    def _run(self, opts: dict):
        """While it rings: everything that makes the first turn fast, done for the language and level just chosen."""
        t0 = time.time()
        l1 = opts.get("l1") if opts.get("l1") in L.L1S else os.environ.get("OLINGO_L1", "en")
        l2 = opts.get("l2") if opts.get("l2") in L.LANGS else os.environ.get("OLINGO_L2", "zh")
        if l2 == l1:
            l2 = "zh" if l1 == "en" else "en"
        level = min(max(int(opts.get("level") or os.environ.get("OLINGO_LEVEL", 1)), 1), 3)
        voice = L.voice_for(l1, opts.get("voice"), l2)
        self.frames_seen = 0
        self.scene.reset()
        self.monitor.reset()
        self.push("phase", phase="calling")
        self.push("ring", pct=5)
        problem = self._preflight()
        if problem:
            self.push("blocked", message=problem)
            return
        warm = {}
        for _ in range(25):                             # the camera needs a moment for its first frame
            if self.frame is not None:
                break
            time.sleep(0.1)
        frame = self.model_frame()

        def warm_chat():
            if brain.available("4B"):
                warm["4B"] = brain.warmup("4B", True, l1, l2, level)     # with a frame: its vision tower loads now
        th = threading.Thread(target=warm_chat)
        th.start()
        self.push("ring", pct=15)
        warm["9B"] = brain.warmup("9B", True, l1, l2, level)
        self.push("ring", pct=40)
        room = []
        if frame is not None:                           # a first look at the room, so the call can open with it
            room, info = brain.room(frame, l1, l2)
            self.push("ring", pct=60)
            warm["room"] = info.get("ms")
            if L.LANGS[l2]["cjk"]:
                room = [t for t in room if has_cjk(t["l2"])]
            self.scene.absorb()
        th.join()
        made = 0
        if not MOCK:
            extra = [(t["l2"], l2) for t in room]
            made = self.prerender(l1, l2, voice, stop=lambda: time.time() - t0 > RING_BUDGET_S, extra=extra,
                                  progress=lambda i, n: self.push("ring", pct=60 + round(38 * i / max(n, 1))))
        self.push("ring", pct=100)
        time.sleep(max(0.0, 1.5 - (time.time() - t0)))
        self.push("warm", bonsai_9b_ms=warm.get("9B"), bonsai_4b_ms=warm.get("4B"), room_ms=warm.get("room"),
                  bridge_s=self.speech_secs, chat_model="4B" if "4B" in warm else "9B", lines_made=made,
                  whisper=self.speech.ready_info.get("whisper"), whisper_fast=self.speech.ready_info.get("whisper_fast"),
                  room=[t["l1"] for t in room], ms=round((time.time() - t0) * 1000))
        t_call = time.time()
        call = Call(self.speech, self.model_frame, self.push, l1=l1, l2=l2, level=level, voice=voice,
                    ptt=bool(opts.get("ptt")), focus=opts.get("focus", True) is not False, room=room,
                    on_look=self.scene.absorb)
        self.call = call
        try:
            call.run()
        except Exception as e:                                   # never leave the page hanging
            self.push("error", code="call", message=f"{type(e).__name__}: {e}")
        try:
            self.monitor.once()
        except Exception:
            pass
        vs = call.voice_summary()
        self.speech.forget_voice()                               # the learned voice does not outlive the call
        deck = call.deck()
        total = save_deck(deck) if deck else len(load_deck())
        st = call.stats
        n = st["tier0"] + st["tier1"] + st["tier2"]
        self.push("receipt", frames_seen=self.frames_seen, frames_kept=0, heard_s=round(st["heard_s"], 1),
                  audio_kept_s=0, looks=st["looks"], model_calls=st["model_calls"], responses=n,
                  no_model_pct=round(100 * st["tier0"] / n) if n else 0, tiers=[st["tier0"], st["tier1"], st["tier2"]],
                  prompt_tokens=st["prompt_tokens"], gen_tokens=st["gen_tokens"], cache_hits=st["cache_hits"],
                  live_tts=st["live_tts"], spoke_first=st["spoke_first"], ignored=st["ignored"],
                  words=deck, deck_total=total, voice_focus=vs["on"], voice_learned_s=vs["learned_s"], name=call.name, l2_name=L.name(call.l2, "en"), date=time.strftime("%B %d, %Y"),
                  agent=self.agent.summary() if self.agent else None,
                  call_s=round(time.time() - t_call), external=self.monitor.external, checks=self.monitor.checks,
                  processes=self.monitor.last.get("processes", 0), models=self.monitor.models)

    def preview(self, opts: dict):
        """Start screen: let the learner hear a voice before choosing it (its name and one word of the new language)."""
        if (self.call and self.call.running) or not self.speech_ready.is_set() or self.speech_error:
            return
        l1 = opts.get("l1") if opts.get("l1") in L.L1S else "en"
        l2 = opts.get("l2") if opts.get("l2") in L.LANGS and opts.get("l2") != l1 else ("zh" if l1 == "en" else "en")
        voice = L.voice_for(l1, opts.get("voice"), l2)
        parts = L.preview_parts(voice, l1, l2)
        self.speech.stop()                              # a second click replaces the first sample
        for i, (text, lang) in enumerate(parts):
            last = i == len(parts) - 1
            self.speech.speak(text, lang=lang, speaker=L.voice_for_text(lang, voice),
                              on_start=(lambda h: self.push("preview", voice=voice, state="playing", how=h.kind)) if i == 0 else None,
                              on_done=(lambda h: self.push("preview", voice=voice, state="done")) if last else None)

    def hangup(self):
        if self.call and self.call.running:
            self.push("phase", phase="ending")
            self.call.stop()


S = Session()


@asynccontextmanager
async def lifespan(_app):
    S.loop = asyncio.get_running_loop()
    threading.Thread(target=S.boot_speech, daemon=True).start()
    S.monitor.start()
    yield
    if S.call:
        S.call.stop()
    S.disconnect_agent(quiet=True)
    if S.speech:
        S.speech.close()

app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html", headers={"Content-Security-Policy": CSP, "Cache-Control": "no-store"})


@app.get("/static/{name}")
def static(name: str):
    p = (ROOT / "web" / name).resolve()
    if p.parent != (ROOT / "web").resolve() or not p.is_file():
        return Response(status_code=404)
    return FileResponse(p, headers={"Cache-Control": "no-store"})


@app.get("/api/state")
def state():
    c = S.call
    return {**S.hello(), "frames_seen": S.frames_seen, "stats": c.stats if c else None,
            "privacy": S.monitor.last, "memory": S.monitor.mem}


@app.get("/api/agent/candidates")
def agent_candidates():
    """Where an Olingo device might be reached from this computer."""
    return {"candidates": agentlink.candidates(), "connected": S.agent.summary() if S.agent else None}


@app.get("/api/deck")
def deck():
    """The saved flashcards: words, readings, example sentences. Nothing else is ever stored."""
    return {"cards": load_deck()}


SHOTS = ROOT.parent / "shots_new"                       # outside the repository: looked at before anything is published


@app.post("/api/shot")
async def shot(request: Request, name: str = "shot"):
    """Photo mode (http://localhost:8765/?shoot): the page sends a picture of itself, taken because the person in
    front of it asked for one. This is the one place where a picture is written to disk, and only in that mode."""
    import re
    data = await request.body()
    if not re.fullmatch(r"[a-z0-9_]{1,40}", name) or not data.startswith(b"\x89PNG") or len(data) > 40_000_000:
        return {"ok": False}
    SHOTS.mkdir(exist_ok=True)
    (SHOTS / f"{name}.png").write_bytes(data)
    return {"ok": True, "saved": str(SHOTS / f"{name}.png"), "kb": len(data) // 1024}


@app.post("/api/typed")
def typed(d: dict):
    """Mock speech only: what the learner 'said'."""
    if isinstance(S.speech, MockSpeech):
        S.speech.type(str(d.get("text", "")))
        return {"ok": True}
    return {"ok": False}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    S.socks.add(sock)
    await S._send(sock, json.dumps({"t": "hello", **S.hello()}))
    try:
        while True:
            m = await sock.receive()
            if m.get("type") == "websocket.disconnect":
                break
            if m.get("bytes") is not None:
                if S.agent is None:                 # with the device connected, its camera is the camera
                    S.on_frame(m["bytes"])          # replaces the previous frame; memory only
                    for other in list(S.mirrors):   # a second page that follows this call (?mirror) sees it too
                        if other is not sock:
                            asyncio.ensure_future(S._send_bytes(other, m["bytes"]))
            elif m.get("text"):
                d = json.loads(m["text"])
                kind = d.get("type")
                if kind == "mirror":
                    S.mirrors.add(sock)
                elif kind == "start":
                    S.start_call(d)
                elif kind == "hangup":
                    S.hangup()
                elif kind == "preview":
                    S.preview(d)
                elif kind == "agent":
                    action = d.get("action")
                    threading.Thread(target=(lambda: S.connect_agent(d.get("target"))) if action == "connect"
                                     else S.disconnect_agent, daemon=True).start()
                elif kind == "ptt" and S.call:
                    S.call.ptt_event(bool(d.get("down")))
                elif kind == "ptt_mode" and S.call:
                    S.call.set_ptt(bool(d.get("on")))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        S.socks.discard(sock)
        S.mirrors.discard(sock)
        if not S.socks:                             # the last page is gone: end the call, forget the frame
            S.frame = None
            if S.call:
                S.call.stop()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
