"""Speech through the resident Swift bridge (Argmax WhisperKit + TTSKit), or a mock that needs no models.
Speaking never blocks: `speak()` queues an item on the bridge and returns a handle; the bridge plays items back to
back and synthesises the next one while the previous is still audible. Every sentence is cached by
sha1(text | speaker | language): it is synthesised at most once, afterwards the wav is just played."""
import base64
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_BRIDGE_CANDIDATES = [ROOT / "bridge/.build/release/OlingoBridge", ROOT / "bridge_bin/OlingoBridge"]
BRIDGE_BIN = next((p for p in _BRIDGE_CANDIDATES if p.exists()), _BRIDGE_CANDIDATES[0])


class Handle:
    """One queued utterance. `started` is set when it becomes audible, `done` when it ends (or is cancelled)."""
    def __init__(self, text, lang, kind, on_start=None, on_done=None):
        self.text, self.lang, self.kind = text, lang, kind      # kind: "cached" | "live" | "mock"
        self.on_start, self.on_done = on_start, on_done
        self.sent = time.time()
        self.t_start = None
        self.started, self.done = threading.Event(), threading.Event()
        self.info = {}

    def _start(self, e):
        self.t_start = time.time()
        self.started.set()
        if self.on_start:
            self.on_start(self)

    def _done(self, e):
        self.info = e
        self.started.set()
        self.done.set()
        if self.on_done:
            self.on_done(self)

    @property
    def ttfa_ms(self):
        return round((self.t_start - self.sent) * 1000) if self.t_start else None


class BridgeSpeech:
    def __init__(self, cache_dir=None, speaker=None, env=None, on_level=None, on_mic=None, on_error=None,
                 log_path="/tmp/olingo_bridge.log"):
        self.cache = Path(cache_dir or ROOT / "cache/tts")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.speaker = speaker or os.environ.get("TTS_SPEAKER", "serena")
        self.env = {"OLINGO_VOICE_MEAN": str(ROOT / "backend/voice_mean.json"),
                    "VAD_SILENCE_MS": "1000", "VOICE_ID": "on" if os.environ.get("OLINGO_VOICE_ID") == "1" else "off", **os.environ,
                    "TTS_SPEAKER": self.speaker, **(env or {})}
        self.on_level, self.on_mic, self.on_error = on_level, on_mic, on_error
        self.on_speaker = None                    # agent mode: PCM16 24 kHz that the Olingo device must play
        self.on_flush = None                      # agent mode: the device must stop playing now
        self.agent_audio = False
        self.log_path = log_path
        self.q = queue.Queue()                    # everything that is not a per-item event
        self.items = {}                           # id -> Handle
        self.lock = threading.Lock()
        self.n = 0
        self.proc = None
        self.ready_info = {}

    # ----- process -----
    def start(self, ready_timeout: float = 900.0) -> float:
        t0 = time.time()
        self.proc = subprocess.Popen([str(BRIDGE_BIN)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=open(self.log_path, "w"), text=True, bufsize=1, env=self.env)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.ready_info = self._wait(("ready",), ready_timeout)
        return time.time() - t0

    def _reader(self, proc):
        for line in proc.stdout:
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            e["_t"] = time.time()
            ev = e.get("event")
            if ev == "level":
                if self.on_level:
                    self.on_level(e["v"])
            elif ev == "mic_level":
                if self.on_mic:
                    self.on_mic(e["v"], e.get("speech", False))
            elif ev == "spk":
                if self.on_speaker:
                    self.on_speaker(base64.b64decode(e.get("pcm", "")))
            elif ev == "spk_flush":
                if self.on_flush:
                    self.on_flush()
            elif ev in ("start", "said"):
                h = self.items.get(e.get("id"))
                if h:
                    if ev == "start":
                        h._start(e)
                    else:
                        self.items.pop(e.get("id"), None)
                        h._done(e)
            elif ev == "error":
                if self.on_error:
                    self.on_error(e.get("code", ""), e.get("message", ""))
            else:
                self.q.put(e)
        for h in list(self.items.values()):       # the bridge is gone: release everyone who is waiting
            h._done({"stopped": True, "dead": True})
        self.items.clear()
        self.q.put({"event": "dead"})

    def _send(self, **cmd):
        with self.lock:
            self.proc.stdin.write(json.dumps(cmd, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()

    def _wait(self, names, timeout: float = 60.0):
        end = time.time() + timeout
        while True:
            left = end - time.time()
            if left <= 0:
                raise TimeoutError(f"bridge: none of {names} within {timeout:.0f} s")
            try:
                e = self.q.get(timeout=left)
            except queue.Empty:
                raise TimeoutError(f"bridge: none of {names} within {timeout:.0f} s")
            if e["event"] == "dead":
                raise RuntimeError("the speech bridge stopped; see " + self.log_path)
            if e["event"] == "fatal":
                raise RuntimeError("speech bridge: " + e.get("message", "fatal error"))
            if e["event"] in names:
                return e

    def alive(self, timeout: float = 2.0) -> dict | None:
        """-> the bridge's pong (with audio_out / audio_in flags), or None when it does not answer."""
        if not self.proc or self.proc.poll() is not None:
            return None
        try:
            self._send(cmd="ping")
            return self._wait(("pong",), timeout)
        except (TimeoutError, RuntimeError, OSError):
            return None

    # ----- speaking -----
    def key(self, text: str, lang: str = "en", speaker: str | None = None) -> Path:
        return self.cache / (hashlib.sha1(f"{text}|{speaker or self.speaker}|{lang}".encode()).hexdigest()[:16] + ".wav")

    def speak(self, text: str, lang: str = "en", rate: float = 1.0, on_start=None, on_done=None,
              speaker: str | None = None) -> Handle:
        """Queue one utterance and return at once."""
        speaker = speaker or self.speaker
        p = self.key(text, lang, speaker)
        self.n += 1
        i = str(self.n)
        h = Handle(text, lang, "cached" if p.exists() else "live", on_start, on_done)
        self.items[i] = h
        try:
            if p.exists():
                self._send(cmd="play", id=i, file=str(p), rate=rate)
            else:
                self._send(cmd="say", id=i, text=text, lang=lang, rate=rate, save=str(p), speaker=speaker)
        except OSError:
            self.items.pop(i, None)
            h._done({"stopped": True, "dead": True})
        return h

    def render(self, text: str, lang: str = "en", timeout: float = 60.0, speaker: str | None = None) -> bool:
        """Synthesise into the cache without playing. -> True if a file was made."""
        speaker = speaker or self.speaker
        p = self.key(text, lang, speaker)
        if p.exists():
            return False
        self.n += 1
        i = str(self.n)
        h = Handle(text, lang, "live")
        self.items[i] = h
        self._send(cmd="say", id=i, text=text, lang=lang, save=str(p), mute=True, speaker=speaker)
        h.done.wait(timeout)
        return p.exists()

    # ----- listening -----
    def listen(self, lang: str = "en", alt: str | None = None, wait: float = 10.0, max_s: float = 12.0, langs=None,
               filler: str | None = None, ptt: bool = False, fast: bool = False, min_level: float = 0.0,
               hint: str | None = None, voice: str | None = None, voice_min: float | None = None,
               voice_learn: float | None = None, on_event=None, timeout: float | None = None) -> dict:
        """Blocks until the learner has spoken (or `wait` seconds passed). The bridge first lets queued speech finish.
        lang: a code, or "auto" = pick among `langs`. alt: second language tried when the transcript is unsure.
        -> the heard event plus t_end (wall clock at which the learner stopped talking)."""
        cmd = {"cmd": "listen", "lang": lang, "wait": wait, "max": max_s, "ptt": ptt, "fast": fast}
        if min_level > 0:
            cmd["min_level"] = min_level                 # quieter sound than this is not treated as speech
        if hint:
            cmd["hint"] = hint                           # the word the learner is repeating
        if voice:
            cmd["voice"] = voice                         # "enroll": learn this speaker; "check": compare with them
            if voice_min is not None:
                cmd["voice_min"] = voice_min             # less similar than this = someone else: dropped unheard
            if voice_learn is not None:
                cmd["voice_learn"] = voice_learn
        if alt and alt != lang:
            cmd["alt"] = alt
        if langs:
            cmd["langs"] = langs
        if filler:
            cmd["filler"] = filler
        self._send(**cmd)
        t_end = None
        deadline = timeout or (wait + max_s + 45)
        while True:
            e = self._wait(("listening", "speech_start", "speech_end", "heard"), deadline)
            if e["event"] == "speech_end":
                t_end = e["_t"]
            if on_event:
                on_event(e)
            if e["event"] == "heard":
                e["t_end"] = t_end or e["_t"]
                return e

    def endpoint(self):
        self._send(cmd="endpoint")

    # ----- the Olingo device as microphone and speaker -----
    def set_agent_audio(self, on: bool, mic: bool = True):
        """Speech is no longer played on this Mac but handed to on_speaker(); the microphone is whatever feed_mic() gets.
        mic=False: only the speaker moves to the device, this Mac keeps listening with its own microphone."""
        self.agent_audio = on
        self._send(cmd="audio_mode", agent=on, mic=bool(on and mic))

    def feed_mic(self, pcm16: bytes):
        """16 kHz mono PCM16 from the device's microphone."""
        try:
            self._send(cmd="mic", pcm=base64.b64encode(pcm16).decode())
        except OSError:
            pass

    def forget_voice(self):
        """Erase the learned voice. It only ever lived in the bridge's memory."""
        try:
            self._send(cmd="forget_voice")
        except OSError:
            pass

    def voice_test(self, path: str, mode: str = "check", timeout: float = 30.0) -> dict:
        self._send(cmd="voice_test", file=path, mode=mode)
        return self._wait(("voice",), timeout)

    def transcribe(self, path: str, lang: str = "en", timeout: float = 30.0, fast: bool = False) -> dict:
        self._send(cmd="transcribe", file=path, lang=lang, fast=fast)
        return self._wait(("transcribed",), timeout)

    def stop(self):
        """Cancel everything that is playing, queued or listening."""
        if self.proc and self.proc.poll() is None:
            try:
                self._send(cmd="stop")
            except OSError:
                pass

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self._send(cmd="quit")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    @property
    def pid(self):
        return self.proc.pid if self.proc else None


class MockSpeech:
    """No audio, no microphone. `listen` returns text pushed with .type(); speech just completes at once."""
    pid = None

    def __init__(self, **_):
        self.typed = queue.Queue()
        self.said_log = []
        self.voices = []
        self.on_speaker = self.on_flush = None
        self.agent_audio = False
        self.ready_info = {"whisper": "mock", "tts": "mock", "audio_out": True, "audio_in": True}

    def start(self, ready_timeout=0):
        return 0.0

    def alive(self, timeout=0):
        return {"event": "pong", "audio_out": True, "audio_in": True}

    def key(self, text, lang="en", speaker=None):
        return Path("/nonexistent")

    def speak(self, text, lang="en", rate=1.0, on_start=None, on_done=None, speaker=None):
        self.said_log.append((text, lang))
        self.voices.append(speaker)
        h = Handle(text, lang, "mock", on_start, on_done)
        h._start({})
        h._done({"ms": 0, "audio_s": 0.0})
        return h

    def render(self, text, lang="en", timeout=0, speaker=None):
        return False

    def type(self, text: str):
        self.typed.put(text)

    def listen(self, lang="en", alt=None, wait=10.0, max_s=12.0, langs=None, filler=None, ptt=False, fast=False,
               min_level=0.0, hint=None, voice=None, voice_min=None, voice_learn=None, on_event=None, timeout=None):
        if on_event:
            on_event({"event": "listening"})
        try:
            text = self.typed.get(timeout=wait)
        except queue.Empty:
            text = None
        if text is None or text == "\x00":
            return {"event": "heard", "text": "", "audio_s": 0.0, "ms": 0, "words": [], "t_end": time.time(),
                    "cancelled": text == "\x00", "timeout": text is None}
        cjk = any("一" <= c <= "鿿" for c in text)
        got = (alt if cjk and alt in ("zh", "ja") else lang) if lang != "auto" else ("zh" if cjk else "en")
        words = [{"w": w, "p": 0.9} for w in (list(text) if cjk else text.split())]
        if on_event:
            on_event({"event": "speech_start"})
            on_event({"event": "speech_end"})
        return {"event": "heard", "text": text, "lang": got, "audio_s": 1.0, "ms": 100, "detect_ms": 0, "avg_p": 0.9,
                "words": words, "t_end": time.time(), "level": 0.05}

    def endpoint(self):
        pass

    def forget_voice(self):
        pass

    def set_agent_audio(self, on, mic=True):
        self.agent_audio = on

    def feed_mic(self, pcm16):
        pass

    def stop(self):
        self.typed.put("\x00")

    def close(self):
        pass
