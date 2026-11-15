#!/usr/bin/env python3
"""Run with the network off (scripts/offline_check.sh does that): every part of a call, each with PASS / FAIL and
its measured time. Starts its own speech bridge, so the models are loaded cold, with no network to fall back on."""
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image  # noqa: E402
from backend import brain, privacy  # noqa: E402
from backend.speech import BridgeSpeech  # noqa: E402
from backend.vision import to_jpeg  # noqa: E402

results = []


def check(name, fn):
    t = time.time()
    try:
        ok, note = fn()
    except Exception as e:
        ok, note = False, f"{type(e).__name__}: {e}"
    ms = round((time.time() - t) * 1000)
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:44} {ms:6} ms  {note}", flush=True)


def offline():
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
        return False, "the internet is reachable: this run does not prove anything"
    except OSError:
        return True, "no route to the internet"


def bonsai(tier, image):
    def run():
        jpeg = to_jpeg(Image.new("RGB", (512, 288), (40, 90, 200))) if image else None
        d, i = brain.answer("What color do you see?" if image else "I had rice for lunch.", jpeg, tier=tier)
        return d is not None, (f"{d['say']!r} ({i['predicted_n']} tok at {i['tok_s']} tok/s)" if d else i.get("error", ""))
    return run


sp = BridgeSpeech(log_path="/tmp/olingo_offline_bridge.log")


def bridge():
    s = sp.start(ready_timeout=300)
    r = sp.ready_info
    return bool(r.get("audio_out") and r.get("audio_in")), f"Whisper {r.get('whisper')} + {r.get('whisper_fast') or 'no'} fast + TTS {r.get('tts')} loaded in {s:.1f} s, audio out {r.get('audio_out')}, in {r.get('audio_in')}"


TEXT = "This Mac is offline right now."
WAV = sp.key(TEXT, "en")


def synth():
    if WAV.exists():
        WAV.unlink()
    return sp.render(TEXT, "en", timeout=60), f"wrote {WAV.name}"


def play():
    h = sp.speak(TEXT, "en")
    h.done.wait(20)
    return h.kind == "cached" and not h.info.get("stopped"), f"cached wav, audible after {h.ttfa_ms} ms"


def live_zh():
    h = sp.speak("你好，我是 Olingo。", "zh", speaker="serena")
    h.done.wait(20)
    return not h.info.get("stopped") and h.t_start is not None, f"audible after {h.ttfa_ms} ms"


def hear():
    r = sp.transcribe(str(WAV), "en")
    return "offline" in r.get("text", "").lower(), f"{r.get('text')!r} in {r.get('ms')} ms"


def audit():
    pids = [p for p in [sp.pid, *privacy.llama_pids()] if p]
    ext = privacy.external_sockets(pids)
    return not ext, f"{len(ext)} external sockets across {len(pids)} processes" + (f": {ext[:2]}" if ext else "")


print(f"Olingo Live offline check, {time.strftime('%Y-%m-%d %H:%M:%S')}")
check("no internet", offline)
check("Bonsai 9B sees an image and answers", bonsai("9B", True))
check("Bonsai 4B sees an image and answers", bonsai("4B", True))
check("Bonsai 4B answers small talk", bonsai("4B", False))


def first_look():
    things, i = brain.room(to_jpeg(Image.open(Path(__file__).resolve().parent.parent / "web" / "sample.jpg")) if False else
                           to_jpeg(Image.new("RGB", (512, 288), (200, 60, 40))))
    return "error" not in i, f"{len(things)} things named in {i.get('ms')} ms"


def phrase():
    p, i = brain.phrases("apple", "苹果")
    return bool(p), f"{[x['l2'] for x in p]} in {i.get('ms')} ms"


def device():
    """The Olingo device link, against the simulated device on this Mac: pictures, microphone, speech, link delay."""
    import subprocess
    from backend import agent
    root = Path(__file__).resolve().parent.parent
    sim = subprocess.Popen([sys.executable, str(root / "agent" / "olingo_agent.py"), "--simulate", "--tcp", "127.0.0.1:8779"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    frames, mic = [], []
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", 8779), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        link = agent.AgentLink("tcp://127.0.0.1:8779", on_frame=frames.append, on_mic=mic.append)
        link.connect()
        link.speaker(bytes(24000 * 2))
        t0 = time.time()
        while time.time() - t0 < 5 and (len(frames) < 3 or sum(map(len, mic)) < 32000 or link.stats["delay_ms"] is None):
            time.sleep(0.05)
        sim.stdin.write("stats\n")
        sim.stdin.flush()
        spoken = __import__("json").loads(sim.stdout.readline())["spoken_s"]
        d = link.stats["delay_ms"]
        link.close()
        ok = len(frames) >= 3 and sum(map(len, mic)) >= 32000 and spoken > 0.9 and d is not None
        return ok, f"{len(frames)} pictures, {sum(map(len, mic)) / 32000:.1f} s of microphone in, {spoken} s of speech out, link delay {d} ms"
    finally:
        sim.kill()


check("Bonsai 9B first look at the room (helper slot)", first_look)
check("Bonsai 9B phrases for a new word", phrase)
check("Argmax bridge cold start, local models only", bridge)
check("TTSKit synthesises a new sentence", synth)
check("cached sentence plays", play)
check("TTSKit speaks Chinese live", live_zh)
check("WhisperKit transcribes it back", hear)
check("Olingo device link (simulated device)", device)
check("no socket leaves this Mac", audit)
sp.close()
print(f"{sum(results)} of {len(results)} passed")
sys.exit(0 if all(results) else 1)
