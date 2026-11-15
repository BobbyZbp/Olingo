#!/usr/bin/env python3
"""Two accelerators, at the same time. Bonsai runs on the GPU (Metal, llama.cpp); Argmax WhisperKit and TTSKit run
their Core ML models on the Neural Engine. This measures each alone and then together: if they shared one
accelerator, both would slow down when run at once. Writes what it measured to docs/accelerators.log.
Run while the Bonsai servers are up (scripts/run_llms.sh); it starts its own speech bridge."""
import json
import statistics
import sys
import struct
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import brain  # noqa: E402
from backend.speech import BridgeSpeech  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N = 3
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


def bonsai(tier="9B", tokens=160) -> float:
    """Tokens per second of one text generation."""
    body = {**brain.COMMON, "messages": [{"role": "user", "content": "Write eighty short numbered sentences about fruit."}],
            "max_tokens": tokens, "id_slot": 1 if tier == "9B" else 0, "cache_prompt": False}
    req = urllib.request.Request(brain.URLS[tier] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["timings"]["predicted_per_second"]


made = []


def wav_seconds(path) -> float:
    """Length of a WAV file (the bridge writes 32-bit float, which the standard library reader does not open)."""
    b = Path(path).read_bytes()
    i, rate, block, data = 12, 0, 0, 0
    while i + 8 <= len(b):
        tag, n = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if tag == b"fmt ":
            rate, block = struct.unpack("<I", b[i + 12:i + 16])[0], struct.unpack("<H", b[i + 20:i + 22])[0]
        elif tag == b"data":
            data = min(n, len(b) - i - 8)
        i += 8 + n + (n & 1)
    return data / block / rate if rate and block else 0.0


def tts(sp, i) -> tuple:
    """(seconds of speech made, seconds it took) for a sentence that is not in the cache."""
    text = f"Measurement number {i}: the orange on the table is small, round and sweet, and I would like to eat it."
    p = sp.key(text, "en")
    if p.exists():
        p.unlink()
    t = time.time()
    sp.render(text, "en", timeout=60)
    took = time.time() - t
    made.append(p)
    return wav_seconds(p), took


def whisper(sp, wav) -> float:
    return sp.transcribe(str(wav), "en").get("ms", 0) / 1000


def med(xs):
    return statistics.median(xs)


def main():
    sp = BridgeSpeech(log_path="/tmp/olingo_accel_bridge.log")
    sp.start(ready_timeout=300)
    r = sp.ready_info
    out(f"Olingo Live: two accelerators at once, {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"Bonsai-2 9B: llama.cpp, Metal GPU (-ngl 99). Argmax: Whisper {r.get('whisper')}, TTS {r.get('tts')}, Core ML on the Neural Engine.")
    bonsai(), tts(sp, 0)                                    # warm both
    alone_b = [bonsai() for _ in range(N)]
    alone_t = [tts(sp, 10 + i) for i in range(N)]
    alone_w = [whisper(sp, made[-1]) for _ in range(N)]
    out(f"alone     Bonsai-2 9B {med(alone_b):5.1f} tok/s | TTSKit {med([a / t for a, t in alone_t]):4.2f}x real time | WhisperKit {med(alone_w) * 1000:4.0f} ms for {alone_t[-1][0]:.1f} s of speech")
    both_b, both_t, both_w = [], [], []
    for i in range(N):
        res = {}
        th = threading.Thread(target=lambda: res.update(b=bonsai(tokens=700)))
        th.start()
        time.sleep(0.3)                                    # Bonsai is generating from here on
        t = tts(sp, 20 + i)
        w = whisper(sp, made[-1])
        busy = th.is_alive()                               # was Bonsai still generating when the speech work ended?
        th.join()
        if busy:
            both_b.append(res["b"]); both_t.append(t); both_w.append(w)
        else:
            out("  (a round was left out: Bonsai had finished before the speech work did)")
    if not both_b:
        out("no round overlapped: nothing to report")
        sp.close()
        return
    out(f"together  Bonsai-2 9B {med(both_b):5.1f} tok/s | TTSKit {med([a / t for a, t in both_t]):4.2f}x real time | WhisperKit {med(both_w) * 1000:4.0f} ms")
    out(f"change    Bonsai-2 9B {100 * (med(both_b) / med(alone_b) - 1):+.0f}% | TTSKit {100 * (med([a / t for a, t in both_t]) / med([a / t for a, t in alone_t]) - 1):+.0f}% | WhisperKit {100 * (med(both_w) / med(alone_w) - 1):+.0f}% time")
    out(f"(median of {N} runs alone, {len(both_b)} together; 'together' = speech synthesised and transcribed while Bonsai is generating)")
    for p in made:
        if p.exists():
            p.unlink()
    sp.close()
    (ROOT / "docs" / "accelerators.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
