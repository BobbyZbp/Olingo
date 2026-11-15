#!/usr/bin/env python3
"""Olingo device: the eyes, ears and mouth of Olingo. Runs on a Raspberry Pi 5 (or, with --simulate, on any computer).

It does no thinking. It sends what its camera sees and what its microphone hears to the computer that runs Olingo,
and plays the speech that comes back. Nothing is stored on the device: frames and audio pass through memory only.

  python3 olingo_agent.py --bluetooth            # Bluetooth serial (RFCOMM) server, channel 1  [Raspberry Pi]
  python3 olingo_agent.py --tcp 0.0.0.0:8770     # the same link over a cable (USB-C network or Ethernet)
  python3 olingo_agent.py --simulate --tcp 127.0.0.1:8770   # no hardware: test pictures, silence, no sound

One file, standard library only. Camera: Picamera2 if present, otherwise OpenCV. Microphone and speaker: ALSA
(arecord / aplay), so any USB or I2S microphone and any speaker the Pi's sound system knows will do.

The link is a byte stream of frames:   A5 5A | type (1 byte) | length (4 bytes, big endian) | payload
  device -> computer   H  hello, JSON          V  one JPEG picture          M  microphone, PCM16 mono 16 kHz
  computer -> device   S  speech, PCM16 mono 24 kHz        C  control, JSON: {"cmd": "flush" | "ping" | "camera" | "codec", ...}
  Over Bluetooth the computer asks for mu-law sound (types m and s: 8 bits a sample instead of 16).
"""
import argparse
import io
import json
import os
import queue
import socket
import struct
import subprocess
import sys
import threading
import time

MAGIC = b"\xa5\x5a"
MIC_RATE, SPK_RATE = 16000, 24000


def pack(kind: bytes, payload: bytes) -> bytes:
    return MAGIC + kind + struct.pack(">I", len(payload)) + payload


def _ulaw():
    """(encode, decode) between PCM16 and 8-bit mu-law, or (None, None) when neither audioop nor numpy is here."""
    try:
        import audioop                                    # Python up to 3.12 (Raspberry Pi OS)
        return (lambda b: audioop.lin2ulaw(b, 2)), (lambda b: audioop.ulaw2lin(b, 2))
    except ImportError:
        pass
    try:
        import numpy as np

        def enc(b):
            x = np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0
            return ((np.sign(x) * np.log1p(255 * np.abs(x)) / np.log(256.0) + 1) * 127.5).round().astype(np.uint8).tobytes()

        def dec(b):
            y = np.frombuffer(b, dtype=np.uint8).astype(np.float32) / 127.5 - 1
            return (np.clip(np.sign(y) * (np.power(256.0, np.abs(y)) - 1) / 255, -1, 1) * 32767).astype("<i2").tobytes()
        return enc, dec
    except ImportError:
        return None, None


def read_frame(read) -> tuple:
    """read(n) -> exactly n bytes or raises EOFError. Resynchronises on the two magic bytes."""
    a = read(1)
    while True:
        if a != MAGIC[:1]:
            a = read(1)
            continue
        b = read(1)
        if b == MAGIC[1:]:
            break
        a = b
    kind = read(1)
    (n,) = struct.unpack(">I", read(4))
    if n > 8_000_000:
        raise EOFError("frame too large")
    return kind, read(n)


# ---------------------------------------------------------------- camera
class PiCamera:
    """Raspberry Pi camera module (CSI) through Picamera2."""
    def __init__(self, width, height):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"}))
        self.cam.start()
        time.sleep(0.5)

    def jpeg(self, quality):
        from PIL import Image
        im = Image.fromarray(self.cam.capture_array()[:, :, ::-1])
        b = io.BytesIO()
        im.save(b, "JPEG", quality=quality)
        return b.getvalue()


class CvCamera:
    """Any USB webcam through OpenCV."""
    def __init__(self, width, height, index=0):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError("no camera")

    def jpeg(self, quality):
        ok, frame = self.cap.read()
        if not ok:
            return None
        ok, buf = self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None


class SimCamera:
    """No hardware: the pictures in a folder, one after the other (or a moving test card)."""
    def __init__(self, width, height, folder=None):
        from PIL import Image
        self.Image, self.size, self.files, self.t0 = Image, (width, height), [], time.time()
        if folder and os.path.isdir(folder):
            self.files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
        self.hold = None                                  # set by the test harness: show this picture

    def jpeg(self, quality):
        if self.hold:
            im = self.Image.open(self.hold).convert("RGB").resize(self.size)
        elif self.files:
            im = self.Image.open(self.files[int((time.time() - self.t0) / 4) % len(self.files)]).convert("RGB").resize(self.size)
        else:
            k = int((time.time() - self.t0) * 20) % 255
            im = self.Image.new("RGB", self.size, (k, 120, 255 - k))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=quality)
        return b.getvalue()


# ---------------------------------------------------------------- microphone and speaker
class AlsaMic:
    def __init__(self, device=None):
        # small ALSA buffers: with the defaults, sound sat in buffers for over a second (measured on a Pi 5)
        self.cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(MIC_RATE), "-c", "1", "-t", "raw",
                    "--buffer-time=80000", "--period-time=20000"] + (["-D", device] if device else [])
        self.p = None

    def start(self):
        """Called when a computer connects: recording starts now, so no old sound is waiting in a pipe."""
        self.stop()
        self.p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, bufsize=0)

    def stop(self):
        p, self.p = self.p, None
        if p:
            p.kill()
            p.wait()

    def read(self, n):
        p = self.p
        return p.stdout.read(n) if p else b""


class SimMic:
    """Silence, in real time. `say(path)` plays a 16 kHz mono 16-bit wav into it, as if someone spoke."""
    def __init__(self):
        self.q = queue.Queue()
        self.next = time.time()

    def say(self, path):
        import wave
        with wave.open(path, "rb") as w:
            assert w.getframerate() == MIC_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16 kHz mono 16-bit"
            data = w.readframes(w.getnframes())
        for i in range(0, len(data), 640):
            self.q.put(data[i:i + 640])

    def read(self, n):
        self.next = max(self.next + n / 2 / MIC_RATE, time.time() - 0.2)
        time.sleep(max(0, self.next - time.time()))
        try:
            b = self.q.get_nowait()
        except queue.Empty:
            b = b""
        return b + b"\x00" * (n - len(b))


class Speaker:
    """Speech arrives faster than it is spoken, so it waits in a queue here: the link is never held up by the
    loudspeaker, and "flush" (Olingo was cut off) empties the queue at once."""
    TAIL = 0.25                                           # the room still rings this long after the last sample

    def __init__(self):
        self.q = queue.Queue()
        self.until = 0.0                                  # when everything received so far will have been spoken
        self.gen = 0
        threading.Thread(target=self._run, daemon=True).start()

    def play(self, pcm):
        self.until = max(self.until, time.time()) + len(pcm) / 2 / SPK_RATE
        self.q.put((self.gen, pcm))

    def flush(self):
        self.gen += 1                                     # whatever is still queued belongs to the old sentence
        self.until = 0.0
        self._silence()

    def busy(self) -> bool:
        """True while the loudspeaker is talking: the microphone must not hear Olingo's own voice."""
        return time.time() < self.until + self.TAIL

    def _run(self):
        while True:
            gen, pcm = self.q.get()
            if gen == self.gen:
                self._write(pcm)

    def _write(self, pcm):
        pass

    def _silence(self):
        pass


class AlsaSpeaker(Speaker):
    def __init__(self, device=None):
        self.cmd = ["aplay", "-q", "-f", "S16_LE", "-r", str(SPK_RATE), "-c", "1", "-t", "raw",
                    "--buffer-time=120000", "--period-time=20000"] + (["-D", device] if device else [])
        self.p = None
        super().__init__()

    def _write(self, pcm):
        p = self.p
        if p is None or p.poll() is not None:
            p = self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, bufsize=0)
        try:
            p.stdin.write(pcm)
        except (BrokenPipeError, OSError):
            self.p = None

    def _silence(self):
        p, self.p = self.p, None
        if p:
            p.kill()


class SimSpeaker(Speaker):
    def __init__(self):
        self.seconds, self.flushes = 0.0, 0
        super().__init__()

    def _write(self, pcm):
        self.seconds += len(pcm) / 2 / SPK_RATE

    def _silence(self):
        self.flushes += 1


# ---------------------------------------------------------------- transports
def serve_tcp(host, port, iface=None):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if iface:                                             # Linux: only connections that arrive on this interface
        s.setsockopt(socket.SOL_SOCKET, 25, iface.encode())            # 25 = SO_BINDTODEVICE
    s.bind((host, port))
    s.listen(1)
    return s, f"tcp {host}:{port}" + (f" on {iface} only" if iface else "")


def serve_bluetooth(channel):
    """Bluetooth serial (RFCOMM). Linux only. Pair the computer with this device first (see agent/README.md)."""
    s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    s.bind((socket.BDADDR_ANY, channel))
    s.listen(1)
    # advertise the Serial Port service so the computer creates a serial port for it
    subprocess.run(["sdptool", "add", f"--channel={channel}", "SP"], capture_output=True)
    return s, f"bluetooth serial, channel {channel}"


# ---------------------------------------------------------------- one connection
def session(conn, cam, mic, spk, args, log):
    send_lock = threading.Lock()
    alive = threading.Event()
    alive.set()
    cfg = {"fps": args.fps, "quality": args.quality, "ulaw": False}
    enc, dec = _ulaw()
    try:                                                  # a small send buffer: on a slow link we notice at once,
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32768)     # instead of sound queueing up for seconds
    except OSError:
        pass

    def send(kind, payload):
        with send_lock:
            conn.sendall(pack(kind, payload))

    def read(n):
        buf = b""
        while len(buf) < n:
            c = conn.recv(n - len(buf))
            if not c:
                raise EOFError
            buf += c
        return buf

    def camera_loop():
        while alive.is_set():
            t = time.time()
            took = 0.0
            try:
                j = cam.jpeg(cfg["quality"])
                if j:
                    t1 = time.time()
                    send(b"V", j)
                    took = time.time() - t1
            except (OSError, EOFError):
                alive.clear()
            except Exception as e:
                log(f"camera: {e}")
                time.sleep(1)
            # sound comes first: pictures may use at most a third of the link, so on a slow link they get rarer
            gap = max(1 / max(cfg["fps"], 0.2), took * 3)
            time.sleep(max(0, gap - (time.time() - t)))

    def mic_loop():
        while alive.is_set():
            try:
                b = mic.read(1600)                         # 50 ms
                if not b:
                    time.sleep(0.05)
                    continue
                if spk.busy() and not args.duplex:
                    b = bytes(len(b))                      # Olingo is talking: do not send its own voice back
                send(b"m", enc(b)) if cfg["ulaw"] else send(b"M", b)
            except (OSError, EOFError):
                alive.clear()

    send(b"H", json.dumps({"name": args.name, "camera": [args.width, args.height], "fps": args.fps,
                           "mic_rate": MIC_RATE, "spk_rate": SPK_RATE, "simulated": args.simulate}).encode())
    if hasattr(mic, "start"):
        mic.start()
    for f in (camera_loop, mic_loop):
        threading.Thread(target=f, daemon=True).start()
    try:
        while alive.is_set():
            kind, payload = read_frame(read)
            if kind == b"S":
                spk.play(payload)
            elif kind == b"s" and dec:
                spk.play(dec(payload))
            elif kind == b"C":
                c = json.loads(payload or b"{}")
                if c.get("cmd") == "flush":
                    spk.flush()
                elif c.get("cmd") == "ping":
                    send(b"C", json.dumps({"cmd": "pong", "t": c.get("t")}).encode())
                elif c.get("cmd") == "codec":
                    cfg["ulaw"] = c.get("sound") == "ulaw" and enc is not None
                elif c.get("cmd") == "camera":
                    cfg["fps"] = float(c.get("fps", cfg["fps"]))
                    cfg["quality"] = int(c.get("quality", cfg["quality"]))
    except (EOFError, OSError):
        pass
    finally:
        alive.clear()
        spk.flush()
        if hasattr(mic, "stop"):
            mic.stop()


def main():
    ap = argparse.ArgumentParser(description="Olingo device: camera, microphone and speaker for Olingo")
    ap.add_argument("--bluetooth", action="store_true", help="serve over Bluetooth serial (RFCOMM)")
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--tcp", help="serve over TCP, host:port")
    ap.add_argument("--iface", help="with --tcp: accept connections on this network interface only, e.g. usb0 (the cable)")
    ap.add_argument("--simulate", action="store_true", help="no hardware: test pictures, silent microphone, no sound")
    ap.add_argument("--frames", help="simulate: folder of pictures to show")
    ap.add_argument("--name", default="Olingo")
    ap.add_argument("--width", type=int, default=512, help="512x288 is the size Bonsai was measured on")
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--fps", type=float, default=4.0, help="pictures per second (over Bluetooth the computer asks for 2)")
    ap.add_argument("--quality", type=int, default=60)
    ap.add_argument("--mic", help="ALSA capture device, e.g. plughw:1,0")
    ap.add_argument("--speaker", help="ALSA playback device")
    ap.add_argument("--duplex", action="store_true",
                    help="keep the microphone open while the speaker talks (only with a headset: otherwise Olingo hears itself)")
    args = ap.parse_args()

    def log(m):
        print(f"[olingo-agent] {m}", file=sys.stderr, flush=True)

    if args.simulate:
        cam, mic, spk = SimCamera(args.width, args.height, args.frames), SimMic(), SimSpeaker()

        def control():                                    # test harness: "say file.wav", "show picture.jpg", "stats"
            for line in sys.stdin:
                cmd, _, arg = line.strip().partition(" ")
                if cmd == "say":
                    mic.say(arg)
                elif cmd == "show":
                    cam.hold = arg or None
                elif cmd == "stats":
                    print(json.dumps({"spoken_s": round(spk.seconds, 2), "flushes": spk.flushes}), flush=True)
        threading.Thread(target=control, daemon=True).start()
    else:
        try:
            cam = PiCamera(args.width, args.height)
            log("camera: Raspberry Pi camera module")
        except Exception as e:
            cam = CvCamera(args.width, args.height)
            log(f"camera: USB camera (no Pi camera: {e})")
        mic, spk = AlsaMic(args.mic), AlsaSpeaker(args.speaker)

    servers = []
    if args.bluetooth:
        try:
            servers.append(serve_bluetooth(args.channel))
        except (OSError, AttributeError) as e:            # no Bluetooth here: the cable may still work
            log(f"Bluetooth is not available: {e}")
    if args.tcp:
        host, _, port = args.tcp.rpartition(":")
        try:
            servers.append(serve_tcp(host or "0.0.0.0", int(port), args.iface))
        except OSError as e:                              # e.g. the cable interface does not exist (yet)
            log(f"TCP is not available: {e}")
    if not servers:
        if args.bluetooth or args.tcp:
            log("no way to be reached; trying again in 5 s")
            time.sleep(5)
            sys.exit(1)
        ap.error("give --bluetooth and/or --tcp host:port")

    def accept_loop(srv, label):
        log(f"waiting on {label}")
        while True:
            conn, peer = srv.accept()
            log(f"connected: {peer}")
            session(conn, cam, mic, spk, args, log)
            log("disconnected")
            try:
                conn.close()
            except OSError:
                pass
    for srv, label in servers[1:]:
        threading.Thread(target=accept_loop, args=(srv, label), daemon=True).start()
    accept_loop(*servers[0])


if __name__ == "__main__":
    main()
