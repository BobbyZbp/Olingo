"""The link to the Olingo device (a Raspberry Pi with a camera, a microphone and a speaker).

The computer stays the brain. The device is the eyes, ears and mouth: its pictures replace the laptop camera, its
microphone replaces the laptop microphone, and Olingo's speech is played on its speaker.

The link is a byte stream, so the same code runs over
  bt://<address>            Bluetooth serial of a paired device, asked for directly (bridge/bt/olingo_bt.swift)
  serial:/dev/cu.<name>     Bluetooth serial through the port macOS made when the device was paired (can go stale)
  tcp://host:port           a cable (USB-C network, Ethernet), or the simulated device on this computer
Two devices are understood, told apart by the first byte they send:
  the Olingo device   agent/olingo_agent.py (frames start with A5 5A)
  Luma                the team's own relay on the Raspberry Pi, ~/LUMA (frames: type, length, payload; hello = type 1)
Frames and audio only pass through memory."""
import collections
import glob
import json
import os
import socket
import re
import struct
import subprocess
import termios
import threading
import time
import tty

MAGIC = b"\xa5\x5a"


def ulaw_encode(pcm16: bytes) -> bytes:
    """16-bit sound -> 8-bit mu-law (the telephone codec): half the bytes, speech stays clear."""
    import numpy as np
    x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
    y = np.sign(x) * np.log1p(255 * np.abs(x)) / np.log(256.0)
    return ((y + 1) * 127.5).round().astype(np.uint8).tobytes()


def ulaw_decode(u8: bytes) -> bytes:
    import numpy as np
    y = np.frombuffer(u8, dtype=np.uint8).astype(np.float32) / 127.5 - 1
    x = np.sign(y) * (np.power(256.0, np.abs(y)) - 1) / 255
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


BUILT_IN_PORTS = ("Bluetooth-Incoming-Port", "debug-console", "wlan-debug")
LIKELY = ("olingo", "raspberry", "rasp", "pi", "luma", "wali")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT_HELPER = os.path.join(ROOT, "bridge", ".build", "olingo-bt")
# Luma's message types (luma/protocol.py on the Pi)
L_HELLO, L_PHOTO, L_MIC, L_STATUS, L_SPEAKER, L_SPEAKER_STOP, L_CONFIG = 0x01, 0x02, 0x03, 0x04, 0x10, 0x11, 0x12
L_MIC_ADPCM, L_SPEAKER_ADPCM = 0x06, 0x14          # compressed sound, added to Luma for slow links (same code there)


# --- compressed audio: IMA ADPCM, 4 bits a sample (a quarter of 16-bit PCM), for slow Bluetooth links ---
# A block is: predictor (2 bytes, signed, big endian) | step index (1 byte) | codes, two per byte, low nibble first.
# The predictor and index are the coder's state at the start of the block, so every block can be decoded on its
# own: a block lost on a busy link costs only its own 20 ms.

_ADPCM_STEPS = (7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80,
                88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598,
                658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
                3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289,
                16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767)
_ADPCM_INDEX = (-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8)
_ADPCM_HEAD = struct.Struct(">hB")


def adpcm_encode(pcm: bytes, state: tuple = (0, 0)) -> tuple:
    """16-bit little-endian PCM (an even number of samples) -> (block, state for the next call)."""
    samples = struct.unpack("<%dh" % (len(pcm) // 2), pcm[:len(pcm) // 2 * 2])
    pred, index = state
    steps, table = _ADPCM_STEPS, _ADPCM_INDEX
    out = bytearray(_ADPCM_HEAD.pack(pred, index))
    low = None
    for s in samples:
        step = steps[index]
        diff = s - pred
        code = 0
        if diff < 0:
            code = 8
            diff = -diff
        delta = step >> 3
        if diff >= step:
            code |= 4
            diff -= step
            delta += step
        half = step >> 1
        if diff >= half:
            code |= 2
            diff -= half
            delta += half
        quarter = step >> 2
        if diff >= quarter:
            code |= 1
            delta += quarter
        pred = pred - delta if code & 8 else pred + delta
        if pred > 32767:
            pred = 32767
        elif pred < -32768:
            pred = -32768
        index += table[code]
        if index < 0:
            index = 0
        elif index > 88:
            index = 88
        if low is None:
            low = code
        else:
            out.append(low | (code << 4))
            low = None
    if low is not None:
        out.append(low)
    return bytes(out), (pred, index)


def adpcm_decode(block: bytes) -> bytes:
    """One block from adpcm_encode -> 16-bit little-endian PCM."""
    if len(block) < _ADPCM_HEAD.size:
        return b""
    pred, index = _ADPCM_HEAD.unpack_from(block)
    index = min(max(index, 0), 88)
    steps, table = _ADPCM_STEPS, _ADPCM_INDEX
    out = []
    for byte in block[_ADPCM_HEAD.size:]:
        for code in (byte & 15, byte >> 4):
            step = steps[index]
            delta = step >> 3
            if code & 4:
                delta += step
            if code & 2:
                delta += step >> 1
            if code & 1:
                delta += step >> 2
            pred = pred - delta if code & 8 else pred + delta
            if pred > 32767:
                pred = 32767
            elif pred < -32768:
                pred = -32768
            index += table[code]
            if index < 0:
                index = 0
            elif index > 88:
                index = 88
            out.append(pred)
    return struct.pack("<%dh" % len(out), *out)


def bt_helper() -> str:
    """The small program that opens the Bluetooth serial channel. Built from source the first time it is needed."""
    src = os.path.join(ROOT, "bridge", "bt", "olingo_bt.swift")
    if not os.path.exists(BT_HELPER) or os.path.getmtime(BT_HELPER) < os.path.getmtime(src):
        os.makedirs(os.path.dirname(BT_HELPER), exist_ok=True)
        subprocess.run(["swiftc", "-O", src, "-o", BT_HELPER], check=True, capture_output=True, timeout=180)
    return BT_HELPER


def paired_devices() -> list:
    """[(name, address)] of the Bluetooth devices this Mac is paired with."""
    try:
        out = subprocess.run(["system_profiler", "SPBluetoothDataType"], capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return []
    return [(m.group(1).strip(), m.group(2)) for m in
            re.finditer(r"^ {10}(\S[^\n]*):\n {14}Address: ([0-9A-F:]{17})", out, re.M)]


class Down16:
    """Speech is made at 24 kHz; Luma's speaker takes 16 kHz. Three samples in, two out, seamless across pieces."""
    def __init__(self):
        self.rest = b""

    def feed(self, pcm24: bytes) -> bytes:
        import numpy as np
        b = self.rest + pcm24
        n = len(b) // 6 * 6
        self.rest = b[n:]
        x = np.frombuffer(b[:n], dtype="<i2").astype(np.float32).reshape(-1, 3)
        y = np.empty((len(x), 2), dtype=np.float32)
        y[:, 0] = 0.75 * x[:, 0] + 0.25 * x[:, 1]
        y[:, 1] = 0.5 * (x[:, 1] + x[:, 2])
        return y.astype("<i2").tobytes()


def pack(kind: bytes, payload: bytes) -> bytes:
    return MAGIC + kind + struct.pack(">I", len(payload)) + payload


def candidates() -> list:
    """Where an Olingo device might be: Bluetooth serial ports of paired devices, and the simulator if it runs."""
    out = []
    for name, addr in paired_devices():                    # asked for directly: works after the device restarts
        if any(k in name.lower() for k in LIKELY):
            out.append({"target": "bt://" + addr.replace(":", "-"), "label": f"{name} (Bluetooth)", "likely": True})
    for p in sorted(glob.glob("/dev/cu.*")):
        name = p.split("/dev/cu.", 1)[1]
        if name not in BUILT_IN_PORTS and not any(k in name.lower() for k in LIKELY):
            out.append({"target": "serial:" + p, "label": f"{name} (Bluetooth serial port)", "likely": False})
    cable = os.environ.get("OLINGO_DEVICE", "")            # a device on a cable: OLINGO_DEVICE=tcp://<its address>:8770
    if cable.startswith("tcp://"):
        out.append({"target": cable, "label": f"{cable[6:]} (cable)", "likely": True})
    try:
        socket.create_connection(("127.0.0.1", 8770), timeout=0.2).close()
        out.append({"target": "tcp://127.0.0.1:8770", "label": "Device on port 8770 of this Mac (tunnel or simulator)", "likely": True})
    except OSError:
        pass
    return sorted(out, key=lambda c: not c["likely"])


class AgentLink:
    def __init__(self, target: str, on_frame=None, on_mic=None, on_close=None):
        self.target = target
        self.on_frame, self.on_mic, self.on_close = on_frame, on_mic, on_close
        self.info = {}
        self.alive = False
        self.out = collections.deque()                    # speech, in order
        self.ctl = collections.deque()                    # control messages: they overtake speech
        self.wake = threading.Event()
        self.hello = threading.Event()
        self.peer_ip = None
        self.stats = {"frames": 0, "mic_s": 0.0, "spk_s": 0.0, "bytes_in": 0, "bytes_out": 0, "since": time.time(),
                      "delay_ms": None}
        self.radio = target.startswith(("serial:", "bt://"))
        self.compact = self.radio                         # Bluetooth: mu-law sound, fewer pictures
        self.fps = self.fps_goal = 2.0 if self.compact else 4.0
        self.luma = False                                 # the device at the other end is the team's Luma relay
        self.luma_adpcm = False                           # ... and it has agreed to compressed sound
        self.spk_state = (0, 0)
        self.down = Down16()
        self.proc = None
        self.note = ""                                    # what the Bluetooth helper last reported
        self._read = self._write = self._closer = None

    # ----- opening -----
    def connect(self, timeout: float = 8.0) -> dict:
        if self.target.startswith("tcp://"):
            host, _, port = self.target[6:].rpartition(":")
            ip = socket.getaddrinfo(host, int(port), socket.AF_INET, socket.SOCK_STREAM)[0][4][0]   # a name like olingo.local
            s = socket.create_connection((ip, int(port)), timeout=timeout)
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.peer_ip = s.getpeername()[0]
            self._read, self._write, self._closer = s.recv, s.sendall, s.close
        elif self.target.startswith("serial:"):
            # opening the port is what makes macOS dial the device, and it blocks until the channel is up
            box = {}

            def dial():
                try:
                    box["fd"] = os.open(self.target[7:], os.O_RDWR | os.O_NOCTTY)
                except OSError as e:
                    box["err"] = e
                if box.get("late") and "fd" in box:
                    os.close(box["fd"])
            t = threading.Thread(target=dial, daemon=True)
            t.start()
            t.join(max(timeout, 15.0))
            if "err" in box:
                raise box["err"]
            if "fd" not in box:
                box["late"] = True
                raise TimeoutError("Bluetooth did not connect: is the device on, paired and close by?")
            fd = box["fd"]
            tty.setraw(fd)
            attrs = termios.tcgetattr(fd)
            attrs[6][termios.VMIN], attrs[6][termios.VTIME] = 1, 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            self._read = lambda n: os.read(fd, n)

            def write(b):
                while b:
                    b = b[os.write(fd, b):]
            self._write, self._closer = write, lambda: os.close(fd)
        elif self.target.startswith("bt://"):
            addr, _, chan = self.target[5:].partition("/")
            p = self.proc = subprocess.Popen([bt_helper(), addr, chan or "1"], stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            timeout = max(timeout, 20.0)                   # the radio link alone takes about five seconds

            def status():
                for line in p.stderr:
                    self.note = line.decode("utf-8", "replace").strip()
            threading.Thread(target=status, daemon=True).start()
            self._read = lambda n: os.read(p.stdout.fileno(), n)

            def write(b):
                p.stdin.write(b)
                p.stdin.flush()
            self._write, self._closer = write, p.kill
        else:
            raise ValueError("unknown target: " + self.target)
        self.alive = True
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._writer, daemon=True).start()
        if not self.hello.wait(timeout):
            self.close()
            raise TimeoutError("the device did not answer" + (f" (Bluetooth: {self.note})" if self.note else "")
                               + ": is it switched on and close by?")
        if self.luma:                                      # Luma: its own settings message, no mu-law, no ping
            self.compact = False
            want = {"photo_interval": 1.0, "photo_width": 512, "photo_height": 288, "photo_quality": 60}
            if self.info.get("codec"):                     # this Luma knows compressed sound: a quarter of the bytes
                want["mic_codec"] = "adpcm"
            self.ctl.append((L_CONFIG, json.dumps(want).encode()))
            self.wake.set()
            return self.info
        if self.compact:
            self.control(cmd="codec", sound="ulaw")
        self.control(cmd="camera", fps=self.fps, quality=45 if self.compact else 60)
        threading.Thread(target=self._pinger, daemon=True).start()
        return self.info

    # ----- the link is measured all the time: sound must stay current, so pictures give way on a slow link -----
    def _pinger(self):
        while self.alive:
            self.control(cmd="ping", t=time.time())
            time.sleep(2.0)

    def _pong(self, sent: float):
        delay = time.time() - sent                         # there and back, behind whatever the device was sending
        self.stats["delay_ms"] = round(delay * 1000)
        if delay > 0.5 and self.fps > 0.5:
            self.fps = max(0.5, self.fps * 0.6)
        elif delay < 0.2 and self.fps < self.fps_goal:
            self.fps = min(self.fps_goal, self.fps * 1.3)
        else:
            return
        self.control(cmd="camera", fps=round(self.fps, 2))

    # ----- device -> computer -----
    def _exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            c = self._read(n - len(buf))
            if not c:
                raise EOFError
            buf += c
        self.stats["bytes_in"] += n
        return buf

    def _luma(self, first: bytes):
        """Luma's stream: type (1) | length (4, big endian) | payload. Photos come in pieces; microphone frames
        start with an 8-byte timestamp."""
        kind = first
        photo, total = bytearray(), 0
        while self.alive:
            (n,) = struct.unpack(">I", self._exact(4))
            if n > 8_000_000:
                raise EOFError
            payload = self._exact(n)
            t = kind[0]
            if t in (L_MIC, L_MIC_ADPCM):
                pcm = adpcm_decode(payload[8:]) if t == L_MIC_ADPCM else payload[8:]
                self.stats["mic_s"] += len(pcm) / 2 / 16000
                if self.on_mic:
                    self.on_mic(pcm)
            elif t == L_PHOTO and len(payload) >= 16:
                _, total, offset = struct.unpack(">QII", payload[:16])
                if offset == 0:
                    photo = bytearray()
                if offset == len(photo):
                    photo += payload[16:]
                    if len(photo) == total:
                        self.stats["frames"] += 1
                        if self.on_frame:
                            self.on_frame(bytes(photo))
                        photo = bytearray()
            elif t == L_HELLO:
                h = json.loads(payload or b"{}")
                c = h.get("config") or {}
                self.luma = True
                self.info = {"name": "Luma", "camera": [c.get("photo_width"), c.get("photo_height")],
                             "protocol": h.get("protocol"), "simulated": False, "codec": "mic_codec" in c}
                self.hello.set()
            elif t == L_STATUS:
                st = json.loads(payload or b"{}")
                if st.get("event") == "config":
                    c = st.get("config") or {}
                    self.info["camera"] = [c.get("photo_width"), c.get("photo_height")]
                    self.fps = round(1 / c["photo_interval"], 1) if c.get("photo_interval") else 0
                    self.luma_adpcm = c.get("mic_codec") == "adpcm"
                elif st.get("event") == "error":
                    self.note = "Luma: " + str(st.get("message"))
            kind = self._exact(1)

    def _reader(self):
        try:
            a = self._exact(1)
            if a[0] == L_HELLO:                            # not our magic byte: the device is Luma
                self._luma(a)
            first = True
            while self.alive:
                if not first:
                    a = self._exact(1)
                first = False
                while True:                                # resynchronise on the two magic bytes
                    if a != MAGIC[:1]:
                        a = self._exact(1)
                        continue
                    b = self._exact(1)
                    if b == MAGIC[1:]:
                        break
                    a = b
                kind = self._exact(1)
                (n,) = struct.unpack(">I", self._exact(4))
                if n > 8_000_000:
                    raise EOFError
                payload = self._exact(n)
                if kind == b"V":
                    self.stats["frames"] += 1
                    if self.on_frame:
                        self.on_frame(payload)
                elif kind in (b"M", b"m"):                 # m = mu-law
                    if kind == b"m":
                        payload = ulaw_decode(payload)
                    self.stats["mic_s"] += len(payload) / 2 / 16000
                    if self.on_mic:
                        self.on_mic(payload)
                elif kind == b"H":
                    self.info = json.loads(payload)
                    self.hello.set()
                elif kind == b"C":
                    c = json.loads(payload or b"{}")
                    if c.get("cmd") == "pong" and c.get("t"):
                        self._pong(float(c["t"]))
        except (EOFError, OSError, ValueError):
            pass
        was = self.alive
        self.alive = False
        if was and self.on_close:
            self.on_close()

    # ----- computer -> device -----
    def _writer(self):
        while self.alive:
            self.wake.wait(0.5)
            self.wake.clear()
            while self.alive:
                try:
                    kind, payload = self.ctl.popleft()
                except IndexError:
                    try:
                        kind, payload = self.out.popleft()
                    except IndexError:
                        break
                try:
                    if isinstance(kind, int):              # a Luma message: type, length, payload
                        self._write(struct.pack(">BI", kind, len(payload)) + payload)
                    else:
                        self._write(pack(kind, payload))
                    self.stats["bytes_out"] += len(payload) + 7
                except (OSError, ValueError):
                    self.alive = False

    def speaker(self, pcm16: bytes):
        """Speech for the device to play: PCM16 mono 24 kHz."""
        self.stats["spk_s"] += len(pcm16) / 2 / 24000
        if self.luma:                                      # Luma plays 16 kHz
            pcm = self.down.feed(pcm16)
            for i in range(0, len(pcm), 3200):             # 0.1 s pieces
                if self.luma_adpcm:
                    block, self.spk_state = adpcm_encode(pcm[i:i + 3200], self.spk_state)
                    self.out.append((L_SPEAKER_ADPCM, block))
                else:
                    self.out.append((L_SPEAKER, pcm[i:i + 3200]))
            self.wake.set()
            return
        for i in range(0, len(pcm16), 4800):               # 0.1 s pieces: a flush never waits long
            piece = pcm16[i:i + 4800]
            self.out.append((b"s", ulaw_encode(piece)) if self.compact else (b"S", piece))
        self.wake.set()

    def flush(self):
        """Stop talking at once: drop what has not been sent yet, and tell the device to fall silent."""
        self.out.clear()
        if self.luma:
            self.down, self.spk_state = Down16(), (0, 0)
            self.ctl.append((L_SPEAKER_STOP, b""))
            self.wake.set()
            return
        self.control(cmd="flush")

    def control(self, **cmd):
        if self.luma:
            return                                         # Luma has its own messages (see connect and flush)
        self.ctl.append((b"C", json.dumps(cmd).encode()))
        self.wake.set()

    def summary(self) -> dict:
        s = self.stats
        return {"name": self.info.get("name", "Olingo"), "target": self.target, "simulated": bool(self.info.get("simulated")),
                "over": "Bluetooth" if self.radio else "cable",
                "frames": s["frames"], "mic_s": round(s["mic_s"], 1), "spk_s": round(s["spk_s"], 1),
                "kb_in": round(s["bytes_in"] / 1024), "kb_out": round(s["bytes_out"] / 1024),
                "kbps_in": round(s["bytes_in"] * 8 / 1000 / max(time.time() - s["since"], 1)),
                "delay_ms": s["delay_ms"], "fps": round(self.fps, 1),
                "sound": "compressed" if (self.luma_adpcm or (self.compact and not self.luma)) else "uncompressed"}

    def close(self):
        self.alive = False
        try:
            if self._closer:
                self._closer()
        except OSError:
            pass
