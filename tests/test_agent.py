"""The device link, against the simulated device (no hardware, no models): hello, pictures, microphone, speech, flush."""
import io
import json
import os
import socket
import subprocess
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from backend import agent  # noqa: E402


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Codec(unittest.TestCase):
    def test_ulaw_round_trip(self):
        import numpy as np
        t = np.arange(16000) / 16000
        x = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
        y = np.frombuffer(agent.ulaw_decode(agent.ulaw_encode(x.tobytes())), dtype="<i2")
        self.assertEqual(len(y), len(x))
        noise = (x.astype(float) - y.astype(float)).std()
        self.assertGreater(20 * np.log10(x.astype(float).std() / noise), 30)       # dB

    def test_pack(self):
        self.assertEqual(agent.pack(b"V", b"abc"), b"\xa5\x5aV\x00\x00\x00\x03abc")


class Audit(unittest.TestCase):
    """The socket audit: loopback is ours, the cable to the device is reported apart, anything else counts."""
    LSOF = """COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME
Python 1 me 5u IPv4 0x1 0t0 TCP 127.0.0.1:8765 (LISTEN)
Python 1 me 6u IPv4 0x2 0t0 TCP 127.0.0.1:51000->127.0.0.1:8082 (ESTABLISHED)
Python 1 me 7u IPv4 0x3 0t0 TCP 169.254.10.2:51001->169.254.77.5:8770 (ESTABLISHED)
Python 1 me 8u IPv4 0x4 0t0 TCP 10.0.0.5:51002->93.184.216.34:443 (ESTABLISHED)
"""

    def test_device_link_is_not_external(self):
        from backend import privacy
        real = privacy._run
        privacy._run = lambda cmd, timeout=4: self.LSOF
        try:
            self.assertEqual([d for _, d in privacy.external_sockets([1], "169.254.77.5")],
                             ["10.0.0.5:51002->93.184.216.34:443 (ESTABLISHED)"])
            self.assertEqual(len(privacy.external_sockets([1])), 2)        # no device connected: the cable address counts too
        finally:
            privacy._run = real


class Link(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.sim = subprocess.Popen([sys.executable, os.path.join(ROOT, "agent", "olingo_agent.py"), "--simulate",
                                     "--tcp", f"127.0.0.1:{self.port}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)

    def tearDown(self):
        self.sim.kill()
        self.sim.wait()
        self.sim.stdin.close()
        self.sim.stdout.close()

    def stats(self) -> dict:
        self.sim.stdin.write("stats\n")
        self.sim.stdin.flush()
        return json.loads(self.sim.stdout.readline())

    def test_camera_microphone_speaker(self):
        frames, mic = [], []
        link = agent.AgentLink(f"tcp://127.0.0.1:{self.port}", on_frame=frames.append, on_mic=mic.append)
        info = link.connect()
        try:
            self.assertEqual(info["camera"], [512, 288])
            self.assertTrue(info["simulated"])
            link.speaker(bytes(24000 * 2))                     # one second of speech for the device
            t0 = time.time()
            while time.time() - t0 < 5 and (len(frames) < 2 or sum(map(len, mic)) < 16000 or link.stats["delay_ms"] is None):
                time.sleep(0.05)
            self.assertGreaterEqual(len(frames), 2)
            from PIL import Image
            self.assertEqual(Image.open(io.BytesIO(frames[0])).size, (512, 288))
            self.assertGreaterEqual(sum(map(len, mic)), 16000)                 # half a second of PCM16 at 16 kHz
            self.assertLess(link.stats["delay_ms"], 500)
            before = self.stats()
            self.assertAlmostEqual(before["spoken_s"], 1.0, places=1)
            link.speaker(bytes(24000 * 2 * 5))
            link.flush()                                       # cut off: the device must fall silent
            time.sleep(0.3)
            s = self.stats()
            self.assertEqual(s["flushes"], before["flushes"] + 1)
            self.assertLess(s["spoken_s"], 6.0)
            self.assertEqual(link.summary()["over"], "cable")
        finally:
            link.close()


if __name__ == "__main__":
    unittest.main()
