import io, os, sys, unittest
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.gate import Gate

RNG = np.random.default_rng(0)


def frame(obj=None, bright=0.0, noise=1.5, side=384):
    """Synthetic aperture crop: textured background + optional coloured box (x0,y0,x1,y1,gray)."""
    bg = np.full((side, side), 110.0)
    bg += np.sin(np.arange(side) / 17.0)[None, :] * 12 + np.cos(np.arange(side) / 23.0)[:, None] * 10
    img = bg + bright + RNG.normal(0, noise, (side, side))
    if obj:
        x0, y0, x1, y1, v = obj
        img[y0:y1, x0:x1] = v
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def run(gate, frames, fps=8, t0=0.0):
    fires = []
    for i, f in enumerate(frames):
        out = gate.update(f, t0 + i / fps)
        if out is not None:
            fires.append(i)
    return fires


class GateTests(unittest.TestCase):
    def test_empty_never_fires(self):
        g = Gate()
        self.assertEqual(run(g, [frame() for _ in range(40)]), [])
        self.assertEqual(g.ring(), "empty")

    def test_lighting_drift_never_fires(self):
        g = Gate()
        self.assertEqual(run(g, [frame(bright=i * 0.15) for i in range(60)]), [])

    def test_object_fires_once(self):
        g = Gate()
        frames = [frame() for _ in range(6)]
        frames += [frame((40 + 12 * k, 90, 200 + 12 * k, 300, 30)) for k in range(5)]     # sliding in: motion
        frames += [frame((100, 90, 260, 300, 30)) for _ in range(20)]                      # held still
        fires = run(g, frames)
        self.assertEqual(len(fires), 1)
        self.assertGreaterEqual(fires[0], 11)

    def test_waving_hand_does_not_fire(self):
        g = Gate()
        frames = [frame() for _ in range(6)]
        frames += [frame((30 + (k * 53) % 200, 60 + (k * 37) % 120, 190 + (k * 53) % 200, 280, 60)) for k in range(30)]
        self.assertEqual(run(g, frames), [])

    def test_rearm_after_removal(self):
        g = Gate()
        held = [frame((100, 90, 260, 300, 30)) for _ in range(14)]
        frames = [frame() for _ in range(6)] + held + [frame() for _ in range(10)] + \
                 [frame((120, 80, 280, 310, 220)) for _ in range(14)]
        self.assertEqual(len(run(g, frames)), 2)

    def test_swap_without_leaving_rearms(self):
        g = Gate()
        a = [frame((100, 90, 260, 300, 30)) for _ in range(14)]
        moving = [frame((60 + 20 * k, 70, 220 + 20 * k, 290, 120)) for k in range(4)]
        b = [frame((140, 70, 300, 290, 200)) for _ in range(14)]
        self.assertEqual(len(run(g, [frame() for _ in range(6)] + a + moving + b)), 2)

    def test_does_not_refire_while_held(self):
        g = Gate()
        frames = [frame() for _ in range(6)] + [frame((100, 90, 260, 300, 30)) for _ in range(60)]
        self.assertEqual(len(run(g, frames)), 1)

    def test_picks_sharpest(self):
        g = Gate()
        sharp = frame((100, 90, 260, 300, 30), noise=1.5)
        out = None
        for i, f in enumerate([frame()] * 6 + [frame((100, 90, 260, 300, 30), noise=8)] * 2 + [sharp] + [frame((100, 90, 260, 300, 30), noise=8)] * 6):
            r = g.update(f, i / 8)
            out = r or out
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
