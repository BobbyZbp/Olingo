"""Scene watcher: notices, without any model, that something new is being held still in front of the camera.
Each frame is reduced to a 96x54 grey image (mean brightness removed) and two numbers are computed:
  chg = share of pixels that differ from the quiet background (a slow running average)
  mot = share of pixels that differ from the previous frame
It fires when the picture has changed enough (chg) and then stays still (mot) for a moment.
Once the call has looked, the current picture becomes the background, so the same object does not fire twice."""
import io
import os

import numpy as np
from PIL import Image


def _env(name, default):
    return float(os.environ.get(name, default))


def gray(jpeg: bytes, size=(96, 54)) -> np.ndarray:
    im = Image.open(io.BytesIO(jpeg)).convert("L").resize(size, Image.BILINEAR)
    g = np.asarray(im, dtype=np.float32) / 255.0
    return g - g.mean()


class SceneWatch:
    def __init__(self):
        self.pix = _env("SCENE_PIX", 0.07)          # a pixel counts as changed above this difference
        self.chg_min = _env("SCENE_CHG", 0.13)      # changed share of the picture that means "something new"
        self.mot_max = _env("SCENE_STILL", 0.02)    # moving share below this = held still
        self.arm = _env("SCENE_ARM", 0.035)         # a real movement must have happened first
        self.hold_s = _env("SCENE_HOLD_S", 1.2)
        self.alpha = _env("SCENE_ALPHA", 0.04)
        self.bg = None
        self.prev = None
        self.armed = False
        self.still_since = None
        self.chg = self.mot = 0.0

    def reset(self):
        self.bg = self.prev = None
        self.armed = False
        self.still_since = None

    def update(self, jpeg: bytes, t: float) -> bool:
        """Feed one frame. -> True while something new is being held still. It keeps returning True until the caller
        takes it with absorb(): a remark that had to wait (Olingo was talking) is made as soon as the floor is free."""
        try:
            g = gray(jpeg)
        except Exception:
            return False
        if self.bg is None:
            self.bg, self.prev = g, g
            return False
        self.mot = float((np.abs(g - self.prev) > self.pix).mean())
        self.chg = float((np.abs(g - self.bg) > self.pix).mean())
        self.prev = g
        if self.mot > self.arm:
            self.armed = True
        if self.mot >= self.mot_max:
            self.still_since = None
            return False
        if self.still_since is None:
            self.still_since = t
        still = t - self.still_since
        if self.chg > self.chg_min:
            return self.armed and still >= self.hold_s
        # quiet and close to the background: follow slow drift (light, posture)
        self.bg = (1 - self.alpha) * self.bg + self.alpha * g
        if still > 2.0:
            self.armed = False
        return False

    def still_for(self, t: float) -> float:
        """Seconds the picture has been still."""
        return (t - self.still_since) if self.still_since is not None else 0.0

    def absorb(self, g=None):
        """Make the current picture the background (after firing, or when the model saw nothing new)."""
        self.bg = self.prev if g is None else g
        self.armed = False
        self.still_since = None
