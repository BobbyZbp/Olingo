"""Aperture gate: decides WHEN to look (a held object is stable) so the VLM is called once per object.
Input: JPEG crops of the aperture (browser already cropped; the full face never reaches the backend).
Cheap numpy only: 96x96 grayscale, mean-subtracted, compared with an 'empty aperture' reference and the last frame."""
import io
import os
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

SIDE = 96


def _env(name, default):
    return float(os.environ.get(name, default))


@dataclass
class GateConfig:
    pix: float = field(default_factory=lambda: _env("GATE_PIX", 0.06))          # |delta| per pixel that counts as changed
    empty: float = field(default_factory=lambda: _env("GATE_EMPTY", 0.05))      # occ below this = nothing in the ring
    enter: float = field(default_factory=lambda: _env("GATE_ENTER", 0.12))      # occ above this = something entered
    mot: float = field(default_factory=lambda: _env("GATE_MOT", 0.015))        # mot below this = still
    spike: float = field(default_factory=lambda: _env("GATE_SPIKE", 0.08))      # mot above this = object swapped / moved
    hold_s: float = field(default_factory=lambda: _env("GATE_HOLD_S", 0.3))
    ref_alpha: float = field(default_factory=lambda: _env("GATE_REF_ALPHA", 0.05))


def decode_gray(jpeg: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(jpeg)).convert("L").resize((SIDE, SIDE), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a - a.mean()


def sharpness(g: np.ndarray) -> float:
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


class Gate:
    EMPTY, ENTERING, HOLDING, DONE = "empty", "entering", "holding", "done"

    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self.ref = None
        self.prev = None
        self.state = self.EMPTY
        self.hold_start = None
        self.cands = []            # (sharpness, jpeg) collected while the object is still
        self.moved_since_fire = False
        self.occ = 0.0
        self.mot = 0.0

    def reset_reference(self):
        self.ref = None
        self.state = self.EMPTY
        self.hold_start = None
        self.cands = []

    def ring(self) -> str:
        return {self.EMPTY: "empty", self.ENTERING: "entering", self.HOLDING: "holding", self.DONE: "holding"}[self.state]

    def update(self, jpeg: bytes, t: float | None = None):
        """-> (trigger_jpeg or None). Metrics in self.occ / self.mot, ring state via self.ring()."""
        t = time.monotonic() if t is None else t
        c = self.cfg
        g = decode_gray(jpeg)
        if self.ref is None:
            self.ref, self.prev = g.copy(), g
            return None
        self.occ = float((np.abs(g - self.ref) > c.pix).mean())
        self.mot = float((np.abs(g - self.prev) > c.pix).mean())
        self.prev = g
        fire = None

        if self.state == self.EMPTY:
            if self.occ < c.empty:
                self.ref = (1 - c.ref_alpha) * self.ref + c.ref_alpha * g       # follow slow lighting changes
            elif self.occ > c.enter:
                self.state = self.ENTERING
                self.hold_start, self.cands = None, []
        elif self.state in (self.ENTERING, self.HOLDING):
            if self.occ < c.empty:
                self.state, self.hold_start, self.cands = self.EMPTY, None, []
            elif self.mot < c.mot:
                if self.hold_start is None:
                    self.hold_start, self.cands = t, []
                self.cands.append((sharpness(g), jpeg))
                self.state = self.HOLDING
                if t - self.hold_start >= c.hold_s and len(self.cands) >= 2:
                    fire = max(self.cands, key=lambda x: x[0])[1]
                    self.state, self.moved_since_fire = self.DONE, False
                    self.cands = []
            else:
                self.state, self.hold_start, self.cands = self.ENTERING, None, []
        elif self.state == self.DONE:
            # re-arm: object removed, or a clear movement followed by stillness (a different object)
            if self.occ < c.empty:
                self.state = self.EMPTY
            elif self.mot > c.spike:
                self.moved_since_fire = True
            elif self.moved_since_fire and self.mot < c.mot:
                self.state, self.hold_start, self.cands = self.ENTERING, None, []
        return fire
