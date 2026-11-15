"""Bonsai (llama-server) client: SEE = the one vision call per object, TEACH = background text call, plus mocks."""
import base64
import math
import http.client
import io
import json
import re
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from PIL import Image

from . import prompts, spanish as sp, vocab


def crop_aperture(img: Image.Image, out: int = 384, cx: float = 0.22, side_frac: float = 0.55) -> Image.Image:
    """Full webcam frame -> aperture crop (square, centred at (cx*W, 0.5*H), side = side_frac*H). Square input = already cropped."""
    w, h = img.size
    if w == h:
        return img.convert("RGB").resize((out, out), Image.BILINEAR)
    s = int(side_frac * h)
    x0 = min(max(int(cx * w - s / 2), 0), w - s)
    y0 = max(int(h / 2 - s / 2), 0)
    return img.convert("RGB").crop((x0, y0, x0 + s, y0 + s)).resize((out, out), Image.BILINEAR)


def to_jpeg(img: Image.Image, quality: int = 80) -> bytes:
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=quality)
    return b.getvalue()


def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _info(out: dict, t0: float) -> dict:
    tm = out.get("timings") or {}
    return {"ms": round((time.time() - t0) * 1000), "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
            "prompt_ms": round(tm.get("prompt_ms", 0)), "predicted_n": tm.get("predicted_n"),
            "predicted_ms": round(tm.get("predicted_ms", 0)),
            "tok_s": round(tm.get("predicted_per_second", 0), 1)}


def validate_see(d: dict):
    """-> cleaned dict or None."""
    try:
        en, es = str(d["en"]).strip().lower(), str(d["es"]).strip().lower()
        g = d["g"] if d["g"] in ("m", "f") else "m"
        n = int(d["n"]) if 1 <= int(d["n"]) <= 5 else 1
        color = str(d["color"]).strip().lower()
    except (KeyError, ValueError, TypeError):
        return None
    color = sp.COLOR_EN_ALIASES.get(color, color)
    if color not in prompts.COLORS:
        color = "none"
    return {"en": en, "es": es, "g": g, "n": n, "color": color}


class BonsaiVision:
    # structured-output variants, tried in order until one yields valid output (see prompts.py note)
    MODES = ["grammar", "rf_openai", "top_schema", "json_object"]

    def __init__(self, url="http://127.0.0.1:8080", see_slot=0, teach_slot=1):
        self.url = url.rstrip("/")
        u = urlparse(self.url)
        self.host, self.port = u.hostname, u.port or 80
        self.see_slot, self.teach_slot = see_slot, teach_slot
        self.mode = None
        self._teach_conn = None
        self._teach_cancel = threading.Event()

    # ----- helpers -----
    def _rf(self, mode, kind):
        schema, gbnf = (prompts.SEE_SCHEMA, prompts.SEE_GBNF) if kind == "see" else (prompts.TEACH_SCHEMA, prompts.TEACH_GBNF)
        if mode == "grammar":
            return {"grammar": gbnf}
        if mode == "rf_openai":
            return {"response_format": {"type": "json_schema", "json_schema": {"name": "r", "strict": True, "schema": schema}}}
        if mode == "top_schema":
            return {"json_schema": schema}
        return {"response_format": {"type": "json_object"}}

    def _post(self, body, timeout=60):
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    # ----- SEE (lean: en, n, colour + confidence of the object name) -----
    @staticmethod
    def name_confidence(out: dict) -> float:
        """Mean token probability of the `en` value (0..1), robust to how the tokenizer splits the JSON."""
        try:
            toks = out["choices"][0]["logprobs"]["content"]
        except (KeyError, TypeError, IndexError):
            return -1.0
        text, spans = "", []
        for t in toks:
            spans.append((len(text), len(text) + len(t["token"]), t["logprob"]))
            text += t["token"]
        m = re.search(r'"en"\s*:\s*"([^"]*)"', text)
        if not m:
            return -1.0
        a, b = m.span(1)
        lps = [lp for (s0, s1, lp) in spans if s0 < b and s1 > a]
        return round(math.exp(sum(lps) / len(lps)), 3) if lps else -1.0

    def see_lean(self, jpeg: bytes):
        """One vision call -> ({en, n, color} | None, info incl. conf). ~1 s on Bonsai 2B, ~1.9 s on 4B."""
        body = {**prompts.COMMON_PARAMS, "id_slot": self.see_slot, "max_tokens": 40, "grammar": prompts.LEAN_GBNF,
                "logprobs": True, "top_logprobs": 1,
                "messages": [{"role": "system", "content": prompts.LEAN_SYSTEM},
                             {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(jpeg)}}]}]}
        t0 = time.time()
        try:
            out = self._post(body)
            d = json.loads(re.search(r"\{.*\}", out["choices"][0]["message"]["content"], re.S).group(0))
            color = sp.COLOR_EN_ALIASES.get(str(d["color"]).lower(), str(d["color"]).lower())
            clean = {"en": str(d["en"]).strip().lower(), "n": int(d["n"]), "color": color if color in prompts.COLORS else "none"}
        except (urllib.error.URLError, ValueError, KeyError, AttributeError, TypeError) as e:
            return None, {"error": str(e), "ms": round((time.time() - t0) * 1000)}
        info = _info(out, t0)
        info["conf"] = self.name_confidence(out)
        return clean, info

    def spanish_for(self, en: str):
        """Only for objects missing from vocab.py: (es, g) from a tiny text call."""
        body = {**prompts.COMMON_PARAMS, "id_slot": self.teach_slot, "max_tokens": 24, "grammar": prompts.ES_GBNF,
                "messages": [{"role": "system", "content": prompts.ES_SYSTEM}, {"role": "user", "content": en}]}
        try:
            d = json.loads(self._post(body)["choices"][0]["message"]["content"])
            return d["es"].strip().lower(), d["g"]
        except (urllib.error.URLError, ValueError, KeyError):
            return None, None

    # ----- SEE (full: spec 5.3 original) -----
    def see(self, jpeg: bytes, max_tokens: int = 48):
        """One vision call. -> (clean dict | None, info dict). Falls back through response_format modes once."""
        messages = [{"role": "system", "content": prompts.SEE_SYSTEM},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(jpeg)}}]}]
        modes = [self.mode] if self.mode else self.MODES
        last_err = None
        for mode in modes:
            body = {**prompts.COMMON_PARAMS, "messages": messages, "max_tokens": max_tokens, "id_slot": self.see_slot,
                    **self._rf(mode, "see")}
            t0 = time.time()
            try:
                out = self._post(body)
                raw = out["choices"][0]["message"].get("content") or ""
                m = re.search(r"\{.*\}", raw, re.S)
                clean = validate_see(json.loads(m.group(0))) if m else None
                if clean is None:
                    last_err = f"mode {mode}: bad output {raw[:80]!r}"
                    continue
                self.mode = mode
                info = _info(out, t0)
                info.update(raw=raw, mode=mode)
                return clean, info
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as e:
                last_err = f"mode {mode}: {e}"
        return None, {"error": last_err, "ms": 0}

    # ----- TEACH (streamed, cancellable) -----
    def cancel_teach(self):
        self._teach_cancel.set()
        c = self._teach_conn
        if c is not None:
            try:
                c.sock.shutdown(2)
            except Exception:
                pass
            try:
                c.close()
            except Exception:
                pass

    @staticmethod
    def _partials(buf: str) -> dict:
        d = {}
        for key in ("s", "s_en"):
            m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % key, buf)
            if m:
                d[key] = json.loads('"%s"' % m.group(1))
        rel = re.findall(r'\{\s*"es"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"en"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', buf)
        if rel:
            d["rel"] = [{"es": json.loads('"%s"' % a), "en": json.loads('"%s"' % b)} for a, b in rel]
        return d

    def teach(self, es: str, g: str, color: str, on_partial=None, max_tokens: int = 96):
        """-> (dict | None, info). Streams; on_partial(dict) as fields complete. Returns None if cancelled/invalid."""
        self._teach_cancel.clear()
        user = f"{sp.definite(g, 1)} {es}"
        mode = self.mode or "grammar"
        body = {**prompts.COMMON_PARAMS, "messages": [{"role": "system", "content": prompts.TEACH_SYSTEM},
                                                       {"role": "user", "content": user}],
                "max_tokens": max_tokens, "id_slot": self.teach_slot, "stream": True,
                **self._rf(mode, "teach")}
        t0 = time.time()
        buf, tm, seen = "", {}, {}
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=60)
            self._teach_conn = conn
            conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                return None, {"error": f"HTTP {resp.status}", "ms": 0}
            for raw in resp:
                if self._teach_cancel.is_set():
                    return None, {"cancelled": True, "ms": round((time.time() - t0) * 1000)}
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                tm = ev.get("timings") or tm
                for ch in ev.get("choices", []):
                    buf += ch.get("delta", {}).get("content") or ""
                part = self._partials(buf)
                if part != seen and on_partial:
                    seen = part
                    on_partial(part)
        except (OSError, http.client.HTTPException, ValueError):
            if self._teach_cancel.is_set():
                return None, {"cancelled": True, "ms": round((time.time() - t0) * 1000)}
            return None, {"error": "connection", "ms": round((time.time() - t0) * 1000)}
        finally:
            self._teach_conn = None
        info = {"ms": round((time.time() - t0) * 1000), "tok_s": round(tm.get("predicted_per_second", 0), 1),
                "predicted_n": tm.get("predicted_n"), "prompt_ms": round(tm.get("prompt_ms", 0))}
        try:
            d = json.loads(re.search(r"\{.*\}", buf, re.S).group(0))
            return validate_teach(d, es), info
        except (AttributeError, ValueError):
            return None, {**info, "error": "bad json"}

    # ----- warm-up (during the ringing animation) -----
    def warmup(self):
        gray = Image.new("RGB", (384, 384), (128, 128, 128))
        t0 = time.time()
        self.see(to_jpeg(gray))
        self.teach("taza", "f", "azul", max_tokens=1)
        return round((time.time() - t0) * 1000)


def validate_teach(d: dict, noun: str):
    """Keep only a usable example sentence: contains the noun, at most 9 words."""
    s = str(d.get("s", "")).strip()
    key = sp.normalize(noun.split(" ")[0])[:5]
    if not s or len(s.split()) > 9 or key not in sp.normalize(s):
        d = {**d, "s": "", "s_en": ""}
    seen, rel = {sp.normalize(noun), sp.normalize(sp.definite("m", 1) + " " + noun), sp.normalize(sp.definite("f", 1) + " " + noun)}, []
    for r in d.get("rel", []):
        k = sp.normalize(str(r.get("es", ""))) if isinstance(r, dict) else ""
        if k and r.get("en") and k not in seen:
            seen.add(k)
            rel.append(r)
    rel = rel[:2]
    return {"s": d.get("s", ""), "s_en": d.get("s_en", ""), "rel": rel}


class MockVision:
    """Cycles through a fixed list of 'seen' objects; no model."""
    THINGS = [("cup", "azul"), ("keys", "negro"), ("phone", "negro"), ("pen", "rojo"), ("book", "verde")]
    COLORS = {"azul": "blue", "negro": "black", "rojo": "red", "verde": "green"}

    def __init__(self):
        self.i = 0
        self.mode = "mock"

    def see(self, jpeg=None, max_tokens=48):
        en, c = self.THINGS[self.i % len(self.THINGS)]
        self.i += 1
        v = vocab.lookup(en)
        n = 3 if en == "keys" else 1
        time.sleep(0.3)
        return {"en": v["en"], "es": v["es"], "g": v["g"], "n": n, "color": self.COLORS[c]}, {"ms": 300, "mock": True}

    def teach(self, es, g, color, on_partial=None, max_tokens=96):
        time.sleep(0.4)
        d = {"s": f"Me gusta {es}.", "s_en": f"I like {es}.", "rel": [{"es": "grande", "en": "big"}, {"es": "pequeño", "en": "small"}]}
        if on_partial:
            on_partial(d)
        return d, {"ms": 400, "mock": True}

    def cancel_teach(self):
        pass

    def warmup(self):
        return 0


def resolve(see: dict, translator=None) -> dict:
    """Lean SEE result -> adds es, g, pl, only (vocab first, then the model's own translation, else None)."""
    v = vocab.lookup(see["en"])
    if v:
        return {**see, "es": v["es"], "g": v["g"], "pl": v["pl"], "only": v["only"], "in_vocab": True}
    es, g = translator(see["en"]) if translator else (None, None)
    return {**see, "es": es, "g": g or "m", "pl": None, "only": False, "in_vocab": False}


class TieredVision:
    """Cascade: the smallest Bonsai that is sure answers; unsure or unknown objects go one tier up.
    tiers: [(name, BonsaiVision), ...] smallest first. Deep (not time-critical) text jobs use the biggest tier."""
    def __init__(self, tiers, conf_min: float = 0.45, unknown_conf: float = 0.7):
        # conf_min: below this the name is a guess -> ask the next tier.
        # unknown_conf: an object missing from vocab.py is also escalated unless the tier is at least this sure.
        self.tiers, self.conf_min, self.unknown_conf = tiers, conf_min, unknown_conf
        self.mode = "tiered"

    def see(self, jpeg: bytes):
        """-> (resolved dict | None, info). info: tier used, per-tier ms/conf, escalated reason."""
        trail, result = [], None
        for i, (name, v) in enumerate(self.tiers):
            d, info = v.see_lean(jpeg)
            r = resolve(d) if d else None
            trail.append({"tier": name, "ms": info.get("ms"), "conf": info.get("conf"), "en": d and d["en"]})
            if d is None:
                reason = "error"
            elif d["en"] == "none" and info.get("conf", 1) >= self.conf_min:
                result = (r, name); break                       # confidently nothing held (empty ring): do not escalate
            elif d["en"] == "none":
                reason = "unsure whether anything is held"
            elif info.get("conf", 1) < self.conf_min:
                reason = "low confidence"
            elif not r["in_vocab"] and info.get("conf", 1) < self.unknown_conf:
                reason = "not in vocabulary"
            else:
                result = (r, name); break
            if i == len(self.tiers) - 1:
                result = (r, name)                              # last tier: take what it says
            else:
                trail[-1]["escalated"] = reason
            if result:
                break
        if result is None or result[0] is None:
            return None, {"error": "all tiers failed", "trail": trail}
        r, name = result
        if r["en"] == "none":
            return r, {"tier": name, "trail": trail, "ms": sum(t["ms"] or 0 for t in trail), "conf": trail[-1]["conf"]}
        if not r["in_vocab"]:
            r["es"], r["g"] = self.deep().spanish_for(r["en"])      # rare and not time-critical: biggest tier translates
            if not r["es"]:
                return None, {"error": "no Spanish for " + r["en"], "trail": trail}
        return r, {"tier": name, "trail": trail, "ms": sum(t["ms"] or 0 for t in trail), "conf": trail[-1]["conf"]}

    def deep(self):
        return self.tiers[-1][1]

    def warmup(self):
        return sum(v.warmup() for _, v in self.tiers)
