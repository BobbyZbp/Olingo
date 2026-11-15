"""What Olingo remembers within one call.

- The last 30 seconds are kept word for word; that is the conversation Bonsai gets.
- Everything older is compressed. Each turn becomes a sketch (256 numbers: its content words and character pairs,
  hashed; no model involved) and a short gist: of what the learner said, their first ten words; of what Olingo
  said, only the content words.
- Every question Olingo has asked stays for the whole call, with its sketch. Before a new question is asked it is
  compared with all of them, so the same thing is not asked twice (`closest`, and `brain.same_question` for
  questions that mean the same in other words).
Nothing here is written to disk."""
import re
import time
import zlib

import numpy as np

DIM = 256
STOP = set("""a an the and or but so of to in on at for with from by as is are am was were be been being do does did
have has had i you he she it we they me my your his her its our their this that these those there here what which who
whom whose when where why how can could will would shall should may might must not no yes very really just also too
than then about into over some any more most much many please tell let us one thing something anything
le la les un une des et ou mais de du au aux en dans sur pour avec par est sont suis es ai as a ont je tu il elle nous
vous ils elles mon ma mes ton ta tes son sa ses ce cette ces que qui quoi comment pourquoi quand ou tres aussi ne pas
""".split())
CJK_STOP = set("的了吗呢吧啊呀哦嗯你我他她它们是在有和也就都很这那个一不么什怎样想要会能可以请把给对跟让被还再又才只"
               "はがをにのでともへやかねよですますしたいるあるこのそのどれなに")


def _stem(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")) and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    for end in ("ing", "ed"):
        if w.endswith(end) and len(w) - len(end) >= 3:
            return w[:-len(end)]
    return w


def _is_cjk(c: str) -> bool:
    return "぀" <= c <= "ヿ" or "一" <= c <= "鿿" or "가" <= c <= "힯"


def content(text: str) -> list:
    """The words that carry the meaning, in order: stop words out, endings off; for Chinese and Japanese, characters."""
    out = []
    for w in re.findall(r"[a-zà-ÿ]+(?:'[a-zà-ÿ]+)?", text.lower()):
        w = w.split("'")[0]
        if len(w) >= 2 and w not in STOP:
            out.append(_stem(w))
    out += [c for c in text if _is_cjk(c) and c not in CJK_STOP]
    return out


def sketch(text: str) -> np.ndarray:
    """A fixed-size fingerprint of what a sentence is about. Two sentences about the same things lie close together."""
    v = np.zeros(DIM, dtype=np.float32)
    toks = content(text)
    feats = [(t, 1.0) for t in toks]
    cjk = [t for t in toks if _is_cjk(t[0])]
    feats += [(a + b, 1.0) for a, b in zip(cjk, cjk[1:])]                  # character pairs: closer to words
    for t in toks:
        if not _is_cjk(t[0]) and len(t) >= 4:                              # pieces of longer words: day ~ today
            feats += [(t[i:i + 3], 0.3) for i in range(len(t) - 2)]
    for f, w in feats:
        h = zlib.crc32(f.encode())
        v[h % DIM] += w if (h >> 16) & 1 else -w
    n = float(np.linalg.norm(v))
    return v / n if n else v


def gist(text: str, words: int = 6, chars: int = 14) -> str:
    """A sentence boiled down to a few words: what stays of a turn once it is older than the window."""
    cjk = [c for c in text if _is_cjk(c)]
    if len(cjk) > len(text) / 3:
        return re.sub(r"[\s，。！？、,.!?]+", "", text)[:chars]
    seen, out = set(), []
    for w in re.findall(r"[a-zà-ÿ]+(?:'[a-zà-ÿ]+)?", text.lower()):
        if len(w) >= 2 and w not in STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return " ".join(out[:words])


class Memory:
    def __init__(self, window_s: float = 30.0):
        self.window_s = window_s
        self.live = []           # (time, role, text): the last half minute, word for word
        self.old = []            # (role, gist, sketch): everything before that
        self.questions = []      # (text, sketch): every question Olingo asked in this call

    def add(self, role: str, text: str, t: float | None = None):
        text = (text or "").strip()
        if text:
            self.live.append((time.time() if t is None else t, role, text))

    def roll(self, now: float | None = None):
        """Move what is older than the window into the compressed part. The last exchange always stays, so that an
        answer given after a long pause still meets the question it belongs to."""
        now = time.time() if now is None else now
        while len(self.live) > 2 and now - self.live[0][0] > self.window_s:
            _, role, text = self.live.pop(0)
            # what the learner told it is the part worth keeping: their own first words; its own lines: content words
            g = " ".join(text.split()[:10])[:70].rstrip(" .,!?。，！？") if role == "user" else gist(text)
            if g:
                self.old.append((role, g, sketch(text)))
        self.old = self.old[-200:]

    def window(self, now: float | None = None) -> list:
        """The conversation for Bonsai: the last 30 seconds, starting with something the learner said."""
        self.roll(now)
        msgs = []
        for _, role, text in self.live:
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += " " + text
            else:
                msgs.append({"role": role, "content": text})
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        return msgs

    def summary(self, n: int = 10) -> dict:
        """What is left of the older turns: the gists, newest last, without repeats."""
        def last(role):
            return list(dict.fromkeys(g for r, g, _ in self.old if r == role))[-n:]
        return {"told": last("user"), "said": last("assistant")}

    # ----- questions -----
    def asked(self, q: str):
        q = (q or "").strip()
        if q:
            self.questions.append((q, sketch(q)))

    def closest(self, q: str) -> tuple:
        """(similarity 0..1, the earlier question) most like this one."""
        if not self.questions:
            return 0.0, ""
        v = sketch(q)
        sims = [float(v @ s) for _, s in self.questions]
        i = int(np.argmax(sims))
        return max(sims[i], 0.0), self.questions[i][0]

    def nearest(self, q: str, n: int = 6) -> list:
        """The earlier questions worth showing to a judge: the most similar ones and the latest ones."""
        if not self.questions:
            return []
        v = sketch(q)
        order = sorted(range(len(self.questions)), key=lambda i: -float(v @ self.questions[i][1]))[:n]
        pick = sorted(set(order) | set(range(max(0, len(self.questions) - 3), len(self.questions))))
        return [self.questions[i][0] for i in pick]

    def stats(self) -> dict:
        return {"live": len(self.live), "compressed": len(self.old), "questions": len(self.questions),
                "bytes": len(self.old) * DIM * 4 + len(self.questions) * DIM * 4}
