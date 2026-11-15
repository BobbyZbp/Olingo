"""One call, start to finish. This loop is the only conductor: it talks to the speech bridge (Argmax) and to Bonsai
and reports every step, with its measured time, through `emit`.

Routing is decided by code, never by a model:
  tier 0 = no model   commands, the opening question, yes/no, repeat scoring, word swap, pinyin, cached audio
  tier 1 = Bonsai 4B  small talk, text only
  tier 2 = Bonsai 9B  with one small frame: anything about what it sees, teaching a new word; also when unsure

Speaking never blocks the loop: sentences are queued on the bridge, which plays them back to back, so the first
sentence of an answer is audible while Bonsai is still writing the rest.
Frames and audio only ever live in memory. Each turn carries the last 30 seconds of talk word for word; what is
older is carried as a compressed memory (backend/memory.py)."""
import difflib
import math
import os
import random
import re
import threading
import time
import unicodedata

from . import brain
from . import lang as L
from . import memory

try:
    from pypinyin import Style, lazy_pinyin
    import zhconv
except ImportError:                                  # other targets work without them
    lazy_pinyin = zhconv = None


# ---------------------------------------------------------------- text helpers (no model anywhere below)
def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w\s']", " ", s).strip()


def is_cjk(c: str) -> bool:
    return "一" <= c <= "鿿" or "぀" <= c <= "ヿ" or "가" <= c <= "힯"


def has_cjk(s: str) -> bool:
    return any(is_cjk(c) for c in s)


def readings(text: str, l2: str) -> list:
    """Pinyin for each character of a Chinese text ("" for anything that is not a Chinese character)."""
    if l2 != "zh" or not lazy_pinyin:
        return []
    return lazy_pinyin(text, style=Style.TONE, errors=lambda s: ["" for _ in s])


def reading(word: str, l2: str) -> str:
    return " ".join(r for r in readings(word, l2) if r)


def _toneless(ch: str) -> str:
    return lazy_pinyin(ch)[0] if lazy_pinyin else ch


def score_cjk(target: str, heard: dict, l2: str) -> dict:
    """Clarity of each character of `target` in what Whisper heard. Same character = its probability;
    same syllable written with another character (a homophone) = 0.8 of it; missing = 0."""
    conv = (lambda s: zhconv.convert(s, "zh-cn")) if (l2 == "zh" and zhconv) else (lambda s: s)
    text = conv(heard.get("text", ""))
    chars, probs = [], []
    for w in heard.get("words", []):
        for c in conv(w["w"]):
            if is_cjk(c):
                chars.append(c)
                probs.append(float(w.get("p", 0.5)))
    if not chars:
        chars = [c for c in text if is_cjk(c)]
        probs = [0.7] * len(chars)
    out, used = [], set()
    for c in [c for c in target if is_cjk(c)]:
        best, bi = 0.0, None
        for i, h in enumerate(chars):
            if i in used:
                continue
            # the same syllable written with another character is the same sound: Whisper picks a character, the
            # learner only made a sound
            s = probs[i] if h == c else (0.9 * probs[i] if l2 == "zh" and _toneless(h) == _toneless(c) else 0.0)
            if s > best:
                best, bi = s, i
        if bi is not None:
            used.add(bi)
        out.append({"w": c, "p": round(best, 2)})
    avg = sum(o["p"] for o in out) / len(out) if out else 0.0
    return {"chars": out, "score": round(avg, 2), "heard": text}


def _close(a: str, b: str) -> bool:
    """Equal, or one letter apart for words of four letters or more."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def score_words(target: str, heard: dict) -> dict:
    """Clarity of each word of `target` (languages written with spaces)."""
    hw = [(norm(w["w"]), float(w.get("p", 0.5))) for w in heard.get("words", []) if norm(w["w"])]
    if not hw:
        hw = [(t, 0.7) for t in norm(heard.get("text", "")).split()]
    out, used = [], set()
    for tok in target.split():
        n, best, bi = norm(tok), 0.0, None
        for i, (w, p) in enumerate(hw):
            if i not in used and _close(n, w) and p > best:
                best, bi = p, i
        if bi is not None:
            used.add(bi)
        out.append({"w": tok, "p": round(best, 2)})
    avg = sum(o["p"] for o in out) / len(out) if out else 0.0
    return {"chars": out, "score": round(avg, 2), "heard": heard.get("text", "")}


def score(target: str, heard: dict, l2: str) -> dict:
    return score_cjk(target, heard, l2) if L.LANGS.get(l2, {}).get("cjk") else score_words(target, heard)


RULES = {  # spoken commands. First pattern: on the normalised transcript (English, French, accents removed).
           # Second pattern: on the raw transcript (Chinese, Japanese).
    "bye": (r"\b(bye|goodbye|good bye|hang up|see you|au revoir|a bientot|a plus tard)\b",
            r"再见|拜拜|挂了|挂断|さようなら|バイバイ|またね|じゃあね"),
    "again": (r"^(please )?(can you )?(say (it|that) )?(again|repeat( that| it)?|one more time|come again|pardon|"
              r"what did you say)( please)?$|^(tu peux |vous pouvez )?(repete[rz]?|redis|encore une fois)( s'il (te|vous) plait)?$",
              r"再说一遍|再来一次|再说一次|没听清|もう一度|もういちど|もう一回"),
    "slower": (r"\b(slower|slowly|too fast|slow down|plus lentement|moins vite|trop vite|doucement)\b",
               r"慢一点|慢点|太快|ゆっくり|速すぎ|早すぎ"),
    "more": (r"\bmore (chinese|spanish|french|japanese|korean|german|italian|portuguese|english)\b|\bharder\b|"
             r"\bplus de (chinois|espagnol|francais|japonais|coreen|allemand|italien|portugais|anglais)\b|\bplus difficile\b",
             r"多说|难一点|もっと(中国語|英語|スペイン語|フランス語|韓国語)|難しくして"),
    "less": (r"\bless (chinese|spanish|french|japanese|korean|german|italian|portuguese|english)\b|"
             r"\b(english please|too hard|easier|trop difficile|plus facile)\b|"
             r"\bmoins de (chinois|espagnol|francais|japonais|coreen|allemand|italien|portugais|anglais)\b",
             r"少说|太难|简单一点|難しすぎ|簡単にして|やさしくして"),
    # the outside world: Olingo is offline and says so, by rule, instead of letting a model guess
    "offline": (r"\b(weather|forecast|temperature outside|what time is it|what day is it|what is the date|today's date|"
                r"the news|quel temps|la meteo|quelle heure|quel jour|les nouvelles)\b",
                r"天气|几点|今天几号|星期几|新闻|天気|何時|何日|何曜日|ニュース"),
    "pick": (r"\b(you pick|you choose|your pick|your choice|pick (one|something)|show me around|look around|"
             r"tu choisis|a toi de choisir|choisis pour moi)\b",
             r"你来选|你选吧|你来挑|看看周围|選んで|えらんで|おまかせ"),
}
_LEAD = r"^(?:(?:hey|oh|um|uh|well|hmm|ah|so|euh|ben|bah|alors)\s+)*"       # what people say before yes or no
YES = (_LEAD + r"(yes|yeah|yep|yup|sure|ok|okay|please|of course|i do|i want to|let's do it|absolutely|definitely|"
               r"teach me|go ahead|oui|ouais|d'accord|bien sur|volontiers|vas y|s'il te plait)\b",
       r"^(好|要|想|可以|是的|对|嗯|行|はい|うん|ええ|お願い|おねがい|いいよ|いいです|オーケー|ぜひ)")
NO = (_LEAD + r"(no|nope|nah|not now|not really|later|maybe later|something else|non|pas maintenant|plus tard|"
              r"autre chose)\b", r"^(不|不用|不要|算了|别的|いいえ|いや|結構|けっこう|あとで|別の|ほかの)")


def intent(text: str, offer: bool = False) -> str:
    """-> bye | again | slower | more | less | pick | yes | no | ask. Rules only."""
    t = norm(text)
    for name, (en, zh) in RULES.items():
        if re.search(en, t) or re.search(zh, text):
            return name
    if offer:
        if re.search(YES[0], t) or re.search(YES[1], text.strip()):
            return "yes"
        if re.search(NO[0], t) or re.search(NO[1], text.strip()):
            return "no"
    return "ask"


# The learner is talking about the picture: only then does Bonsai get a frame. A bare "it" or "that" is not enough
# ("I like it", "that is true"): those sentences are conversation, and looking made Olingo describe the picture
# instead of answering.
VISUAL = re.compile(
    r"\b(see|look|looking|watch|holding|hold|show|showing|camera|picked|pick up|in my hand|behind me|in front of|"
    r"what is (this|that|it)|what's (this|that|it)|whats (this|that|it)|what are (these|those)|what am i|what do i have|"
    r"how many|what colou?r|which colou?r|am i wearing|what i'm wearing|what i am wearing|do you like my|"
    r"how do you say (this|that|it)|(this|that|these|those) (one|thing|things)|is (this|that) an?|"
    r"(this|here) is (my|a|an|the)|(these|here) are (my|the)|i have (a|an|my) .* here|"
    r"regarde|vois|voir|tiens|qu'est ce que c'est|c'est quoi|combien|dans ma main|derriere moi|quelle couleur|"
    r"je porte|ca s'appelle|voici)\b"
    r"|这是|那是|这个|那个|这些|那些|你看|看看|看到|看见|拿着|拿的|手里|手上|什么东西|几个|什么颜色|穿的|穿着|戴的|戴着"
    r"|これ|それ|あれ|見て|見え|持って|着て|何色|いくつ")
CHAT = re.compile(
    r"\b(i (ate|had|have|feel|am|was|like|love|want|went|think|need)|i'm|im|my (day|name|job|family|friend|favorite|"
    r"favourite)|weather|today|yesterday|tomorrow|how are you|your name|who are you|thank|thanks|hello|hi|"
    r"how do you say (?!this|that|it)|tell me|lunch|dinner|breakfast|tired|hungry|happy|sad|"
    r"je (suis|me sens|veux|pense|mange|aime)|j'ai|meteo|aujourd'hui|hier|demain|comment ca va|merci|bonjour|salut|"
    r"comment dit on (?!ca|cela)|dejeuner|diner|fatigue|faim)\b"
    r"|我(今天|昨天|明天|觉得|喜欢|想|吃|叫|是)|天气|你好|名字|谢谢|私は|今日|昨日|明日|天気|こんにちは|ありがとう|名前|元気")
ASKS_HOW = re.compile(r"how do you say|how to say|what is .* in |in (chinese|spanish|french|japanese|korean|german)|"
                      r"comment dit on|怎么说|何と言|なんて言")


# "What is this?" and its cousins: naming one thing. Bonsai 4B does this as well as the 9B and sooner (measured on
# the same frames: 17 of 18 right against 15 of 18, first sentence after 0.55 s against 0.91 s).
SIMPLE_LOOK = re.compile(
    r"\b(what is (this|that|it)|what's (this|that|it)|whats (this|that|it)|what am i holding|what do i have|"
    r"what are (these|those)|how do you say (this|that|it)|can you see (this|it|what)|do you (know|see) what|"
    r"what did i (pick|grab))\b|这是什么|那是什么|这个是什么|我(手里|手上)?拿(的|着).{0,4}什么|这个.{0,3}怎么说")
# When the 4B does not know what it is looking at, it says so with a word like these. Then the 9B takes over.
VAGUE = {"object", "thing", "item", "something", "stuff", "device", "tool", "物体", "东西", "物品", "物件"}
ASKING = re.compile(r"[?？]\s*$|^(what|how|why|where|when|who|which|can|could|do|does|did|is|are|am|will|would|should)\b|"
                    r"吗|什么|怎么|为什么|哪|谁|est ce que|qu'est ce|pourquoi|comment|combien|quel|ですか|ますか|何|どう|なぜ")


def route(text: str) -> tuple:
    """-> (tier, why). It looks only when the learner talks about the picture; anything else is conversation,
    answered from the text alone."""
    t = norm(text)
    if VISUAL.search(t) or VISUAL.search(text):
        return 2, "about what it sees"
    if CHAT.search(t) or CHAT.search(text):
        return 1, "small talk, no image needed"
    return 1, "conversation, no need to look"


HALLUCINATED = {"you", "thank you", "thanks for watching", "thank you for watching", "the end", "so", "oh", "uh", "hmm"}


def is_noise(h: dict) -> bool:
    """A cough, a neighbour, or Whisper inventing words from silence."""
    text = h.get("text", "")
    n = norm(text)
    if not n and not has_cjk(text):
        return True
    if re.fullmatch(r"\s*[\(\[][^\)\]]{0,40}[\)\]]\s*", text):
        return True                                      # Whisper's labels for non-speech: (upbeat music), [BLANK_AUDIO]
    avg = h.get("avg_p")
    if avg is not None and avg < 0.3:
        return True
    toks = n.split()
    if len(toks) >= 8 and len(set(toks)) <= max(2, len(toks) // 6):
        return True                                      # "da da da da ...": Whisper looping on noise
    a = h.get("audio_s") or 0.0
    return a < 1.0 and n in HALLUCINATED


FRAMES = {  # phrases that are always correct, whatever the noun: (pattern in the new language, meaning pattern)
    ("en", "zh"): [("我的{w}", "my {k}"), ("这是我的{w}", "this is my {k}"), ("我喜欢{w}", "I like {k}")],
    ("zh", "en"): [("my {w}", "我的{k}"), ("this is my {w}", "这是我的{k}")],
    ("ja", "zh"): [("我的{w}", "私の{k}"), ("这是我的{w}", "これは私の{k}です")],
    ("ja", "en"): [("my {w}", "私の{k}"), ("this is my {w}", "これは私の{k}です")],
    ("en", "es"): [("mi {w}", "my {k}")],
}


def frame_phrase(key_l1: str, key: str, l1: str, l2: str, i: int) -> dict | None:
    opts = FRAMES.get((l1, l2))
    if not opts:
        return None
    a, b = opts[i % len(opts)]
    return {"l2": a.format(w=key), "l1": b.format(k=key_l1)}


# ---------------------------------------------------------------- the call
# An utterance that stops here is not finished: the learner is still thinking.
UNFINISHED = re.compile(r"(\b(and|but|so|because|or|then|like|um+|uh+|the|a|an|to|of|with|my|is|are|am|i|et|mais|parce que|donc|euh)"
                        r"|然后|就是|那个|因为|所以|但是|还有|而且|嗯+|呃+|それで|えっと|でも|から|[,，、…])[\s.。…]*$", re.I)


class Call:
    IDLE_S = 20.0            # silence on the open floor before Olingo says something on its own
    DRILL_S = 10.0           # how long the learner gets to start repeating a word
    QUIET_S = 6.0            # after Olingo stops talking, the learner gets at least this long before any remark
    PRO_GAP = 20.0           # at least this long between two unprompted remarks
    SCENE_GAP = 30.0         # ... and between two remarks about something new in the picture
    WINDOW_S = 30.0          # how much talk is kept word for word; older turns are compressed (memory.py)
    PRO_MAX = 2              # unprompted remarks in a row; then it only waits
    CUT_CHG = 0.16           # how much of the picture must change before Olingo interrupts itself
    VOICE_MIN = 0.2          # less similar than this to the learner's voice = someone else. Measured on 15 voices:
                             # at 0.2 the right voice is never dropped and 70% of other voices are; at 0.4, 97% / 87%.
                             # Dropping the learner is the worse mistake, so the cautious end is used.
    VOICE_LEARN = 0.5        # a voice this similar keeps refining what the learner sounds like
    EXPLORE_WAIT = 10.0      # after "show me around": at least this long before it looks again
    EXPLORE_STILL = 1.5      # ... and only once the picture has been still this long

    def __init__(self, speech, get_frame, emit=None, l1="en", l2="zh", level=1, voice=None, ptt=False, focus=True,
                 room=None, on_look=None, notice=False):
        self.speech, self.get_frame = speech, get_frame
        self.emit = emit or (lambda kind, **d: None)
        self.l1, self.l2, self.level = l1, l2, min(max(int(level or 1), 1), 3)
        self.voice = L.voice_for(l1, voice, l2)
        # Speaking first when the picture changes ("you picked up a banana") is off unless the learner ticks it:
        # a learner who merely moved was looked at, and nothing came of it. It felt like being scanned for no reason.
        self.notice = bool(notice)
        self.room = list(room or [])                 # things seen in the first look, while the call was ringing
        self.on_look = on_look or (lambda: None)     # Bonsai has just seen the current picture
        self.mem = memory.Memory(self.WINDOW_S)      # 30 s word for word, the rest compressed; every question asked
        self.words = {}              # L1 word -> {l2, reading, score, drilled, example, phrases}
        self.seen = []               # (time, what Olingo said it saw): the short visual memory
        self.recent = []             # the last sentences Olingo said, to catch the model repeating itself
        self.last_said = []          # Olingo's last answer, for "again" and "slower"
        self.last_q = ""             # the question Olingo is waiting on
        self.asked = []              # the questions it has asked: never the same one twice
        self.dry_look = 0.0          # when it last looked at the learner because the talk had run dry
        self.stale = 0               # turns in a row whose question had been asked before
        self.carry = None            # something the learner said in a drill that was not the word: answered next
        self.sorry_at = 0.0          # when it last said it had not caught something
        self.mentioned_at = -10      # the turn at which it last told a word in passing
        self.floor_since = 0.0       # when the microphone last opened on the open floor
        self.safe_i = 0
        self.teach_shape = ""        # how the last teaching sentence opened, so the next one opens differently
        self.name = ""               # what the learner said their name is
        # Voice recognition (Argmax SpeakerKit) is built in but off unless OLINGO_VOICE_ID=1: on recorded voices it
        # kept the learner every time and dropped 70% of other people, but live in a room it let a second speaker and
        # bystanders through (the room and microphone weigh more in the embedding than who is speaking).
        self.voice_id = (bool(getattr(speech, "ready_info", {}).get("voice_id")) and focus
                         and os.environ.get("OLINGO_VOICE_ID") == "1")
        self.voice_s = 0.0           # seconds of the learner's speech the voice profile was learned from
        self.askable = [t for t in self.room]      # things in view that have not been talked about yet
        self.declined = set()
        self.exploring = None        # when it asked to be shown around
        self._explore = False
        self.rejects = 0             # utterances in a row dropped as someone else's voice
        # a voice speaking a language that is not its own is slower: 2.4 words a second against 3.0. Nudge it.
        self.rate_l1 = float(os.environ.get("OLINGO_L1_RATE", "1.0"))        # normal speed
        self.next_at = -10           # the turn at which it last suggested looking around
        self.offer = None            # (L1 word, L2 word, time) after "Want to learn it?"
        self.topic, self.topic_turns = None, 0
        self.running = False
        self.ptt = ptt
        self.ptt_down = threading.Event()
        self.floor_open = False      # the microphone is open on the open floor (not in a drill)
        self.hearing = False
        self.speaking = 0            # sentences queued or playing
        self._scene = None           # a frame in which something new is held up, waiting for the loop
        self._cut = False            # ... and Olingo stopped itself mid-sentence for it
        self._in_scene = False
        self.last_pro = 0.0
        self.streak = 0
        self.focus = focus
        self.voice_level = None      # typical loudness of the learner's own speech (the nearest voice)
        self.vi = int(time.time())   # rotates the wording of varied lines
        self.filler_i = self.vi
        self.filler_recent, self.filler_last_turn = [], False
        self.mix = []                # (L2 units, all units) of the last utterances -> "this call is N% Chinese"
        self._phr = {}
        self.stats = {"turns": 0, "tier0": 0, "tier1": 0, "tier2": 0, "looks": 0, "model_calls": 0,
                      "prompt_tokens": 0, "gen_tokens": 0, "heard_s": 0.0, "cache_hits": 0, "live_tts": 0,
                      "spoke_first": 0, "ignored": 0, "escalated": 0}

    # ---------- subtitles ----------
    def segs(self, text: str, lang: str) -> list:
        """Split what is spoken into L1 and L2 runs; Chinese runs carry pinyin and, when known, their meaning."""
        if lang == self.l2:
            return [{"t": text, "l2": True, "r": readings(text, self.l2)}]
        if not L.LANGS.get(self.l2, {}).get("cjk") or L.LANGS.get(self.l1, {}).get("cjk"):
            return [{"t": text}]
        back = {w["l2"]: k for k, w in self.words.items()}
        out = []
        for run in re.findall(r"[一-鿿぀-ヿ가-힯]+|[^一-鿿぀-ヿ가-힯]+", text):
            if has_cjk(run):
                out.append({"t": run, "l2": True, "r": readings(run, self.l2), "g": back.get(run, "")})
            else:
                out.append({"t": run})
        return out

    def swap(self, text: str) -> tuple:
        """Words the learner has learned are spoken in the new language from then on, at every level."""
        known = sorted(((k, w) for k, w in self.words.items() if w.get("drilled") and w["score"] >= 0.6),
                       key=lambda kw: -len(kw[0]))
        if not known:
            return text, self.segs(text, self.l1)
        marks = []
        for k, w in known:
            pat = re.escape(k) if L.LANGS.get(self.l1, {}).get("cjk") else r"\b" + re.escape(k) + r"(?:es|s)?\b"

            def put(m, w=w, k=k):
                marks.append((w["l2"], k))
                return f"\x01{len(marks) - 1}\x02"
            text = re.sub(pat, put, text, flags=re.I)
        segs, spoken = [], ""
        for part in re.split(r"(\x01\d+\x02)", text):
            if not part:
                continue
            if part[0] == "\x01":
                l2w, k = marks[int(part[1:-1])]
                segs.append({"t": l2w, "l2": True, "r": readings(l2w, self.l2), "g": k})
                spoken += l2w
            else:
                segs.append({"t": part})
                spoken += part
        return spoken, segs

    def _units(self, segs) -> tuple:
        l2n = alln = 0
        for s in segs:
            t = s["t"]
            n = max(1, math.ceil(sum(is_cjk(c) for c in t) / 2)) if has_cjk(t) else len(t.split())
            alln += n
            l2n += n if s.get("l2") else 0
        return l2n, alln

    def pct(self) -> int:
        a = sum(x for x, _ in self.mix[-12:])
        b = sum(y for _, y in self.mix[-12:])
        return round(100 * a / b) if b else 0

    def tidy(self, text: str, lang: str) -> str:
        """Simplified characters only; and Chinese punctuation inside a sentence written with spaces reads badly."""
        if zhconv and (lang == "zh" or (self.l2 == "zh" and not L.LANGS.get(self.l1, {}).get("cjk"))):
            text = zhconv.convert(text, "zh-cn")         # never on Japanese text: its kanji must stay as they are
        if L.LANGS.get(lang, {}).get("cjk"):
            return text
        for a, b in (("。", "."), ("，", ","), ("！", "!"), ("？", "?"), ("：", ":")):
            text = text.replace(a, b + " ")
        return re.sub(r"\s+", " ", text).strip()

    # ---------- speaking (never blocks) ----------
    def say(self, text: str, lang: str | None = None, rate: float = 1.0, remember: bool = False, segs=None,
            role: str = "line", gloss: str = ""):
        lang = lang or self.l1
        if not self.running or not text or not text.strip():
            return None
        if self._scene is not None and not self._in_scene:
            return None                                  # something new is being shown: everything else waits
        segs = segs or self.segs(text, lang)
        if lang == self.l1 and role != "part" and any(s.get("l2") for s in segs):
            return self.say_mixed(text, lang, segs, remember, role, gloss)
        if lang == self.l1 and rate == 1.0:
            rate = self.rate_l1
        self.mix.append(self._units(segs))
        if remember:
            self.last_said.append((text, lang, segs))
        pct = self.pct()
        self.speaking += 1

        def started(h):
            self.emit("say", text=text, lang=lang, segs=segs, how=h.kind, ttfa_ms=h.ttfa_ms, role=role, gloss=gloss,
                      pct=pct)

        def done(h):
            self.speaking = max(0, self.speaking - 1)
            if h.info.get("stopped"):
                return
            self.stats["cache_hits" if h.kind == "cached" else "live_tts"] += 1
            self.emit("said", how=h.kind, ms=h.info.get("ms"), gen_ms=h.info.get("gen_ms"))
        return self.speech.speak(text, lang=lang, rate=rate, on_start=started, on_done=done,
                                 speaker=L.voice_for_text(lang, self.voice))

    def say_mixed(self, text: str, lang: str, segs: list, remember: bool, role: str, gloss: str):
        """A sentence with words of the new language in it. Said run by run, each in its own language, by the same
        voice: in one pass the synthesiser reads the foreign word as if it were the sentence's language.
        The subtitle shows the whole sentence from the first sound on."""
        self.mix.append(self._units(segs))
        if remember:
            self.last_said.append((text, lang, segs))
        pct = self.pct()
        first = None
        for seg in segs:
            piece = re.sub(r"^[\s.,!?。，！？:：;；、]+", "", seg["t"]).strip()   # a run never starts on the previous run's full stop
            if not re.search(r"\w", piece):
                continue                                 # punctuation only
            plang = self.l2 if seg.get("l2") else lang
            self.speaking += 1
            lead = first is None

            def started(h, lead=lead):
                self.emit("say", text=text, lang=lang, segs=segs, how=h.kind, ttfa_ms=h.ttfa_ms, gloss=gloss, pct=pct,
                          role=role if lead else "part")

            def done(h):
                self.speaking = max(0, self.speaking - 1)
                if not h.info.get("stopped"):
                    self.stats["cache_hits" if h.kind == "cached" else "live_tts"] += 1
                    self.emit("said", how=h.kind, ms=h.info.get("ms"), gen_ms=h.info.get("gen_ms"))
            h = self.speech.speak(piece, lang=plang, on_start=started, on_done=done,
                                  rate=self.rate_l1 if plang == self.l1 else 1.0,
                                  speaker=L.voice_for_text(plang, self.voice))
            first = first or h
        return first

    def fixed(self, key: str, remember: bool = False, thing: str = ""):
        return self.say(L.line(key, self.l1, self.l2, thing), remember=remember, role="fixed")

    def vary(self, key: str, remember: bool = False):
        self.vi += 1
        return self.say(L.varied(key, self.l1, self.vi), remember=remember, role="fixed")

    def reply(self, text: str, lang: str, gloss: str = ""):
        text = self.tidy(text, lang)
        segs = None
        if lang == self.l1:
            text, segs = self.swap(text)
        return self.say(text, lang=lang, remember=True, segs=segs, role="reply", gloss=gloss)

    def ask(self, text: str, lang: str, gloss: str = ""):
        """A question Olingo now waits on."""
        self.last_q = text
        self.note_question(text)
        self.mem.add("assistant", text)
        return self.reply(text, lang, gloss)

    # ---------- listening ----------
    def hear(self, lang: str, alt: str | None = None, wait: float = 10.0, max_s: float = 12.0, filler: bool = False,
             floor: bool = False, fast: bool = False, hint: str | None = None, voice: str | None = None) -> dict:
        if not self.running or self._scene is not None:
            return {"text": "", "cancelled": True, "t_end": time.time()}
        f = None
        if filler:
            fl = L.FILLERS.get(self.l1, L.FILLERS["en"])
            # not every turn (about one in three, never twice running), and never one of the last three again
            if not self.filler_last_turn and random.random() < 0.4:
                pick = random.choice([x for x in fl if x not in self.filler_recent] or fl)
                p = self.speech.key(pick, self.l1, self.voice)
                if p.exists():
                    f = str(p)
                    self.filler_recent = (self.filler_recent + [pick])[-3:]
            self.filler_last_turn = f is not None
        self.hearing = False
        # focus on the nearest voice: sound much quieter than the learner's own speech does not count as speech
        gate = 0.35 * self.voice_level if (self.focus and self.voice_level and not self.ptt) else 0.0
        vkw = {}
        if self.voice_id and voice == "enroll":
            vkw = {"voice": "enroll"}
        elif self.voice_id and voice == "check" and self.voice_s >= 2.0 and not self.ptt:
            if self.rejects >= 2:
                vkw = {"voice": "enroll"}                # two in a row dropped: it may be wrong. Listen, and learn again.
            else:
                vkw = {"voice": "check", "voice_min": self.VOICE_MIN, "voice_learn": self.VOICE_LEARN}

        def on_event(e):
            ev = e["event"]
            if ev == "listening":
                self.floor_open = floor
                self.floor_since = time.time()
                self.emit("mic", state="open")
                if self.ptt and not self.ptt_down.is_set():
                    self.speech.endpoint()               # the key was already released
            elif ev == "speech_start":
                self.hearing = True
                self.emit("mic", state="hearing")
            elif ev == "speech_end":
                self.floor_open = False
                self.emit("mic", state="thinking")
        try:
            if self.ptt:
                self.floor_open = floor
                self.emit("mic", state="ptt")
                end = time.time() + wait
                while self.running and not self.ptt_down.is_set() and self._scene is None and time.time() < end:
                    if self.speaking > 0:
                        end = time.time() + wait         # the clock starts once Olingo has stopped talking
                    time.sleep(0.03)
                self.floor_open = False
                if not self.running or not self.ptt_down.is_set():
                    self.emit("mic", state="off")
                    return {"text": "", "timeout": True, "t_end": time.time(), "cancelled": self._scene is not None}
                self.speech.stop()                       # holding the key cuts Olingo off
                h = self.speech.listen(lang=lang, alt=alt, wait=60, max_s=20, filler=f, ptt=True, fast=fast,
                                       hint=hint, on_event=on_event, **vkw)
            else:
                h = self.speech.listen(lang=lang, alt=alt, wait=wait, max_s=max_s, filler=f, fast=fast,
                                       min_level=gate, hint=hint, on_event=on_event, **vkw)
        finally:
            self.floor_open = False
        self.emit("mic", state="off")
        self.stats["heard_s"] += h.get("audio_s") or 0.0
        v = h.get("voice") or {}
        if v.get("enrolled_s") is not None and v["enrolled_s"] != self.voice_s:
            self.voice_s = v["enrolled_s"]
            self.emit("voice", learned_s=round(self.voice_s, 1), on=self.voice_s >= 2.0)
        if zhconv and h.get("lang") == "zh" and h.get("text"):       # Whisper often writes traditional characters
            h["text"] = zhconv.convert(h["text"], "zh-cn")
            for w in h.get("words") or []:
                w["w"] = zhconv.convert(w["w"], "zh-cn")
        return h

    def ptt_event(self, down: bool):
        if down:
            self.ptt_down.set()
        else:
            self.ptt_down.clear()
            try:
                self.speech.endpoint()
            except Exception:
                pass

    def set_ptt(self, on: bool):
        """Switch between open microphone and hold-to-talk in the middle of a call."""
        if on == self.ptt:
            return
        self.ptt = on
        self.ptt_down.clear()
        if self.floor_open and not self.hearing:
            self.speech.stop()                           # end the current listen; the next one uses the new mode

    def notify_scene(self, jpeg: bytes, chg: float = 1.0, still_s: float = 1.0) -> bool:
        """The scene watcher saw something new held still. If the floor is open and silent, Olingo speaks first.
        If Olingo itself is talking and the change is big, it stops mid-sentence, like a person who notices something.
        After "show me around" it waits: at least EXPLORE_WAIT, and until the picture has settled."""
        if not self.running or self._scene is not None or self._in_scene or self.hearing:
            return False
        if self.exploring is not None:
            if not self.floor_open or time.time() - self.exploring < self.EXPLORE_WAIT or still_s < self.EXPLORE_STILL:
                return False
            self._scene, self._explore, self._cut = jpeg, True, False
            self.speech.stop()
            return True
        if not self.notice:
            return False                                 # it looks when asked, or when the talk has run dry: not on movement
        if time.time() - self.last_pro < self.SCENE_GAP or self.streak >= self.PRO_MAX:
            return False
        if self.floor_open:
            if time.time() - self.floor_since < self.QUIET_S:
                return False                             # it has only just stopped talking: let the learner think
            self._cut = False
        elif self.speaking > 0 and chg >= self.CUT_CHG:
            self._cut = True
        else:
            return False
        self._scene = jpeg
        self.speech.stop()                               # ends the listen or the sentence; the loop picks the frame up
        return True

    # ---------- opening ----------
    def greet(self):
        """Hello -> "What's your name?" -> "Nice to meet you, X! How is your day going?" -> a short reply -> something
        it can already see. The two answers are also what the voice focus is learned from (in memory only)."""
        self.vary("hello", remember=True)
        self.last_q = L.line("ask_name", self.l1, self.l2)
        self.fixed("ask_name", remember=True)
        h = self.hear(lang=self.l1, wait=15.0, max_s=8.0, voice="enroll")
        if not self.running:
            return
        self.learn_level(h)
        if h.get("text") and not is_noise(h):
            self.stats["turns"] += 1
            self.stats["tier0"] += 1
            self.name = L.parse_name(h["text"], self.l1)
            self.mem.add("user", h["text"])
            self.emit("heard", text=h["text"], lang=h.get("lang"), audio_s=h.get("audio_s"), asr_ms=h.get("ms"))
            self.emit("route", tier=0, why="your name, read by a rule", model="none", image=False)
        self.say(L.line("nice_name", self.l1, self.l2, self.name) if self.name else L.line("nice", self.l1, self.l2),
                 remember=True, role="fixed")
        self.vi += 1
        warm = L.varied("warm", self.l1, self.vi)
        self.last_q = warm
        self.note_question(warm)
        self.mem.add("assistant", warm)
        self.say(warm, remember=True, role="fixed")
        h = self.hear(lang=self.l1, alt=self.l2, wait=15.0, max_s=12.0, filler=True, voice="enroll", fast=self.l1 == "en")
        if not self.running:
            return
        self.last_q = ""
        self.learn_level(h)
        if h.get("text") and not is_noise(h):
            self.stats["turns"] += 1
            self.emit("heard", text=h["text"], lang=h.get("lang"), audio_s=h.get("audio_s"), asr_ms=h.get("ms"))
            if intent(h["text"]) == "bye":
                self.running = False
                return
            self.respond(h["text"], None, h, brief=True)          # one friendly sentence, no question, no lesson
        if not self.running or self._scene is not None:
            return
        thing = self.next_askable()
        if thing:                                        # then straight to something it can already see
            self.make_offer(thing, "see_room")
        else:
            self.vary("open", remember=True)

    def learn_level(self, h: dict):
        """How loud the learner's own voice is. Sound much quieter than that is not treated as speech."""
        lv = h.get("level") or 0.0
        if lv > 0 and h.get("text") and (h.get("avg_p") or 0) >= 0.55:
            self.voice_level = lv if self.voice_level is None else 0.7 * self.voice_level + 0.3 * lv
            self.emit("focus", level=round(self.voice_level, 4), on=self.focus and not self.ptt)

    # ---------- what is left to talk about in view ----------
    def covered(self, word: str) -> bool:
        return word in self.words or word in self.declined or any(w == word for _, w in self.seen)

    def next_askable(self):
        skip = ("person", "man", "woman", "face", "wall", "brick wall", "background")
        return next((t for t in self.askable if t["l1"] not in skip and not self.covered(t["l1"])
                     and (not L.LANGS.get(self.l2, {}).get("cjk") or has_cjk(t["l2"]))), None)

    def make_offer(self, thing: dict, line_key: str):
        self.offer = (thing["l1"], L.trusted(thing["l1"], self.l1, self.l2) or thing["l2"], time.time() + 20)
        self.seen.append((time.time(), thing["l1"]))
        self.vi += 1
        self.last_q = L.offer_line(line_key, self.l1, self.l2, thing["l1"], self.vi)
        self.mem.add("assistant", self.last_q)
        self.say(self.last_q, remember=True, role="fixed")

    # ---------- never the same question twice ----------
    def note_question(self, q: str):
        self.asked.append(q)
        self.mem.asked(q)

    def is_repeat(self, q: str) -> bool:
        """Has this been asked already, in these or in other words? Exact match, then the sketch (no model), then a
        yes/no from Bonsai for questions that mean the same in different words."""
        n = norm(q)
        if not q.strip() or not self.asked:
            return False
        if q.strip() in self.asked or (n and n in [norm(a) for a in self.asked]):
            return True
        sim, prev = self.mem.closest(q)
        if sim >= 0.85:
            self.emit("repeat", how="sketch", sim=round(sim, 2), q=q, earlier=prev)
            return True
        same, info = brain.same_question(q, self.mem.nearest(q))
        self.stats["model_calls"] += 1
        self.stats["prompt_tokens"] += info.get("prompt_n") or 0
        self.stats["gen_tokens"] += info.get("predicted_n") or 0
        if same:
            self.emit("repeat", how="Bonsai 9B", sim=round(sim, 2), q=q, earlier="", ms=info.get("ms"))
        return same

    def new_topic(self, frame, why: str):
        """The talk has run dry: its next question would repeat an earlier one, or the learner stays quiet.
        Only now does it look for something else. First what it saw before; when that is used up, one fresh look at
        the learner (what they wear, what they are doing, what is around them); then it asks to be shown around."""
        thing = self.next_askable()
        if thing:
            self.make_offer(thing, "see_more")
        elif frame is not None and time.time() - self.dry_look > 60:
            self.dry_look = self.last_pro = time.time()
            self.respond("", frame, mode="idle", why_text=why)
        elif self.exploring is None:
            self.exploring = time.time()
            self.last_q = ""
            self.fixed("explore", remember=True)

    def explore_look(self, jpeg: bytes):
        """The camera has moved and settled: one fresh look at what is in view now."""
        self.emit("route", tier=2, why="you showed it around: a fresh look", model="9B", image=True)
        things, info = brain.room(jpeg, self.l1, self.l2)
        self.stats["model_calls"] += 1
        self.stats["tier2"] += 1
        self.stats["looks"] += 1
        self.stats["prompt_tokens"] += info.get("prompt_n") or 0
        self.stats["gen_tokens"] += info.get("predicted_n") or 0
        self.on_look()
        self.askable = [t for t in things if not self.covered(t["l1"])]
        thing = self.next_askable()
        self.emit("turn", tier=2, why="a fresh look" + ("" if thing else ": nothing new"), model="9B", marks={},
                  llm_ms=info.get("ms", 0), prefill_ms=0, prompt_n=info.get("prompt_n", 0), cache_n=0,
                  gen_n=info.get("predicted_n", 0), tok_s=info.get("tok_s", 0), image=True, silent=thing is None)
        self.emit("room", things=[t["l1"] for t in self.askable])
        if thing:
            self.exploring = None
            self.stats["spoke_first"] += 1
            self.make_offer(thing, "see_new")
        else:
            self.exploring = time.time()                 # keep looking, but not before another ten seconds

    # ---------- teaching one word ----------
    def card(self, key_l1: str, key: str):
        self.emit("card", l1=key_l1, l2=key, reading=reading(key, self.l2), readings=readings(key, self.l2))

    def note_word(self, key_l1: str, key: str, sc: float | None, example: str = ""):
        w = self.words.setdefault(key_l1, {"l2": key, "reading": reading(key, self.l2), "score": 0.0, "drilled": False,
                                           "example": "", "phrases": []})
        if example and not w["example"]:
            w["example"] = example
        if sc is not None:
            w["score"] = max(w["score"], sc)
            w["drilled"] = True
        self.emit("words", words=self.words, learned=sum(1 for x in self.words.values() if x["score"] >= 0.6))

    def _fetch_phrases(self, key_l1: str, key: str):
        """In the background, while the learner repeats the word: one phrase from a pattern that is always correct,
        one from Bonsai. They go on the flashcard; from the intermediate level they are also taught aloud."""
        out = []
        fp = frame_phrase(key_l1, key, self.l1, self.l2, len(self.words))
        if fp:
            out.append(fp)
        got, info = brain.phrases(key_l1, key, self.l1, self.l2, tier="9B")
        self.stats["model_calls"] += 1
        self.stats["prompt_tokens"] += info.get("prompt_n") or 0
        self.stats["gen_tokens"] += info.get("predicted_n") or 0
        for g in got:
            if len(out) < 2 and all(g["l2"] != o["l2"] for o in out):
                out.append(g)
        for o in out:
            o["reading"] = reading(o["l2"], self.l2)
            o["readings"] = readings(o["l2"], self.l2)
        self._phr[key_l1] = out

    def word_segs(self, text: str, key: str, key_l1: str) -> list:
        """Mark the new word inside a sentence of the learner's language, so it is spoken and shown as the new language."""
        i = text.lower().find(key.lower())
        if i < 0:
            return self.segs(text, self.l1)
        out = []
        if text[:i].strip():
            out.append({"t": text[:i]})
        out.append({"t": text[i:i + len(key)], "l2": True, "r": readings(key, self.l2), "g": key_l1})
        if text[i + len(key):].strip():
            out.append({"t": text[i + len(key):]})
        return out

    def introduce(self, key_l1: str, key: str, intro: str, kind: str):
        """Say the sentence that gives the learner the word. Bonsai's own sentence when it contains the word exactly
        and is short; otherwise one of several stock wordings. kind: "teach" (then they repeat) or "mention"."""
        text = self.tidy(intro or "", self.l1)
        ok = bool(text) and key.lower() in text.lower() and len(text) <= 120 and (self.l1 in ("en", "zh"))
        if ok and kind == "mention":
            text = re.split(r"(?<=[.!?。！？])\s*", text)[0]           # only the telling, not an invitation to repeat
            ok = key.lower() in text.lower()
        shape = " ".join(norm(text).split()[:2]) or text[:3]           # how the sentence opens
        if ok and shape == self.teach_shape:
            ok = False                                   # Bonsai worded it like last time: take another wording
        if not ok:
            self.vi += 1
            text = L.teach_line(kind, self.l1, self.l2, key_l1, key, self.vi)
            shape = " ".join(norm(text).split()[:2]) or text[:3]
        elif kind == "teach" and not re.search(r"[?？]|\b(say|try|turn|repeat)\b|说|读|试", text):
            self.vi += 1                                 # it told the word but did not hand over the turn
            text = text.rstrip() + " " + L.varied("invite", self.l1, self.vi)
        self.teach_shape = shape
        return self.say(text, segs=self.word_segs(text, key, key_l1), remember=True, role="teach")

    def mention(self, key_l1: str, key: str, example: str = "", intro: str = ""):
        """Small talk: tell them the word, do not drill it."""
        self.card(key_l1, key)
        self.introduce(key_l1, key, intro, "mention")
        self.note_word(key_l1, key, None, example)

    def teach(self, key_l1: str, key: str, example: str = "", intro: str = "") -> bool:
        """"In Chinese, an apple is 苹果. Can you say it?" -> listen -> colour by clarity -> one line of feedback -> phrases."""
        self.topic, self.topic_turns = key_l1, 0
        self.card(key_l1, key)
        th = threading.Thread(target=self._fetch_phrases, args=(key_l1, key), daemon=True)
        th.start()
        self.introduce(key_l1, key, intro, "teach")
        passed = False
        for attempt in (1, 2, 3):                        # a teacher models the word again, as often as it takes
            h = self.drill_hear(key)
            if not self.running or self._scene is not None:
                return False
            if not h.get("text") or is_noise(h):
                break                                    # silence or noise: move on quietly
            if self.not_an_attempt(h, key):
                self.carry = h                           # they said something else: it is answered as a normal turn
                break
            sc = score(key, h, self.l2)
            self.stats["tier0"] += 1
            self.emit("score", target=sc["chars"], score=sc["score"], heard=sc["heard"], asr_ms=h.get("ms"),
                      readings=readings(key, self.l2), heard_reading=reading(sc["heard"], self.l2),
                      ms=round((time.time() - h["t_end"]) * 1000))
            self.note_word(key_l1, key, sc["score"], example)
            # Feedback stays honest about what a speech recogniser can know: how clearly the word came through.
            # It never says which sound or tone was wrong, because on one short word it cannot tell.
            if sc["score"] >= 0.75:
                self.fixed("good")
                passed = True
                break
            if sc["score"] >= 0.45:
                self.fixed("ok")
                passed = True
                break
            if attempt == 3:                             # still not there: once more for the ear, then on with the call
                self.say(key, lang=self.l2, rate=0.75, role="word")
                self.fixed("try")
                break
            # not right yet: say it again for them, slowly and then at normal speed, and let them try again
            self.fixed("almost" if attempt == 1 else "slow_again")
            self.say(key, lang=self.l2, rate=0.7, role="word")
            self.say(key, lang=self.l2, rate=0.95, role="word")
        if not self.running or self._scene is not None or self.carry is not None:
            return False
        th.join(timeout=4.0 if (passed and self.level >= 2) else 0.05)
        ph = self._phr.get(key_l1) or []
        if ph and key_l1 in self.words:
            self.words[key_l1]["phrases"] = ph
            self.emit("words", words=self.words, learned=sum(1 for x in self.words.values() if x["score"] >= 0.6))
        if passed and self.level >= 2 and ph:            # intermediate and up: the word, then how it is really used
            self.emit("phrases", l1=key_l1, l2=key, phrases=ph)
            self.fixed("phrases")
            for p in ph:
                if not self.phrase_drill(p):
                    break
            if self.running:
                self.card(key_l1, key)                   # back to the word on the card
        return passed

    def not_an_attempt(self, h: dict, target: str) -> bool:
        """In a drill the learner said something that is not the word: a command, a question, a sentence in their
        own language. It must be answered, not scored as a bad repeat and not dropped."""
        text = h.get("text") or ""
        if intent(text) in ("bye", "again", "slower", "more", "less", "pick", "offline"):
            return True
        if len(text) > 6 * len(target) + 24:
            return True                                  # far longer than the word: a sentence
        own = h.get("lang") == self.l1 and self.l1 != self.l2
        longish = len(text) >= 8 if has_cjk(text) else len(norm(text).split()) >= 3
        if has_cjk(target):
            sentence = len(text) >= len(target) + 6
        else:                                            # three words more than the word asked for: a sentence
            sentence = len(norm(text).split()) >= len(norm(target).split()) + 3
        return bool(own and longish) or bool(ASKING.search(norm(text)) and longish) or sentence

    def drill_hear(self, target: str) -> dict:
        """Listen for the learner repeating a word or phrase. If they clearly said something but it could not be
        made out, it says so and listens once more, instead of moving on as if nothing had been said."""
        h = {}
        for again in (False, True):
            # alt: when it is unsure in the new language it also tries the learner's own, so a question or a
            # command spoken in the middle of a drill ("what does that mean?", "again", "bye") is understood
            h = self.hear(lang=self.l2, alt=self.l1, wait=self.DRILL_S, max_s=7.0, hint=target)
            if not self.running or self._scene is not None:
                return h
            spoke = (h.get("audio_s") or 0) >= 0.4 and not h.get("timeout") and not h.get("cancelled")
            unclear = not h.get("text") or is_noise(h)
            if again or not (spoke and unclear):
                return h
            self.stats["tier0"] += 1
            self.emit("ignored", why="it heard you, but not the word: it asks once more")
            self.fixed("sorry")
        return h

    def phrase_drill(self, p: dict) -> bool:
        """One phrase: first what it means, then the learner says it, then how clearly it came through.
        -> False when the learner did not try (the remaining phrases are skipped) or the call moved on."""
        self.vi += 1
        text = L.explain_line(self.l1, p["l1"], p["l2"], self.vi)
        self.emit("card", l1=p["l1"], l2=p["l2"], reading=reading(p["l2"], self.l2), readings=readings(p["l2"], self.l2))
        self.say(text, segs=self.word_segs(text, p["l2"], p["l1"]), remember=True, role="teach")
        self.vary("invite")
        h = self.drill_hear(p["l2"])
        if not self.running or self._scene is not None:
            return False
        if not h.get("text") or is_noise(h):
            return False
        if self.not_an_attempt(h, p["l2"]):
            self.carry = h                               # answered as a normal turn by the loop
            return False
        sc = score(p["l2"], h, self.l2)
        self.stats["tier0"] += 1
        self.emit("score", target=sc["chars"], score=sc["score"], heard=sc["heard"], asr_ms=h.get("ms"),
                  readings=readings(p["l2"], self.l2), heard_reading=reading(sc["heard"], self.l2),
                  ms=round((time.time() - h["t_end"]) * 1000))
        p["score"] = round(sc["score"], 2)
        if sc["score"] < 0.45:                           # not right yet: model it twice, slowly first, and one more try
            self.fixed("almost")
            self.say(p["l2"], lang=self.l2, rate=0.7, role="word")
            self.say(p["l2"], lang=self.l2, rate=0.95, role="word")
            h = self.drill_hear(p["l2"])
            if not self.running or self._scene is not None:
                return False
            if h.get("text") and not is_noise(h):
                if self.not_an_attempt(h, p["l2"]):
                    self.carry = h
                    return False
                sc = score(p["l2"], h, self.l2)
                self.stats["tier0"] += 1
                self.emit("score", target=sc["chars"], score=sc["score"], heard=sc["heard"], asr_ms=h.get("ms"),
                          readings=readings(p["l2"], self.l2), heard_reading=reading(sc["heard"], self.l2),
                          ms=round((time.time() - h["t_end"]) * 1000))
                p["score"] = round(sc["score"], 2)
        if sc["score"] >= 0.75:
            self.fixed("good")
        elif sc["score"] >= 0.45:
            self.fixed("ok")
        else:
            self.say(p["l2"], lang=self.l2, rate=0.75, role="word")
            self.fixed("try")
        return True

    def set_level(self, level: int, line_key: str | None = None):
        self.level = min(max(level, 1), 3)
        self.emit("level", level=self.level, name=L.LEVEL_NAMES[self.level])
        if line_key:
            self.fixed(line_key)

    # ---------- one Bonsai turn ----------
    def note(self, sees: bool = True) -> str:
        """What is carried from turn to turn besides the last 30 seconds: the name, learned words, what was just seen,
        and the compressed memory of everything older (what the learner told it, what it has asked about)."""
        parts = []
        if self.name:
            parts.append(f"My name is {self.name}.")
        learned = [k for k, w in self.words.items() if w["score"] >= 0.6][-10:]
        if learned:
            parts.append("Words I already learned: " + ", ".join(learned) + ".")
        now = time.time()
        just = list(dict.fromkeys(w for t, w in self.seen if now - t < 60))[-3:]
        if just and sees:                                # only with a picture: without one it talked as if it could see
            parts.append("You looked a moment ago and saw: " + ", ".join(just) + ".")
        told = self.mem.summary()["told"]                # what is left of turns older than the window
        if told:
            parts.append("Earlier in this call I told you about: " + "; ".join(told[-8:]) + ". Remember it.")
        topics = list(dict.fromkeys(g for g in (memory.gist(q) for q in self.asked) if g))
        if topics:
            parts.append("You already asked me about: " + "; ".join(topics[-14:]) + ". Never ask about any of these "
                         "again, in any words.")
        return "(" + " ".join(parts) + ") " if parts else ""

    def respond(self, text: str, frame, h: dict | None = None, mode: str | None = None, why_text: str | None = None,
                brief: bool = False):
        """mode None = the learner spoke. Otherwise Olingo speaks first:
        scene (something new is held up), idle (silence), stuck (its question went unanswered), tour ("you pick")."""
        why = {None: None, "scene": "noticed something new", "idle": "the talk ran dry, so it looked for something new",
               "stuck": "no answer: it helps instead of repeating", "tour": "you asked it to pick"}[mode]
        if mode and why_text:
            why = why_text
        hist = self.mem.window()
        if mode:
            tier = 2
        elif brief:
            tier, why = 1, "small talk, no image needed"
        else:
            tier, why = route(text)
        if tier == 2 and frame is None:
            tier, why = 1, why_text or "no camera frame yet"
        strong = self.l1 in ("en", "zh")                # the languages Bonsai writes surely
        # What the learner says is answered by the 9B (with a frame only when they talk about the picture): the 4B
        # lost the thread of a conversation too easily. The 4B keeps the two short lines that need no context: the
        # reply to the warm-up question and the remark after a new word.
        light = brief or bool(why_text)
        model = "4B" if (light and strong and tier == 1 and brain.available("4B")) else "9B"
        # right-sized looking: naming one thing goes to the 4B first; the 9B is called in only when the 4B is unsure
        # Looking is the 9B's job. The 4B-first shortcut for "what is this?" was faster (1.1 s against 1.6 s) and as
        # accurate on test frames, but in real use it too readily said the learner was holding something. It stays
        # available behind OLINGO_CASCADE=1; small talk without a picture is where the 4B is used.
        cascade = (os.environ.get("OLINGO_CASCADE") == "1" and tier == 2 and not mode and strong and self.level < 3
                   and brain.available("4B") and (SIMPLE_LOOK.search(norm(text)) or SIMPLE_LOOK.search(text)))
        if cascade:
            model = "4B"
        t_ref = h["t_end"] if h else time.time()
        marks = {"asr": h.get("ms") if h else 0, "transcript": round((time.time() - t_ref) * 1000)}
        self.emit("route", tier=tier, why=why, model=model, image=tier == 2)
        self.last_said = []
        lv3 = self.level >= 3
        say_lang = self.l2 if lv3 else self.l1
        spoken, dup, unsure = [], [], []
        hold = mode in ("scene", "tour") or not strong   # the whole answer is needed before anything is said
        if mode == "tour":
            prompt = brain.PROACTIVE["tour"].format(known=", ".join(list(self.words) + [w for _, w in self.seen][-6:]) or "nothing yet")
        elif mode == "stuck":
            prompt = brain.PROACTIVE["stuck"]
        else:
            prompt = brain.PROACTIVE[mode] if mode else text
        prompt = self.note(sees=tier == 2) + prompt

        echoed, say_q = [], []

        def on_say(piece, final):
            marks.setdefault("first_ready", round((time.time() - t_ref) * 1000))
            if hold or dup or not piece.strip():
                return
            if re.search(r"[?？]\s*$", piece.strip()):
                say_q.append(piece.strip())              # a question inside the answer: it waits its turn (it goes
                return                                   # last, once, and only if it has not been asked before)
            if (text and not mode and float(memory.sketch(piece) @ memory.sketch(text)) >= 0.7
                    and not re.search(r"\b(too|also|as well|either|moi aussi)\b|也|も", piece)):
                echoed.append(piece)                     # it is only handing the learner's sentence back: leave it out
                return
            if norm(piece) in self.recent or piece.strip() in self.recent:
                dup.append(piece)                        # the model is repeating itself: do not say it again
                return
            spoken.append(self.reply(piece, say_lang))

        def on_field(k, v):
            if k == "gloss" and v and not hold:
                self.emit("gloss", text=v)
            if cascade and not unsure and ((k == "key_l1" and (not v.strip() or v.strip().lower() in VAGUE))
                                           or (k == "conf" and v is not None and v < 0.6)):
                unsure.append(k)
                self.speech.stop()                       # take back the sentence before it is heard
        d, info = brain.answer(prompt, frame if tier == 2 else None, tier=model, l1=self.l1, l2=self.l2,
                               level=self.level, history=hist, on_say=on_say, on_field=on_field,
                               confidence=bool(cascade))
        if cascade and (unsure or d is None):            # the 4B was not sure what it saw: the 9B looks
            self.stats["model_calls"] += 1
            self.stats["prompt_tokens"] += info.get("prompt_n") or 0
            self.stats["gen_tokens"] += info.get("predicted_n") or 0
            self.stats["escalated"] += 1
            self.emit("route", tier=2, why="the 4B was unsure, so the 9B looks", model="9B", image=True, again=True)
            model, cascade = "9B", False
            self.last_said, spoken, dup = [], [], []
            d, info = brain.answer(prompt, frame, tier="9B", l1=self.l1, l2=self.l2, level=self.level,
                                   history=hist, on_say=on_say, on_field=on_field)

        def count(i):
            self.stats["model_calls"] += 1
            self.stats["prompt_tokens"] += i.get("prompt_n") or 0
            self.stats["gen_tokens"] += i.get("predicted_n") or 0
        count(info)
        self.stats[f"tier{tier}"] += 1
        if tier == 2:
            self.stats["looks"] += 1
            self.on_look()

        def quiet_turn(reason):
            self.emit("turn", tier=tier, why=reason, model=model, marks=marks, llm_ms=info.get("ms", 0),
                      prefill_ms=info.get("prompt_ms", 0), prompt_n=info.get("prompt_n", 0),
                      cache_n=info.get("cache_n", 0), gen_n=info.get("predicted_n", 0), tok_s=info.get("tok_s", 0),
                      image=tier == 2, silent=True)
        if d is None:
            self.emit("error", code="bonsai", message=info.get("error", "Bonsai did not answer"))
            quiet_turn("Bonsai did not answer")
            if not mode:
                self.fixed("sorry")
            return
        if echoed and not spoken and not dup and not hold:
            spoken.append(self.reply(echoed[0], say_lang))           # it had nothing else to say: better than silence
        if say_q and not d["q"].strip():
            d["q"] = say_q[0]
        if say_q:                                        # what is remembered and shown is what was actually said
            d["say"] = " ".join(p for p in re.split(r"(?<=[.!?。！？])\s*", d["say"]) if p.strip() and p.strip() not in say_q)
        if dup and not hold:                             # ask once more, without the history it was copying from
            d2, info2 = brain.answer(prompt, frame if tier == 2 else None, tier=model, l1=self.l1, l2=self.l2,
                                     level=self.level, history=[], temperature=0.7)
            count(info2)
            fresh = d2 is not None and norm(d2["say"]) not in self.recent and d2["say"].strip() not in self.recent
            if mode and not fresh:
                quiet_turn("looked: nothing new to say")
                return
            if d2 is not None:
                d, info = d2, info2
            spoken.append(self.reply(d["say"], say_lang))
        if not strong and mode not in ("scene", "tour"):
            say2 = d["say"]
            if self.l1 in L.NAMES_IT and not mode and d["key_l1"].strip() and (SIMPLE_LOOK.search(text) or re.search(r"これは何|何を持|なんですか|何ですか|何と言", text)):
                say2 = L.NAMES_IT[self.l1].format(thing=d["key_l1"].strip())      # a sentence that cannot be wrong
            else:
                say2, pi = brain.polish(d["say"], self.l1)
                count(pi)
            d["say"] = say2
            marks["first_ready"] = round((time.time() - t_ref) * 1000)
            spoken.append(self.reply(say2, say_lang))
            if d["q"]:
                bank = L.SAFE_QUESTIONS.get(self.l1)
                if bank and tier == 2:
                    self.safe_i += 1
                    d["q"] = next((q for q in bank[self.safe_i % len(bank):] + bank if q not in self.asked), "")
                else:
                    d["q"], pi = brain.polish(d["q"], self.l1)
                    count(pi)
        key_l1, key = d["key_l1"].strip().lower(), d["key"].strip()
        if self.l2 == "zh" and zhconv:
            key = zhconv.convert(key, "zh-cn")
        # Bonsai's Chinese is reliable, so the word list only fills gaps; for other languages the list wins.
        if self.l2 == "zh":
            key = key if has_cjk(key) else (L.trusted(key_l1, self.l1, self.l2) or key)
        else:
            key = L.trusted(key_l1, self.l1, self.l2) or key
        valid = bool(key and key_l1 and key.lower() != key_l1 and len(key) <= 16
                     and (not L.LANGS.get(self.l2, {}).get("cjk") or has_cjk(key)))
        new = valid and not self.words.get(key_l1, {}).get("drilled")
        if mode in ("scene", "tour"):
            empty_hands = mode == "scene" and re.search(r"\bwear|穿|porte|着て", d["say"])     # it only described clothes
            if norm(d["say"]).startswith("none") or not new or empty_hands:
                quiet_turn("looked: nothing new to show" if mode == "scene" else "looked: found nothing new")
                if mode == "tour":
                    self.fixed("pick_none")
                elif self._cut:
                    self.fixed("nevermind")              # it had stopped itself for nothing
                return
            spoken.append(self.reply(d["say"], say_lang))
            if lv3 and d.get("gloss"):
                self.emit("gloss", text=d["gloss"])
        if not mode and not any(spoken):                 # whatever happened above, the learner is never left unanswered
            fallback = d["say"].strip() or (say_q[0] if say_q else "")
            if fallback:
                spoken.append(self.reply(fallback, say_lang))
                if say_q and fallback == say_q[0]:
                    self.note_question(fallback)
                    self.last_q = fallback
                    d["q"] = ""
            else:
                spoken.append(self.fixed("sorry"))
        first = next((s for s in spoken if s), None)
        if first:
            first.started.wait(8.0)
            if first.t_start and h:                      # only a real spoken turn has a "you stopped talking" moment
                marks["audio"] = round((first.t_start - t_ref) * 1000)
        self.emit("turn", tier=tier, why=why, model=model, marks=marks, llm_ms=info["ms"], prefill_ms=info["prompt_ms"],
                  prompt_n=info["prompt_n"], cache_n=info["cache_n"], gen_n=info["predicted_n"], tok_s=info["tok_s"],
                  image=tier == 2, first_sentence_ms=info["marks"].get("first_sentence"), reply=d)
        if mode:
            self.stats["spoke_first"] += 1
        asked_before = bool(d["q"]) and (norm(d["q"]) in self.recent or d["q"].strip() in self.recent)
        for piece in re.split(r"(?<=[.!?。！？])\s*", d["say"] + " " + d["q"]):
            if piece.strip():
                self.recent.append(norm(piece) or piece.strip())
        self.recent = self.recent[-10:]
        if tier == 2 and valid:
            self.seen.append((time.time(), key_l1))
            self.seen = self.seen[-8:]
        if not mode and not why_text:                    # what the learner really said, not a prompt of its own
            self.mem.add("user", text, t_ref)
        self.mem.add("assistant", d["say"])              # its question is added when (and only if) it is asked
        if brief:
            return
        if mode in ("scene", "tour") and new:            # it noticed something: offer the word, do not push it
            self.offer = (key_l1, key, time.time())
            self.last_q = L.line("offer", self.l1, self.l2)
            self.fixed("offer", remember=True)
            return
        taught = False
        if new and not mode:
            if tier == 2 or ASKS_HOW.search(norm(text)) or ASKS_HOW.search(text):
                self.teach(key_l1, key, example=d["say"], intro=d.get("teach", ""))
                taught = True
            elif key_l1 not in self.words and self.stats["turns"] - self.mentioned_at >= 2:
                self.mentioned_at = self.stats["turns"]  # a word in passing, but not on every single turn
                self.mention(key_l1, key, example=d["say"], intro=d.get("teach", ""))
        if not self.running or self._scene is not None or self.carry is not None:
            return                                       # the learner said something in the drill: that comes first
        if not taught:
            self.topic_turns += 1
        learner_asked = bool(not mode and (ASKING.search(norm(text)) or ASKING.search(text)))
        wants_q = bool(d["q"].strip()) and (taught or not learner_asked)
        # a follow-up only when it is new, and not on top of an answer to the learner's own question:
        # they asked, Olingo answered, the floor is theirs
        repeat = wants_q and (asked_before or self.is_repeat(d["q"]))
        if wants_q and not repeat:
            self.stale = 0
            self.ask(d["q"], say_lang, gloss=d.get("q_gloss", "") if lv3 else "")
            return
        if repeat:
            self.stale += 1                              # the question is left out; the learner has the floor
        if self.stale >= 2 and not learner_asked and not mode:
            # Its questions have started to repeat. Only now does it turn to something else: what it saw earlier,
            # then one fresh look at the learner, then "show me around". With no question at all it simply stops
            # talking: the learner has the floor, and silence is handled by the loop.
            self.stale = 0
            self.next_at = self.stats["turns"]
            self.new_topic(frame, "its questions were starting to repeat")

    # ---------- what the learner said ----------
    def turn(self, h: dict, frame) -> bool:
        text = h["text"]
        self.stats["turns"] += 1
        said_l2 = h.get("lang") == self.l2
        self.emit("heard", text=text, lang=h.get("lang"), words=h.get("words") if said_l2 else None,
                  audio_s=h.get("audio_s"), asr_ms=h.get("ms"), tried=h.get("tried"))
        offer = self.offer if self.offer and time.time() - self.offer[2] < 40 else None
        self.offer = None
        self.last_q = ""
        it = intent(text, offer is not None)
        if it == "yes" and (len(text) >= 14 if has_cjk(text) else len(norm(text).split()) >= 6):
            # a long answer that starts like a yes may ask for something else ("yes, but teach me cat first"):
            # Bonsai reads the whole sentence before the offered word is taught
            ok, _ = brain.accepts_offer(text, offer[0])
            self.stats["model_calls"] += 1
            if not ok:
                it = "ask"
        if it == "ask":
            self.respond(text, frame, h)
            return True
        self.mem.add("user", text)
        self.stats["tier0"] += 1
        self.emit("route", tier=0, why={"bye": "command: goodbye", "again": "command: say it again",
                                        "slower": "command: slower", "more": "command: more of the new language",
                                        "less": "command: less of the new language", "yes": "yes: teach the word",
                                        "no": "no: skip the word", "pick": "command: you pick",
                                        "offline": "about the outside world: it is offline and says so"}[it],
                  model="none", image=False)
        if it == "bye":
            g = self.fixed("bye")
            if g:
                g.done.wait(8.0)                         # let the goodbye finish before the receipt
            return False
        if it in ("again", "slower"):
            if not self.last_said:
                self.fixed("again_none")
            for t, lang, segs in list(self.last_said):
                self.say(t, lang=lang, rate=0.8 if it == "slower" else 1.0, segs=segs, role="reply")
        elif it == "more":
            self.set_level(self.level + 1, "more")
        elif it == "less":
            self.set_level(self.level - 1, "less")
        elif it == "yes":
            self.teach(offer[0], offer[1])
            if self.running and self._scene is None and self.carry is None:   # keep the thread: one remark and a question
                self.respond(f"I just learned the word {offer[0]}. Tell me one thing about it.", None,
                             why_text="keeps the thread on the new word")
        elif it == "no":
            self.declined.add(offer[0])
            self.fixed("no_problem")
            nxt = self.next_askable()
            if nxt:
                self.make_offer(nxt, "see_more")
            else:
                self.vary("open", remember=True)
        elif it == "offline":
            self.last_q = L.line("offline", self.l1, self.l2)
            self.fixed("offline", remember=True)
        elif it == "pick":
            if frame is None:
                self.fixed("pick_none")
            else:
                self.respond("", frame, mode="tour")
        return True

    # ---------- the call ----------
    def run(self):
        self.running = True
        try:
            self.emit("profile", l1=self.l1, l2=self.l2, level=self.level, l1_name=L.name(self.l1, "en"),
                      l2_name=L.name(self.l2, "en"), level_name=L.LEVEL_NAMES[self.level], voice=self.voice,
                      room=self.room)
            self.greet()
            self.emit("phase", phase="floor")
            quiet = 0
            while self.running:
                # the learner's language goes to the small Whisper first (English: 0.16 s); the large one backs it up
                h, self.carry = self.carry, None         # said during a drill and not yet answered
                if h is None:
                    h = self.hear(lang=self.l1, alt=self.l2, wait=self.IDLE_S, filler=True, floor=True,
                                  fast=self.l1 == "en", voice="check")
                frame = self.get_frame()                 # the frame at the moment the learner stopped talking
                if not self.running:
                    break
                if self._scene is not None and self._explore:        # shown around: the picture has settled
                    jpeg, self._scene, self._explore = self._scene, None, False
                    self._in_scene = True
                    try:
                        self.explore_look(jpeg)
                    finally:
                        self._in_scene = False
                    continue
                if self._scene is not None:              # something new was held up
                    self._in_scene = True
                    try:
                        jpeg = self._scene
                        self.last_pro = time.time()
                        self.streak += 1
                        self.respond("", jpeg, mode="scene")
                    finally:
                        self._scene, self._cut, self._in_scene = None, False, False
                    continue
                if h.get("error"):
                    time.sleep(2.0)                      # the microphone is unavailable: do not spin
                    continue
                if h.get("rejected"):                    # someone else's voice: not heard, not answered
                    self.rejects += 1
                    self.stats["ignored"] += 1
                    self.emit("ignored", why="another voice", sim=round((h.get("voice") or {}).get("sim", 0), 2))
                    continue
                if is_noise(h):
                    if h.get("text"):
                        self.stats["ignored"] += 1
                    # They clearly spoke (over a second, loud enough to pass the nearest-voice gate) but no words
                    # came out of it: say so, once in a while, instead of leaving them unanswered.
                    if (not h.get("timeout") and not h.get("cancelled") and (h.get("audio_s") or 0) >= 1.2
                            and time.time() - self.sorry_at > 20):
                        self.sorry_at = time.time()
                        self.stats["tier0"] += 1
                        self.emit("route", tier=0, why="it heard you but could not make out the words", model="none",
                                  image=False)
                        self.fixed("sorry")
                        continue
                    if h.get("timeout"):
                        quiet += 1
                        if self.exploring is not None and time.time() - self.exploring < 40:
                            self.emit("idle", quiet=quiet)           # being shown around: no chatter, just watch
                            continue
                        # Silence is not a reason to look at the learner. A question is waiting: one gentle word,
                        # no model, no picture. Still nothing after that: something else to talk about.
                        if quiet == 1 and time.time() - self.last_pro >= self.PRO_GAP:
                            self.last_pro = time.time()
                            if self.last_q:
                                self.stats["tier0"] += 1
                                self.emit("route", tier=0, why="you are thinking: it waits, without looking",
                                          model="none", image=False)
                                self.vary("nudge")
                            else:
                                self.new_topic(frame, "you were quiet and nothing was asked")
                            continue
                        if quiet == 2 and self.last_q:
                            self.last_q = ""
                            self.last_pro = time.time()
                            self.new_topic(frame, "its question got no answer: something else instead")
                            continue
                    self.emit("idle", quiet=quiet)
                    continue
                # not finished ("and then ...", "because", "就是"): keep listening instead of answering half a thought
                more = 0
                while self.running and not self.ptt and more < 2 and UNFINISHED.search(h["text"].strip()):
                    more += 1
                    h2 = self.hear(lang=self.l1, alt=self.l2, wait=4.0, floor=True, fast=self.l1 == "en", voice="check")
                    if is_noise(h2) or not h2.get("text"):
                        break
                    h = {**h2, "text": h["text"].rstrip(" ,，、.。…") + " " + h2["text"], "t_end": h2.get("t_end", time.time())}
                    frame = self.get_frame()
                quiet = 0
                self.streak = 0
                self.rejects = 0
                self.exploring = None                    # the learner spoke: they lead again
                self.learn_level(h)
                if not self.turn(h, frame):
                    break
        finally:
            self.running = False
            self.floor_open = False
            self.emit("phase", phase="done")

    def deck(self) -> list:
        """What is kept after the call: words, never audio or pictures."""
        return [{"l1": k, "l2": w["l2"], "reading": w["reading"], "score": w["score"], "drilled": w["drilled"],
                 "example": w.get("example", ""), "phrases": w.get("phrases", []), "lang": self.l2, "from": self.l1}
                for k, w in self.words.items()]

    def voice_summary(self) -> dict:
        return {"on": self.voice_id and self.voice_s >= 2.0, "learned_s": round(self.voice_s, 1)}

    def stop(self):
        self.running = False
        try:
            self.speech.stop()
        except Exception:
            pass
