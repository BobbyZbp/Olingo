"""ANSWER: the one conversational Bonsai call. Streamed; the first sentence of `say` is handed to the caller the
moment it is complete, so speech starts while the rest of the JSON is still being generated.
  tier 2 = Bonsai 9B with the current frame      tier 1 = Bonsai 4B, text only (9B without the image in lite mode)
Output is forced by a GBNF grammar (this llama-server silently ignores response_format json_schema)."""
import http.client
import json
import math
import os
import re
import time
import urllib.request
from urllib.parse import urlparse

from . import lang as L
from .vision import data_url

URLS = {"9B": os.environ.get("BONSAI_9B_URL", "http://127.0.0.1:8082"),
        "4B": os.environ.get("BONSAI_4B_URL", "http://127.0.0.1:8084")}      # both are started with their vision tower

COMMON = {"cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False},
          "temperature": 0.3, "top_p": 0.8, "top_k": 20, "min_p": 0, "presence_penalty": 0, "repeat_penalty": 1.0}

_T = r'[^"\\\n:{} ] [^"\\\n{}]{0,150}'             # a sentence: no quotes, never starts with a space or colon
_K = r'([^"\\\n:{}. ] [^"\\\n:{}.]{0,24})?'        # a single word, may be empty
_Q = '(' + _T + ')?'                                # a question, or nothing when there is nothing new to ask
GBNF = ('root ::= "{\\"say\\":\\"" t "\\",\\"key_l1\\":\\"" k "\\",\\"key\\":\\"" k "\\",\\"teach\\":\\"" tq "\\",\\"q\\":\\"" tq "\\"}"\n'
        f't ::= {_T}\ntq ::= {_Q}\nk ::= {_K}\n')
# level 3: Olingo speaks the target language, so every sentence comes with its meaning
GBNF3 = ('root ::= "{\\"say\\":\\"" t "\\",\\"gloss\\":\\"" t "\\",\\"key_l1\\":\\"" k "\\",\\"key\\":\\"" k '
         '"\\",\\"q\\":\\"" t "\\",\\"q_gloss\\":\\"" t "\\"}"\n' f't ::= {_T}\nk ::= {_K}\n')

# How Olingo must write in each language it speaks to the learner in. The rest of the prompt is in English, so
# without this the smaller languages drift: mixed registers, odd questions.
STYLE = {
    "fr": ("Write simple, natural, correct French. Always say tu, never vous. Good questions are short and concrete, "
           "like: Tu l'utilises souvent ? / De quelle couleur est-il ? / Tu aimes ça ? / C'est à toi ? "),
    "ja": ("Write simple, natural, correct Japanese in the polite です/ます style, with everyday words only. Good "
           "questions are short and concrete, like: それは何色ですか？ / よく使いますか？ / それは好きですか？ / "
           "どこで買いましたか？ "),
    "zh": "Write simple, natural Mandarin in simplified characters. ",
}

CUP = {"zh": "杯子", "es": "taza", "fr": "tasse", "ja": "コップ", "ko": "컵", "de": "Tasse", "it": "tazza", "pt": "copo",
       "en": "cup"}
NOODLES = {"zh": "面条", "es": "fideos", "fr": "nouilles", "ja": "麺", "ko": "국수", "de": "Nudeln", "it": "pasta",
           "pt": "macarrão", "en": "noodles"}
LEVEL_HINT = {1: "The learner is a complete beginner.", 2: "The learner knows a few words.",
              3: "The learner can follow short simple sentences."}


def system_prompt(l1: str, l2: str, level: int, sees: bool) -> str:
    L1, L2 = L.name(l1, "en"), L.name(l2, "en")
    # One wording for turns with and without a picture, so the cached prompt is reused whichever comes next.
    see = ("Sometimes a webcam image of the learner is attached to their message. Only then can you see them: talk "
           "only about objects, clothes, colors and actions that are really in that image; never guess age, identity "
           "or feelings; say that they are holding something ONLY when an object is clearly in their hand. When no "
           "image is attached you cannot see them: do not describe them or their room, just talk with them. ")
    head = (f"You are Olingo, a warm {L2} tutor on a video call. The learner speaks {L1} and is learning {L2}. "
            f"{LEVEL_HINT[min(max(level, 1), 3)]} {see}"
            "You are offline: you know nothing about today's news, weather, date or time. When that comes up, say you "
            "cannot know and ask the learner. Never invent facts. "
            "Follow the learner's lead: when they ask you something, answer exactly that; when they talk about "
            "something else, go with it. Never go back to a question you asked earlier. "
            "Talk like a friend, not like a quiz: never repeat or rephrase what the learner just said; add something "
            "of your own instead, a short reaction, a small fact or your opinion. "
            + STYLE.get(l1, "") +
            "Reply with one JSON object and nothing else. ")
    simp = " (simplified characters only, no pinyin)" if l2 == "zh" else ""
    if level >= 3:
        cup2 = CUP.get(l2, "cup")
        ex = {"zh": ("我看到了。那是一个蓝色的杯子。", "你喜欢喝咖啡吗？"), "es": ("Sí, la veo. Es una taza azul.", "¿Te gusta el café?")}
        s2, q2 = ex.get(l2, ("...", "..."))
        return head + (
            f'"say" = your reply in simple {L2}{simp}: first one very short sentence, then one more short sentence; at '
            f'most 14 words in total; no question in it. "gloss" = the same reply in {L1}. '
            f'"key_l1" = the ONE {L1} noun most worth teaching (the thing they show or mention), or "". '
            f'"key" = that word in {L2}{simp}, or "". '
            f'"q" = one short easy follow-up question in {L2}{simp}, at most 8 words. "q_gloss" = that question in {L1}. '
            + ("" if l1 != "en" else
               f'Example: {{"say":"{s2}","gloss":"I see it. That is a blue cup.","key_l1":"cup","key":"{cup2}",'
               f'"q":"{q2}","q_gloss":"Do you like coffee?"}}'))
    cup = CUP.get(l2, "cup")
    ex = {"zh": f'{{"say":"那是一个蓝色的杯子。","key_l1":"杯子","key":"{cup}","teach":"杯子的{L.name(l2, "zh")}是 {cup}。你来说一遍？","q":"你喜欢喝咖啡吗？"}}',
          "fr": f'{{"say":"C\'est une tasse bleue.","key_l1":"tasse","key":"{cup}","teach":"En {L.name(l2, "fr")}, une tasse se dit {cup}. Tu peux le dire ?","q":"Tu aimes le café ?"}}',
          "ja": f'{{"say":"それは青いカップです。","key_l1":"カップ","key":"{cup}","teach":"カップは{L.name(l2, "ja")}で {cup} と言います。言ってみてください。","q":"コーヒーは好きですか？"}}',
          }.get(l1, f'{{"say":"That is a blue cup.","key_l1":"cup","key":"{cup}","teach":"In {L2}, a cup is {cup}. Can you say it?","q":"Do you like coffee?"}}')
    nd = NOODLES.get(l2, "noodles")
    ex2 = {"zh": f'他们说“我中午吃了面条。” -> {{"say":"面条很好吃呢。","key_l1":"面条","key":"{nd}","teach":"面条用{L.name(l2, "zh")}说是 {nd}。","q":"你喜欢什么面条？"}}',
           "fr": f'S\'ils disent « J\'ai mangé des nouilles à midi. » -> {{"say":"Les nouilles, c\'est délicieux.","key_l1":"nouilles","key":"{nd}","teach":"Les nouilles, en {L.name(l2, "fr")}, c\'est {nd}.","q":"Tu en manges souvent ?"}}',
           "ja": f'「昼ごはんに麺を食べました。」と言われたら -> {{"say":"麺はおいしいですよね。","key_l1":"麺","key":"{nd}","teach":"麺は{L.name(l2, "ja")}で {nd} です。","q":"どんな麺が好きですか？"}}',
           }.get(l1, f'If they say "I had noodles for lunch." -> {{"say":"Noodles sound delicious.","key_l1":"noodles","key":"{nd}","teach":"Noodles in {L2}: that is {nd}.","q":"What kind of noodles do you like?"}}')
    return head + (
        f'"say" = your reply in {L1}: one short sentence that directly answers what they said or asked, with one '
        f'concrete detail; a second short sentence only if it adds something. At most 16 words. No question in it. '
        f'Never repeat a sentence you already said. '
        f'"key_l1" = the ONE {L1} noun from your reply most worth teaching (the thing they show or mention), or "". '
        f'"key" = that word in {L2}{simp}, or "". '
        f'"teach" = when key is not empty: one natural sentence in {L1} that tells them this word in {L2}, with the '
        f'{L2} word written exactly as in key; the way a teacher would say it, worded differently every time (never '
        f'"repeat after me"); otherwise "". '
        f'"q" = one short easy follow-up question in {L1}, at most 8 words: something a friendly teacher would really '
        f'ask next, about what THEY just said or showed; or "" when you have nothing new to ask. Never ask for something '
        f'they have just told you, and never ask the same question twice. '
        f'Example when they show a cup: {ex} Example of small talk: {ex2}')


# what replaces the learner's words when Olingo speaks first
PROACTIVE = {
    "scene": ("(The learner did not say anything. They just moved, or picked something up. If they are clearly holding "
              "up or showing an object to the camera, say you noticed it and name it, like: You picked up a banana. "
              "Put the object in key_l1 and key. If their hands are empty or you are not sure, reply with say = none.)"),
    "idle": ("(The learner is quiet. Say one friendly thing about something you can really see that you have not "
             "talked about yet: something they wear or something in the room. Do not say they are holding anything "
             "unless it is clearly in their hand. Then ask an easy question about it.)"),
    "stuck": ("(The learner did not answer your last question. Do NOT ask it again and do not repeat yourself. In one "
              "short sentence, help: give a tiny hint or the answer. Then ask a different, easier question.)"),
    "tour": ("(The learner asked you to choose. Pick ONE object you can clearly see in the picture that is not in this "
             "list: {known}. Say that you can see it, like: I can see a lamp behind you. Put it in key_l1 and key.)"),
}

_FIELD = r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"'
_SENT_END = re.compile(r'[.!?](?=\s)|[。！？]')


def _fields(buf: str, names) -> dict:
    """Fields whose closing quote has already arrived."""
    d = {}
    for key in names:
        m = re.search(_FIELD % key, buf)
        if m:
            try:
                d[key] = json.loads('"%s"' % m.group(1))
            except ValueError:
                pass
    return d


def _first_sentence(buf: str) -> str | None:
    """The first sentence of a `say` value that is still being written, once its end mark has arrived."""
    m = re.search(r'"say"\s*:\s*"((?:[^"\\]|\\.)*)', buf)
    if not m:
        return None
    s = m.group(1)
    e = _SENT_END.search(s)
    if not e:
        return None
    piece = s[:e.end()]
    return piece if len(piece.strip()) >= 3 else None


def _confidence(toks, field: str) -> float | None:
    """How sure the model was of one field: the mean probability of the tokens that make up its value (0 to 1)."""
    text, spans = "", []
    for tok, lp in toks:
        spans.append((len(text), len(text) + len(tok), lp))
        text += tok
    m = re.search(r'"%s"\s*:\s*"([^"]*)"' % field, text)
    if not m or m.end(1) == m.start(1):
        return None
    a, b = m.span(1)
    lps = [lp for (s0, s1, lp) in spans if s0 < b and s1 > a]
    return round(math.exp(sum(lps) / len(lps)), 3) if lps else None


_health = {}


def available(tier: str) -> bool:
    """Is this Bonsai server up? Checked at most every 20 s."""
    now = time.time()
    ok, at = _health.get(tier, (False, 0))
    if now - at < 20:
        return ok
    try:
        with urllib.request.urlopen(URLS[tier] + "/health", timeout=1.5) as r:
            ok = b"ok" in r.read()
    except Exception:
        ok = False
    _health[tier] = (ok, now)
    return ok


def answer(text: str, jpeg: bytes | None = None, tier: str = "9B", l1: str = "en", l2: str = "zh", level: int = 1,
           history: list | None = None, words: list | None = None, on_say=None, on_field=None,
           max_tokens: int = 170, temperature: float | None = None, confidence: bool = False) -> tuple[dict | None, dict]:
    """-> ({say, key_l1, key, q[, gloss, q_gloss]} | None, info).
    on_say(piece, final): the first sentence as soon as it is complete, then the rest of `say`.
    on_field(name, value): every other field as it completes.
    history: [{"role": "user"|"assistant", "content": str}], text only, newest last."""
    lv3 = level >= 3
    names = ("say", "gloss", "key_l1", "key", "q", "q_gloss") if lv3 else ("say", "key_l1", "key", "teach", "q")
    # The system prefix stays byte-identical from turn to turn so the server reuses it; what changes rides at the end.
    if words:
        text = f"(Words I already learned: {', '.join(words[-12:])}.) {text}"
    msgs = [{"role": "system", "content": system_prompt(l1, l2, level, jpeg is not None)}] + list(history or [])
    if jpeg is not None:
        msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(jpeg)}},
                                                  {"type": "text", "text": text}]})
    else:
        msgs.append({"role": "user", "content": text})
    if temperature is None and l1 not in ("en", "zh"):
        temperature = 0.15                              # the weaker languages are safer close to the most likely wording
    extra = {"logprobs": True, "top_logprobs": 1} if confidence else {}
    body = {**COMMON, **extra, **({"temperature": temperature} if temperature is not None else {}), "messages": msgs, "max_tokens": max_tokens + (60 if lv3 else 0), "stream": True,
            "grammar": GBNF3 if lv3 else GBNF, "id_slot": 0, "timings_per_token": True}
    u = urlparse(URLS[tier])
    t0 = time.time()
    buf, tm, seen, marks, first = "", {}, {}, {}, None
    toks = []                                            # (text, log-probability) of every generated token
    try:
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=45)
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            return None, {"error": f"HTTP {resp.status}: {resp.read()[:200]!r}", "ms": 0, "tier": tier}
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            tm = ev.get("timings") or tm
            for ch in ev.get("choices", []):
                buf += ch.get("delta", {}).get("content") or ""
                for t in ((ch.get("logprobs") or {}).get("content") or []):
                    toks.append((t.get("token", ""), t.get("logprob", 0.0)))
            now = round((time.time() - t0) * 1000)
            if "first_token" not in marks and buf:
                marks["first_token"] = now
            if first is None and "say" not in seen:
                piece = _first_sentence(buf)
                if piece:
                    try:
                        first = json.loads('"%s"' % piece)
                    except ValueError:
                        first = None
                    if first:
                        marks["first_sentence"] = now
                        if on_say:
                            on_say(first, False)
            for k, v in _fields(buf, names).items():
                if k in seen:
                    continue
                seen[k] = v
                marks[k] = now
                if k == "say":
                    rest = v[len(first):].strip() if first and v.startswith(first) else (v if not first else "")
                    marks.setdefault("first_sentence", now)
                    if on_say:
                        on_say(rest, True)
                else:
                    if k == "key_l1" and confidence:
                        marks["conf"] = _confidence(toks, "key_l1")
                        if on_field:
                            on_field("conf", marks["conf"])
                    if on_field:
                        on_field(k, v)
        conn.close()
    except (OSError, http.client.HTTPException, ValueError) as e:
        return None, {"error": f"{type(e).__name__}: {e}", "ms": round((time.time() - t0) * 1000), "tier": tier}
    info = {"tier": tier, "image": jpeg is not None, "ms": round((time.time() - t0) * 1000), "marks": marks,
            "prompt_n": tm.get("prompt_n") or 0, "cache_n": tm.get("cache_n") or 0,
            "prompt_ms": round(tm.get("prompt_ms", 0)), "predicted_n": tm.get("predicted_n") or 0,
            "tok_s": round(tm.get("predicted_per_second", 0), 1), "raw": buf, "conf": marks.get("conf")}
    d = _fields(buf, names)
    if not d.get("say"):
        return None, {**info, "error": "the model returned no sentence"}
    for k in names:
        d.setdefault(k, "")
    return d, info


def warmup(tier: str, with_image: bool, l1: str = "en", l2: str = "zh", level: int = 1) -> int:
    """One throw-away call: loads the model (and its vision tower) and caches the system prefix."""
    from PIL import Image
    from .vision import to_jpeg
    t0 = time.time()
    jpeg = to_jpeg(Image.new("RGB", (512, 288), (128, 128, 128))) if with_image else None
    answer("hello", jpeg, tier=tier, l1=l1, l2=l2, level=level, max_tokens=4)
    return round((time.time() - t0) * 1000)


# ---------------------------------------------------------------- two small helper calls
_W = r'[^"\\\n:{}. ] [^"\\\n:{}.]{0,28}'
ROOM_GBNF = ('root ::= "{\\"things\\":[" item ("," item){0,4} "]}"\n'
             'item ::= "{\\"l1\\":\\"" w "\\",\\"l2\\":\\"" w "\\"}"\n' f'w ::= {_W}\n')
PHRASE_GBNF = ('root ::= "{\\"p\\":[" item "," item "]}"\n'
               'item ::= "{\\"l2\\":\\"" w "\\",\\"l1\\":\\"" w "\\"}"\n' f'w ::= {_W}\n')


def _json_call(tier: str, system: str, user, grammar: str, max_tokens: int, timeout: float = 30.0):
    """One short non-streamed call. -> (dict | None, info)."""
    body = {**COMMON, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens, "grammar": grammar,
            "id_slot": 1 if tier == "9B" else 0}       # the 9B keeps slot 0 (and its cached prefix) for the conversation
    t0 = time.time()
    try:
        req = urllib.request.Request(URLS[tier] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
        d = json.loads(out["choices"][0]["message"]["content"])
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}", "ms": round((time.time() - t0) * 1000), "tier": tier}
    tm = out.get("timings") or {}
    return d, {"tier": tier, "ms": round((time.time() - t0) * 1000), "prompt_n": tm.get("prompt_n") or 0,
               "predicted_n": tm.get("predicted_n") or 0, "tok_s": round(tm.get("predicted_per_second", 0), 1)}


def room(jpeg: bytes, l1: str = "en", l2: str = "zh") -> tuple:
    """A first look while the call is ringing: up to five things in view, each with its word in the new language.
    Their audio is made ahead of time, and Olingo can open with one of them. -> ([{l1, l2}], info)"""
    L1, L2 = L.name(l1, "en"), L.name(l2, "en")
    simp = " in simplified characters" if l2 == "zh" else ""
    system = (f"The image is a webcam frame of a person on a video call. List up to five different everyday objects you "
              f"can clearly see (on the desk, in their hands, behind them, what they wear). Never list the person. "
              f"For each: l1 = its common noun written in {L1}, l2 = that word in {L2}{simp}. Reply with JSON only. "
              f'Example: {{"things":[{{"l1":"{CUP.get(l1, "cup")}","l2":"{CUP.get(l2, "cup")}"}}]}}')
    d, info = _json_call("9B", system, [{"type": "image_url", "image_url": {"url": data_url(jpeg)}}], ROOM_GBNF, 110)
    things, seen = [], set()
    for t in (d or {}).get("things", []):
        a, b = str(t.get("l1", "")).strip().lower(), str(t.get("l2", "")).strip()
        if a and b and a not in seen and a != b.lower():
            seen.add(a)
            things.append({"l1": a, "l2": b})
    return things, info


def phrases(key_l1: str, key: str, l1: str = "en", l2: str = "zh", tier: str = "9B") -> tuple:
    """Two very short everyday phrases that use a word just learned. -> ([{l2, l1}], info)"""
    L1, L2 = L.name(l1, "en"), L.name(l2, "en")
    ex = {"zh": '{"p":[{"l2":"吃苹果","l1":"eat an apple"},{"l2":"一个红苹果","l1":"a red apple"}]}',
          "es": '{"p":[{"l2":"comer una manzana","l1":"eat an apple"},{"l2":"una manzana roja","l1":"a red apple"}]}',
          "en": '{"p":[{"l2":"eat an apple","l1":"吃苹果"},{"l2":"a red apple","l1":"一个红苹果"}]}'}
    system = (f"You teach {L2} to a {L1} speaker. For the given word, give two very short, useful everyday {L2} "
              f"phrases that contain it (two to four words each, not full sentences), each with its {L1} meaning. "
              f"Reply with JSON only." + (f" Example for apple: {ex[l2]}" if l2 in ex and (l1 == "en" or l2 == "en") else ""))
    d, info = _json_call(tier, system, f"{key_l1} = {key}", PHRASE_GBNF, 70)
    out = []
    for t in (d or {}).get("p", []):
        a, b = str(t.get("l2", "")).strip(), str(t.get("l1", "")).strip()
        if a and b and a != b and key.lower() in a.lower():
            out.append({"l2": a, "l1": b})
    return out[:2], info


POLISH_GBNF = 'root ::= [^\\n]{1,160}\n'


def polish(text: str, lang: str) -> tuple:
    """A second, text-only pass for the languages Bonsai writes less surely: proofread one sentence.
    Writing a sentence while also looking at a picture and filling a JSON form is harder than correcting it."""
    name = L.name(lang, "en")
    system = (f"You are a careful {name} proofreader. Rewrite the given {name} sentence so that it is natural and "
              f"grammatically correct. {STYLE.get(lang, '')}Keep the meaning, keep it as short, change as little as "
              f"possible. If it is already correct, repeat it unchanged. Reply with the sentence only.")
    body = {**COMMON, "temperature": 0.0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            "max_tokens": 80, "grammar": POLISH_GBNF, "id_slot": 1}
    t0 = time.time()
    try:
        req = urllib.request.Request(URLS["9B"] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            out = json.load(r)
        fixed = out["choices"][0]["message"]["content"].strip().strip('"「」')
    except Exception as e:
        return text, {"error": str(e), "ms": round((time.time() - t0) * 1000)}
    tm = out.get("timings") or {}
    if not fixed or len(fixed) > 2.2 * len(text) + 8:
        fixed = text
    return fixed, {"ms": round((time.time() - t0) * 1000), "prompt_n": tm.get("prompt_n") or 0,
                   "predicted_n": tm.get("predicted_n") or 0}


def same_question(new: str, earlier: list) -> tuple:
    """Does the new question ask for what an earlier one already asked, even in other words? -> (bool, info).
    A text-only yes/no on the 9B's helper slot, while the answer before it is being spoken."""
    if not earlier:
        return False, {"ms": 0}
    system = ("You compare questions from one conversation. Answer yes if the NEW question asks for the same "
              "information as one of the EARLIER questions, even when it uses other words. Answer no if it asks "
              "about something else, or about another detail of the same thing.")
    user = "EARLIER:\n" + "\n".join(f"- {q}" for q in earlier) + f"\nNEW: {new}\nSame as an earlier one?"
    body = {**COMMON, "temperature": 0.0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 2, "grammar": 'root ::= "yes" | "no"\n', "id_slot": 1}
    t0 = time.time()
    try:
        req = urllib.request.Request(URLS["9B"] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as r:
            out = json.load(r)
        ans = out["choices"][0]["message"]["content"].strip().lower()
    except Exception as e:
        return False, {"error": str(e), "ms": round((time.time() - t0) * 1000)}
    tm = out.get("timings") or {}
    return ans.startswith("yes"), {"ms": round((time.time() - t0) * 1000), "prompt_n": tm.get("prompt_n") or 0,
                                   "predicted_n": tm.get("predicted_n") or 0}
