"""Languages, fixed lines, the opening-question parser and the trusted word list.
L1 = the learner's own language (what Olingo speaks in), L2 = the language being learned.
Fixed lines exist for L1 = English and L1 = Chinese. L2 can be any language the Argmax TTS speaks."""
import re
import unicodedata

# code -> names in each supported L1, and whether it is written without spaces
LANGS = {
    "en": {"en": "English", "zh": "英语", "fr": "anglais", "ja": "英語", "cjk": False},
    "zh": {"en": "Chinese", "zh": "中文", "fr": "chinois", "ja": "中国語", "cjk": True},
    "es": {"en": "Spanish", "zh": "西班牙语", "fr": "espagnol", "ja": "スペイン語", "cjk": False},
    "fr": {"en": "French", "zh": "法语", "fr": "français", "ja": "フランス語", "cjk": False},
    "ja": {"en": "Japanese", "zh": "日语", "fr": "japonais", "ja": "日本語", "cjk": True},
    "ko": {"en": "Korean", "zh": "韩语", "fr": "coréen", "ja": "韓国語", "cjk": True},
    "de": {"en": "German", "zh": "德语", "fr": "allemand", "ja": "ドイツ語", "cjk": False},
    "it": {"en": "Italian", "zh": "意大利语", "fr": "italien", "ja": "イタリア語", "cjk": False},
    "pt": {"en": "Portuguese", "zh": "葡萄牙语", "fr": "portugais", "ja": "ポルトガル語", "cjk": False},
}
L1S = ("en", "zh", "fr", "ja")      # languages Olingo itself can speak to the learner in

TARGET_WORDS = {
    "zh": r"chinese|mandarin|中文|汉语|漢語|普通话|普通話|国语|國語|华语",
    "es": r"spanish|espanol|español|西班牙",
    "en": r"english|英语|英語|英文",
    "fr": r"french|francais|français|法语|法語|法文",
    "ja": r"japanese|日语|日語|日文",
    "ko": r"korean|韩语|韓語|韩文|韓文",
    "de": r"german|deutsch|德语|德語|德文",
    "it": r"italian|italiano|意大利",
    "pt": r"portuguese|portugues|葡萄牙",
}
LEVEL_WORDS = [
    (3, r"\badvanced\b|\bfluent\b|\bvery good\b|\bpretty good\b|高级|高級|流利|很好"),
    (2, r"\bintermediate\b|\bin the middle\b|\bsome\b|\ba little\b|\ba bit\b|\bmedium\b|\bokay\b|中级|中級|一点|一點|一些|还行|還行"),
    (1, r"\bbeginner\b|\bbeginning\b|\bbegin\b|\bnever\b|\bzero\b|\bbasic\b|\bnothing\b|\bfrom scratch\b|\bnew\b|"
        r"初学|初學|初级|初級|零基础|零基礎|入门|入門|新手|不会|不會"),
]
LEVEL_NAMES = {1: "beginner", 2: "intermediate", 3: "advanced"}


def parse_setup(text: str) -> dict:
    """What the learner answered to "What language do you want to learn, and what level are you?"
    -> {"l2": code | None, "level": 1 | 2 | 3 | None}. Rules only, no model."""
    t = unicodedata.normalize("NFKC", text.lower())
    found = [(m.start(), code) for code, pat in TARGET_WORDS.items() for m in [re.search(pat, t)] if m]
    l2 = min(found)[1] if found else None            # the first language mentioned
    level = next((lv for lv, pat in LEVEL_WORDS if re.search(pat, t)), None)
    return {"l2": l2, "level": level}


def name(code: str, l1: str) -> str:
    return LANGS.get(code, {}).get(l1) or LANGS.get(code, {}).get("en") or code


# Every line Olingo says without a model. {L2} = the name of the language being learned, {thing} = a word.
FIXED = {
    "en": {
        "ask_name": "What's your name?",
        "nice_name": "Nice to meet you, {thing}!",
        "nice": "Nice to meet you!",
        "see_room": "I can see your {thing}. Want to learn that word, or show me something else?",
        "see_more": "I can also see your {thing}. Want to learn that word?",
        "see_new": "Now I can see your {thing}. Want to learn that word?",
        "explore": "I think we have covered what I can see. Show me around your room!",
        "in_l2": "In {L2}:",
        "good": "Perfect!",
        "ok": "Good!",
        "almost": "Almost. Listen again:",
        "slow_again": "Once more, slowly:",
        "try": "Nice try. Let's keep going.",
        "phrases": "Now, how it is used.",
        "offer": "Want to learn how to say it in {L2}?",
        "oh": "Oh!",
        "nevermind": "Never mind, go on.",
        "pick_none": "I can't find anything new. Show me something.",
        "no_problem": "No problem.",
        "offline": "I'm offline, so I can't know that. You tell me!",
        "again_none": "I have not said anything yet.",
        "sorry": "Sorry, I didn't catch that. Can you say it again?",
        "bye": "Goodbye! See you next time.",
        "more": "Okay, more {L2}.",
        "less": "Okay, less {L2}.",
    },
    "zh": {
        "ask_name": "你叫什么名字？",
        "nice_name": "很高兴认识你，{thing}！",
        "nice": "很高兴认识你！",
        "see_room": "我看到了你的{thing}。想学这个词，还是给我看别的东西？",
        "see_more": "我还看到了你的{thing}。想学这个词吗？",
        "see_new": "现在我看到了你的{thing}。想学这个词吗？",
        "explore": "我能看到的东西我们都聊过了。带我看看你的房间吧！",
        "in_l2": "用{L2}说：",
        "good": "很好！",
        "ok": "不错！",
        "almost": "差一点，再听一遍：",
        "slow_again": "再来一次，慢一点：",
        "try": "没关系，我们继续。",
        "phrases": "再看看它怎么用。",
        "offer": "想学它用{L2}怎么说吗？",
        "oh": "哦！",
        "nevermind": "没事，你继续。",
        "pick_none": "我没看到新的东西，给我看一样吧。",
        "no_problem": "没问题。",
        "offline": "我没有联网，所以不知道。你来告诉我吧！",
        "again_none": "我还没说话呢。",
        "sorry": "不好意思，我没听清，能再说一遍吗？",
        "bye": "再见！下次见。",
        "more": "好的，多说一点{L2}。",
        "less": "好的，少说一点{L2}。",
    },
    "fr": {
        "ask_name": "Comment tu t'appelles ?",
        "nice_name": "Enchanté, {thing} !",
        "nice": "Enchanté !",
        "see_room": "Je vois quelque chose : {thing}. Tu veux apprendre ce mot, ou me montrer autre chose ?",
        "see_more": "Je vois aussi autre chose : {thing}. Tu veux apprendre ce mot ?",
        "see_new": "Maintenant je vois autre chose : {thing}. Tu veux apprendre ce mot ?",
        "explore": "On a parlé de tout ce que je vois. Fais-moi visiter ta pièce !",
        "in_l2": "En {L2} :",
        "good": "Parfait !",
        "ok": "Bien !",
        "almost": "Presque. Écoute encore une fois :",
        "slow_again": "Encore une fois, lentement :",
        "try": "Bien essayé. On continue.",
        "phrases": "Voyons comment on l'utilise.",
        "offer": "Tu veux apprendre à le dire en {L2} ?",
        "oh": "Oh !",
        "nevermind": "Ce n'est rien, continue.",
        "pick_none": "Je ne vois rien de nouveau. Montre-moi quelque chose.",
        "no_problem": "Pas de problème.",
        "offline": "Je suis hors ligne, donc je ne peux pas le savoir. Dis-moi, toi !",
        "again_none": "Je n'ai encore rien dit.",
        "sorry": "Pardon, je n'ai pas compris. Tu peux répéter ?",
        "bye": "Au revoir ! À la prochaine.",
        "more": "D'accord, plus de {L2}.",
        "less": "D'accord, moins de {L2}.",
    },
    "ja": {
        "ask_name": "お名前は何ですか？",
        "nice_name": "{thing}さん、よろしくお願いします！",
        "nice": "よろしくお願いします！",
        "see_room": "{thing}が見えますね。この言葉を覚えてみますか？それとも、ほかのものを見せてくれますか？",
        "see_more": "{thing}も見えますね。この言葉を覚えてみますか？",
        "see_new": "今度は{thing}が見えます。この言葉を覚えてみますか？",
        "explore": "見えるものは全部話しましたね。お部屋を見せてください！",
        "in_l2": "{L2}では：",
        "good": "完璧です！",
        "ok": "いいですね！",
        "almost": "惜しいです。もう一度聞いてください。",
        "slow_again": "もう一度、ゆっくり言いますね。",
        "try": "よく頑張りました。続けましょう。",
        "phrases": "使い方も見てみましょう。",
        "offer": "{L2}で何と言うか、覚えてみますか？",
        "oh": "あ！",
        "nevermind": "なんでもないです。続けてください。",
        "pick_none": "新しいものが見えません。何か見せてください。",
        "no_problem": "大丈夫です。",
        "offline": "オフラインなので、わかりません。教えてください。",
        "again_none": "まだ何も言っていません。",
        "sorry": "すみません、聞き取れませんでした。もう一度お願いします。",
        "bye": "さようなら！また今度お話ししましょう。",
        "more": "わかりました。{L2}を増やします。",
        "less": "わかりました。{L2}を減らします。",
    },
}
# Lines with several wordings, so two calls do not open and move on the same way. Picked in rotation.
VARIED = {
    "en": {
        "hello": ["Hi! I'm Olingo.", "Hey there! Olingo here.", "Hello! I'm Olingo."],
        "warm": ["How is your day going?", "How are you feeling today?", "What have you been up to today?"],
        "invite": ["Can you say it?", "Try it.", "Your turn.", "Give it a try."],
        "nudge": ["Take your time.", "No rush.", "I'm listening whenever you're ready.", "It's okay, think it over."],
        "open": ["Show me something, or ask me anything.", "What do you want to show me?",
                 "Hold something up, or just ask me."],
        "next": ["Nice. Show me something else, or say: you pick.", "Good. What else is around you? Or say: you pick.",
                 "Great. Show me another thing, or say: you pick."],
    },
    "zh": {
        "hello": ["你好！我是 Olingo。", "嗨！我是 Olingo。"],
        "warm": ["你今天过得怎么样？", "你今天心情怎么样？"],
        "invite": ["你来说一遍？", "试试看。", "你也说说看。"],
        "nudge": ["不着急，慢慢来。", "慢慢想，我在听。", "没关系，想好了再说。"],
        "open": ["给我看一样东西，或者问我问题。", "你想给我看什么？"],
        "next": ["不错。再给我看一样东西，或者说：你来选。", "很好。你周围还有什么？或者说：你来选。"],
    },
    "fr": {
        "hello": ["Bonjour ! Je suis Olingo.", "Salut ! Moi, c'est Olingo."],
        "warm": ["Comment se passe ta journée ?", "Comment tu te sens aujourd'hui ?"],
        "invite": ["Tu peux le dire ?", "Essaie.", "À toi."],
        "nudge": ["Prends ton temps.", "Pas de souci, je t'écoute.", "Rien ne presse."],
        "open": ["Montre-moi quelque chose, ou pose-moi une question.", "Qu'est-ce que tu veux me montrer ?"],
        "next": ["Très bien. Montre-moi autre chose, ou dis : tu choisis.",
                 "Bien. Qu'est-ce qu'il y a autour de toi ? Ou dis : tu choisis."],
    },
    "ja": {
        "hello": ["こんにちは！オリンゴです。", "こんにちは！オリンゴといいます。"],
        "warm": ["今日はどんな一日ですか？", "今日の調子はどうですか？"],
        "invite": ["言ってみてください。", "どうぞ。"],
        "nudge": ["ゆっくりでいいですよ。", "急がなくて大丈夫です。"],
        "open": ["何か見せてください。質問でもいいですよ。", "何を見せてくれますか？"],
        "next": ["いいですね。ほかのものを見せてください。または「選んで」と言ってください。",
                 "よくできました。周りに何がありますか？または「選んで」と言ってください。"],
    },
}
# Japanese is the language Bonsai writes least surely, so two things are not left to it: the sentence that names an
# object, and the question that follows. Both come from here and are always correct.
NAMES_IT = {"ja": "それは{thing}ですね。"}
SAFE_QUESTIONS = {"ja": ["それは好きですか？", "よく使いますか？", "それは何色ですか？", "どこで買いましたか？", "毎日使いますか？"]}
# What Olingo says the moment the learner stops, while it thinks. Plain words, no "oh" or "hmm".
FILLERS = {"en": ["Let me see.", "Okay.", "Let me think.", "One moment.", "Hmm, let me look.", "Right.", "Good question.",
                  "Let me take a look.", "I see.", "Alright."],
           "zh": ["我看看。", "好的。", "让我想想。", "稍等一下。", "我来看一下。", "明白。", "好问题。", "这样啊。", "行。"],
           "fr": ["Voyons.", "D'accord.", "Laisse-moi voir.", "Un instant.", "Je regarde.", "Bien.", "Bonne question."],
           "ja": ["そうですね。", "はい。", "見てみますね。", "少々お待ちください。", "なるほど。", "いい質問ですね。"]}
# When Bonsai's own teaching sentence cannot be used, one of these is. {k} = the word in the learner's language,
# {w} = the word in the new one. "teach" ends by handing the turn over; "mention" does not.
TEACH = {
    "en": {"teach": ["In {L2}, {k} is {w}. Can you say it?", "The {L2} word for {k} is {w}. Try it.",
                     "{k} in {L2} is {w}. Your turn.", "We say {w} for {k} in {L2}. Give it a try."],
           "mention": ["In {L2}, that is {w}.", "{k} is {w} in {L2}.", "By the way, {k} in {L2} is {w}."]},
    "zh": {"teach": ["{k}的{L2}是 {w}。你来说一遍？", "{k}用{L2}说是 {w}。试试看。", "{L2}里，{k}叫 {w}。你也说说看。"],
           "mention": ["{k}的{L2}是 {w}。", "顺便说一下，{k}用{L2}说是 {w}。"]},
    "fr": {"teach": ["En {L2}, {k} se dit {w}. Tu peux le dire ?", "Le mot {L2} pour {k}, c'est {w}. Essaie."],
           "mention": ["En {L2}, {k} se dit {w}."]},
    "ja": {"teach": ["{k}は{L2}で {w} と言います。言ってみてください。", "{L2}では、{k}は {w} です。どうぞ。"],
           "mention": ["{k}は{L2}で {w} と言います。"]},
}


def teach_line(kind: str, l1: str, l2: str, k: str, w: str, i: int) -> str:
    opts = TEACH.get(l1, TEACH["en"])[kind]
    t = opts[i % len(opts)].format(L2=name(l2, l1 if l1 in L1S else "en"), k=k, w=w)
    return t[:1].upper() + t[1:]


# The same offer in several wordings, so it does not sound like a recording.
OFFERS = {
    "en": {"see_room": ["I can see your {thing}. Want to learn that word, or show me something else?",
                        "I notice your {thing}. Shall we start with that word, or do you have something else?",
                        "I spotted your {thing}. Want to learn how to say it, or show me another thing?"],
           "see_more": ["I can also see your {thing}. Want to learn that word?", "What about your {thing}? Want to learn it?",
                        "I also noticed your {thing}. Shall we learn that one?"],
           "see_new": ["Now I can see your {thing}. Want to learn that word?", "Your {thing} just caught my eye. Shall we learn it?"]},
    "zh": {"see_room": ["我看到了你的{thing}。想学这个词，还是给我看别的东西？", "我注意到你的{thing}。从这个词开始，还是你有别的想给我看？"],
           "see_more": ["我还看到了你的{thing}。想学这个词吗？", "你的{thing}呢？想学怎么说吗？"],
           "see_new": ["现在我看到了你的{thing}。想学这个词吗？", "这里有{thing}。我们学这个词好吗？"]},
}


# A phrase is explained before the learner is asked to say it. {g} = what it means, {p} = the phrase.
EXPLAIN = {
    "en": ['"{g}" is {p}.', 'To say "{g}", you say {p}.', '{p} means "{g}".'],
    "zh": ["“{g}”是 {p}。", "“{g}”可以说 {p}。", "{p} 的意思是“{g}”。"],
    "fr": ["« {g} » se dit {p}.", "{p} veut dire « {g} »."],
    "ja": ["「{g}」は {p} です。", "{p} は「{g}」という意味です。"],
}


def explain_line(l1: str, meaning: str, phrase: str, i: int) -> str:
    opts = EXPLAIN.get(l1, EXPLAIN["en"])
    return opts[i % len(opts)].format(g=meaning, p=phrase)


def offer_line(key: str, l1: str, l2: str, thing: str, i: int) -> str:
    opts = OFFERS.get(l1, {}).get(key)
    return opts[i % len(opts)].format(thing=thing) if opts else line(key, l1, l2, thing)

# One voice per call, chosen by the learner's language (measured: Aiden speaks English at 3.1 words/s and says
# Chinese words correctly; Serena is a native Mandarin voice).
VOICES = {"en": "aiden", "zh": "serena", "fr": "aiden", "ja": "ono-anna"}
VOICE_CHOICES = ["aiden", "serena", "vivian", "eric", "ono-anna"]
VOICE_LABELS = {"aiden": "Aiden", "serena": "Serena", "vivian": "Vivian", "eric": "Eric", "ono-anna": "Anna"}


HELLO = {"zh": "你好", "es": "hola", "fr": "bonjour", "ja": "こんにちは", "ko": "안녕하세요", "de": "Hallo",
         "it": "ciao", "pt": "olá", "en": "hello"}


def preview_parts(voice: str, l1: str, l2: str) -> list:
    """What is played when the learner tries a voice: [(text, lang)]. The tutor voice introduces itself, then the
    word of the new language is said the way it will be in the call, by a native voice."""
    who = VOICE_LABELS.get(voice, voice).split(" ")[0]
    lead = {"zh": f"你好，我是 {who}。{name(l2, 'zh')}的“你好”是：", "fr": f"Bonjour, je suis {who}. En {name(l2, 'fr')}, bonjour se dit :",
            "ja": f"こんにちは、{who}です。{name(l2, 'ja')}の「こんにちは」は："}.get(l1, f"Hi, I'm {who}. In {name(l2, 'en')}, hello is:")
    return [(lead, l1), (HELLO.get(l2, "hello"), l2)]


def preview_line(voice: str, l1: str, l2: str) -> str:
    """The same sample as one string (for display)."""
    who = VOICE_LABELS.get(voice, voice).split(" ")[0]
    if l1 == "zh":
        return f"你好，我是 {who}。{name(l2, 'zh')}的“你好”是 {HELLO.get(l2, 'hello')}。"
    if l1 == "fr":
        return f"Bonjour, je suis {who}. En {name(l2, 'fr')}, bonjour se dit {HELLO.get(l2, 'hello')}."
    if l1 == "ja":
        return f"こんにちは、{who}です。{name(l2, 'ja')}の「こんにちは」は {HELLO.get(l2, 'hello')} です。"
    return f"Hi, I'm {who}. In {name(l2, 'en')}, hello is {HELLO.get(l2, 'hello')}."


# One voice for the whole call: the same person speaks the learner's language and the new one. Which voice: the
# one that is native in the language being learned, because its sounds and tones are what the learner copies.
NATIVE_VOICE = {"zh": "vivian", "en": "aiden", "ja": "ono-anna"}
NATIVE_IN = {"aiden": "en", "vivian": "zh", "serena": "zh", "eric": "zh", "ono-anna": "ja"}
# Measured on 24 Chinese words that differ from another word only by tone, read aloud and written down by Whisper:
# Vivian 20 right, Serena 15, Eric 11, Aiden 10. Speaking English: Aiden 3.0 words a second, Serena 2.8, Vivian 2.4.


def voice_for_text(lang: str, tutor: str) -> str:
    """Every piece of text is said by the call's one voice."""
    return tutor


def voice_for(l1: str, chosen: str | None = None, l2: str | None = None) -> str:
    if chosen in VOICE_CHOICES:
        return chosen
    return NATIVE_VOICE.get(l2) or NATIVE_VOICE.get(l1) or VOICES.get(l1, "aiden")


def parse_name(text: str, l1: str) -> str:
    """The learner's name out of their answer to "What's your name?". "" when it is not clear."""
    t = text.strip().strip(".!?。！？ ")
    pats = {"en": r"(?:my name is|my name's|i am|i'm|it's|this is|call me|name is)\s+([A-Za-z][\w'-]{1,15})",
            "fr": r"(?:je m'appelle|moi c'est|je suis|mon nom est|c'est)\s+([A-Za-zÀ-ÿ][\w'À-ÿ-]{1,15})",
            "zh": r"(?:我叫|我是|叫我|我的名字是|我的名字叫)\s*([\u4e00-\u9fffA-Za-z]{1,6})",
            "ja": r"(?:私は|わたしは|名前は)?\s*([\u3040-\u30ff\u4e00-\u9fffA-Za-z]{1,8}?)(?:です|といいます|と申します|だよ|だ)"}
    m = re.search(pats.get(l1, pats["en"]), t, flags=re.I)
    if m:
        name = m.group(1)
    elif l1 in ("en", "fr") and 1 <= len(t.split()) <= 2 and re.fullmatch(r"[A-Za-zÀ-ÿ' -]{2,20}", t):
        name = t.split()[0]
    elif l1 in ("zh", "ja") and 1 <= len(t) <= 4 and not re.search(r"[，。？！、]", t):
        name = t
    else:
        return ""
    return name[:1].upper() + name[1:] if name.isascii() else name


def line(key: str, l1: str, l2: str, thing: str = "") -> str:
    table = FIXED.get(l1, FIXED["en"])
    return (table.get(key) or FIXED["en"][key]).format(L2=name(l2, l1 if l1 in L1S else "en"), thing=thing)


def varied(key: str, l1: str, i: int) -> str:
    opts = VARIED.get(l1, VARIED["en"])[key]
    return opts[i % len(opts)]


def fixed_texts(l1: str, l2: str) -> list:
    """Every (text, lang) that can be made ahead of time for this pair: fixed lines, their variants and fillers."""
    out = [(f, l1) for f in FILLERS.get(l1, FILLERS["en"])]
    out += [(line(k, l1, l2), l1) for k in FIXED.get(l1, FIXED["en"]) if "{thing}" not in FIXED["en"][k]]
    out += [(t, l1) for opts in VARIED.get(l1, VARIED["en"]).values() for t in opts if "{" not in t]
    return out


# English -> Chinese for things people actually hold up to a webcam. When a word is here, it beats the model.
EN_ZH = {
    "phone": "手机", "cell phone": "手机", "smartphone": "手机", "iphone": "手机", "keys": "钥匙", "key": "钥匙",
    "wallet": "钱包", "bottle": "瓶子", "water bottle": "水瓶", "water": "水", "pen": "笔", "pencil": "铅笔",
    "laptop": "笔记本电脑", "computer": "电脑", "glasses": "眼镜", "sunglasses": "太阳镜", "watch": "手表",
    "cup": "杯子", "mug": "杯子", "coffee": "咖啡", "tea": "茶", "notebook": "笔记本", "book": "书",
    "headphones": "耳机", "earphones": "耳机", "earbuds": "耳机", "airpods": "耳机", "charger": "充电器",
    "cable": "线", "mouse": "鼠标", "keyboard": "键盘", "backpack": "背包", "bag": "包", "hat": "帽子", "cap": "帽子",
    "mask": "口罩", "coin": "硬币", "card": "卡", "credit card": "信用卡", "marker": "马克笔", "sticker": "贴纸",
    "can": "罐子", "snack": "零食", "banana": "香蕉", "apple": "苹果", "orange": "橙子", "tangerine": "橘子",
    "ball": "球", "shirt": "衬衫", "t-shirt": "T恤", "hoodie": "卫衣", "jacket": "外套", "sweater": "毛衣",
    "shoe": "鞋", "shoes": "鞋", "sock": "袜子", "scarf": "围巾", "ring": "戒指", "necklace": "项链",
    "hand": "手", "finger": "手指", "face": "脸", "hair": "头发", "eye": "眼睛", "chair": "椅子", "table": "桌子",
    "desk": "桌子", "door": "门", "window": "窗户", "wall": "墙", "light": "灯", "lamp": "灯", "plant": "植物",
    "flower": "花", "paper": "纸", "tissue": "纸巾", "napkin": "餐巾纸", "fork": "叉子", "spoon": "勺子",
    "knife": "刀", "chopsticks": "筷子", "plate": "盘子", "bowl": "碗", "bread": "面包", "egg": "鸡蛋",
    "rice": "米饭", "noodles": "面条", "milk": "牛奶", "juice": "果汁", "candy": "糖果", "chocolate": "巧克力",
    "cookie": "饼干", "pizza": "披萨", "umbrella": "雨伞", "toy": "玩具", "teddy bear": "泰迪熊", "stapler": "订书机",
    "scissors": "剪刀", "tape": "胶带", "remote": "遥控器", "camera": "相机", "badge": "工牌", "lanyard": "挂绳",
    "money": "钱", "cat": "猫", "dog": "狗", "tired": "累", "hungry": "饿", "happy": "开心", "weather": "天气",
    "friend": "朋友", "school": "学校", "lunch": "午饭", "dinner": "晚饭", "breakfast": "早饭", "music": "音乐",
    "red": "红色", "orange color": "橙色", "yellow": "黄色", "green": "绿色", "blue": "蓝色", "purple": "紫色",
    "pink": "粉色", "brown": "棕色", "black": "黑色", "white": "白色", "gray": "灰色", "grey": "灰色",
}


def trusted(key_l1: str, l1: str, l2: str) -> str | None:
    """A vetted translation of an L1 word, if we have one."""
    k = key_l1.strip().lower()
    if l1 == "en" and l2 == "zh":
        return EN_ZH.get(k) or EN_ZH.get(re.sub(r"(es|s)$", "", k))
    if l1 == "en" and l2 == "es":
        try:
            from . import vocab
            v = vocab.lookup(k)
            return v["es"] if v else None
        except Exception:
            return None
    return None
