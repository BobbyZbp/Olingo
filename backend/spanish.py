"""Rule-based Spanish (A1): articles, plurals, colour agreement, phrase building, answer checking, word alignment.
Grammar is guaranteed by code, never by the model."""
import re
import unicodedata

VOWELS = "aeiouáéíóú"


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize(s: str) -> str:
    s = strip_accents(s.lower())
    s = re.sub(r"[^a-z0-9ñ\s]", " ", s.replace("ñ", "\x00")).replace("\x00", "ñ")
    return re.sub(r"\s+", " ", s).strip()


def words_of(s: str) -> list:
    return normalize(s).split()


# ---------- plurals ----------
_ACCENT_DROP = {"ón": "on", "án": "an", "én": "en", "ín": "in", "ún": "un",
                "ás": "as", "és": "es", "ís": "is", "ós": "os", "ús": "us"}


def plural_word(w: str) -> str:
    w = w.strip()
    if not w:
        return w
    for suf, rep in _ACCENT_DROP.items():
        if w.endswith(suf):
            return w[: -len(suf)] + rep + "es"
    if w.endswith("z"):
        return w[:-1] + "ces"
    if w.endswith("s"):
        if len(re.findall(r"[aeiouáéíóú]+", w)) == 1:
            return w + "es"                       # monosyllables: gris -> grises, mes -> meses
        return w                                  # unstressed -s: paraguas, lunes, gafas
    if w[-1] in VOWELS and w[-1] not in "íú":
        return w + "s"
    if w[-1] in "íú":
        return w + "es"                           # rubí -> rubíes
    return w + "es"


def plural_noun(noun: str, override: str | None = None) -> str:
    """Multi-word nouns: 'botella de agua' -> 'botellas de agua'. Use `override` for noun+adjective."""
    if override:
        return override
    parts = noun.split(" ")
    parts[0] = plural_word(parts[0])
    return " ".join(parts)


# ---------- articles ----------
def indefinite(g: str, n: int) -> str:
    return {("m", 1): "un", ("f", 1): "una", ("m", 2): "unos", ("f", 2): "unas"}[(g, 1 if n <= 1 else 2)]


def definite(g: str, n: int) -> str:
    return {("m", 1): "el", ("f", 1): "la", ("m", 2): "los", ("f", 2): "las"}[(g, 1 if n <= 1 else 2)]


# ---------- colours ----------
# english -> (masculine singular, feminine singular, kind)   kind: "oa" agrees in gender; "inv" = same for both
COLORS = {
    "red": ("rojo", "roja", "oa"), "orange": ("naranja", "naranja", "inv"),
    "yellow": ("amarillo", "amarilla", "oa"), "green": ("verde", "verde", "inv"),
    "blue": ("azul", "azul", "inv"), "purple": ("morado", "morada", "oa"),
    "pink": ("rosa", "rosa", "inv"), "brown": ("marrón", "marrón", "inv"),
    "black": ("negro", "negra", "oa"), "white": ("blanco", "blanca", "oa"),
    "gray": ("gris", "gris", "inv"), "silver": ("plateado", "plateada", "oa"),
    "gold": ("dorado", "dorada", "oa"),
}
COLOR_EN_ALIASES = {"grey": "gray", "violet": "purple", "beige": "brown", "tan": "brown", "golden": "gold"}


def color_es(color: str, g: str, n: int = 1) -> str:
    color = COLOR_EN_ALIASES.get(color, color)
    if color not in COLORS:
        return ""
    m, f, _ = COLORS[color]
    base = f if g == "f" else m
    return plural_word(base) if n > 1 else base


def color_forms(color: str) -> set:
    """Every accepted spoken form of a colour (normalized), for checking an answer."""
    color = COLOR_EN_ALIASES.get(color, color)
    if color not in COLORS:
        return set()
    m, f, _ = COLORS[color]
    return {normalize(x) for x in (m, f, plural_word(m), plural_word(f))}


# ---------- phrase building ----------
def noun_form(noun: str, n: int, pl_override: str | None = None, plural_only: bool = False) -> str:
    return noun if plural_only else (plural_noun(noun, pl_override) if n > 1 else noun)


def build_phrase(noun: str, g: str, n: int = 1, color: str = "none", pl_override: str | None = None,
                 plural_only: bool = False) -> str:
    """'una taza azul', 'unas llaves negras', 'unas gafas negras'."""
    n_eff = 2 if plural_only else n
    parts = [indefinite(g, n_eff), noun_form(noun, n, pl_override, plural_only)]
    c = color_es(color, g, n_eff) if color and color != "none" else ""
    if c and normalize(c) == normalize(parts[1].split(" ")[0]):
        c = ""                                    # "una naranja naranja": the noun already says the colour
    if c:
        parts.append(c)
    return " ".join(parts)


def build_description(phrase: str) -> str:
    return f"Tienes {phrase} en la mano."


def deck_word(noun: str, g: str, plural_only: bool = False, pl_override: str | None = None) -> str:
    """Dictionary form shown in 'my words': 'la taza', 'las gafas'."""
    if plural_only:
        return f"{definite(g, 2)} {noun}"
    return f"{definite(g, 1)} {noun}"


# ---------- numbers ----------
NUMBER_WORDS = {1: ("uno", "una", "un"), 2: ("dos",), 3: ("tres",), 4: ("cuatro",), 5: ("cinco",)}


def number_forms(n: int) -> set:
    return {normalize(x) for x in NUMBER_WORDS.get(n, ())} | {str(n)}


def number_word(n: int, g: str = "m") -> str:
    if n == 1:
        return "una" if g == "f" else "uno"
    return NUMBER_WORDS[n][0]


# ---------- answer checking ----------
def has_color(transcript: str, color: str) -> bool:
    forms = color_forms(color)
    return bool(forms & set(words_of(transcript)))


def yes_no(transcript: str):
    """'yes' / 'no' / None. First of the two words wins ('no, es azul' -> no)."""
    for w in words_of(transcript):
        if w == "si":
            return "yes"
        if w == "no":
            return "no"
    return None


def has_number(transcript: str, n: int) -> bool:
    return bool(number_forms(n) & set(words_of(transcript)))


# ---------- word alignment / clarity ----------
def edit_distance_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def word_match(target: str, heard: str) -> bool:
    if target == heard:
        return True
    return len(target) >= 4 and edit_distance_le1(target, heard)


def align(target_phrase: str, heard_words: list) -> list:
    """target words -> [{'w': word, 'p': clarity 0..1}]. heard_words: [{'w': str, 'p': float}] from Whisper.
    Unmatched target words get p = 0. Matching is sequential and accent/punctuation-insensitive."""
    heard = [(normalize(h["w"]), float(h.get("p", 0.0))) for h in heard_words]
    heard = [(w, p) for w, p in heard if w]
    out, pos = [], 0
    for tw_raw in target_phrase.split():
        tw = normalize(tw_raw)
        found = None
        for j in range(pos, len(heard)):
            if word_match(tw, heard[j][0]):
                found = j
                break
        if found is None:
            out.append({"w": tw_raw, "p": 0.0})
        else:
            out.append({"w": tw_raw, "p": round(heard[found][1], 2)})
            pos = found + 1
    return out


def clarity_color(p: float) -> str:
    return "green" if p >= 0.75 else "yellow" if p >= 0.5 else "red"


def noun_passed(target_phrase: str, scored: list, noun: str, thr: float = 0.6) -> bool:
    nw = normalize(noun.split(" ")[0])
    for item in scored:
        tw = normalize(item["w"])
        if tw == nw or tw == normalize(plural_word(noun.split(" ")[0])):
            return item["p"] >= thr
    return all(s["p"] >= thr for s in scored)
