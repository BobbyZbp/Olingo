"""All prompts and JSON schemas. Fixed text goes in the system message, the variable part (image / object) comes last,
so the server can reuse the fixed prefix. System messages stay under ~60 tokens."""

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
          "black", "white", "gray", "silver", "gold", "none"]

SEE_SYSTEM = ('You name the object a person holds up to a webcam. Reply with JSON only. '
              'en: common English noun, singular, or "none" if nothing is held. '
              'es: the Spanish noun, singular, no article. g: its gender. n: how many. color: main color.')

SEE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["en", "es", "g", "n", "color"],
    "properties": {
        "en": {"type": "string", "maxLength": 24},
        "es": {"type": "string", "maxLength": 24},
        "g": {"enum": ["m", "f"]},
        "n": {"enum": [1, 2, 3, 4, 5]},
        "color": {"enum": COLORS},
    },
}

TEACH_SYSTEM = (
    'You teach Spanish to beginners. Reply with JSON only. s: ONE simple Spanish sentence, at most 8 words, that uses the '
    'given noun. s_en: its English translation. rel: two OTHER Spanish words related to the object, with English. '
    'Example for "la llave": {"s":"Abro la puerta con la llave.","s_en":"I open the door with the key.",'
    '"rel":[{"es":"abrir","en":"to open"},{"es":"la puerta","en":"the door"}]}')

TEACH_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["s", "s_en", "rel"],
    "properties": {
        "s": {"type": "string", "maxLength": 60},
        "s_en": {"type": "string", "maxLength": 60},
        "rel": {"type": "array", "minItems": 2, "maxItems": 2, "items": {
            "type": "object", "additionalProperties": False, "required": ["es", "en"],
            "properties": {"es": {"type": "string", "maxLength": 20}, "en": {"type": "string", "maxLength": 20}}}},
    },
}

COMMON_PARAMS = {
    "cache_prompt": True,
    "chat_template_kwargs": {"enable_thinking": False},
    "temperature": 0.2, "top_p": 0.8, "top_k": 20, "min_p": 0,
    "presence_penalty": 0, "repeat_penalty": 1.0,
}

# Compact, always-valid output (no newlines/indent: 28 tokens instead of 46) with enums enforced.
# NOTE: {"type":"json_schema","schema":...} is silently IGNORED by this llama-server build; only the OpenAI-style
# nested json_schema, a top-level json_schema, or a GBNF grammar are enforced.
_COLOR_ALT = " | ".join('"%s"' % c for c in COLORS)
SEE_GBNF = (
    'root ::= "{\\"en\\":\\"" w "\\",\\"es\\":\\"" w "\\",\\"g\\":\\"" ("m" | "f") "\\",\\"n\\":" [1-5] '
    '",\\"color\\":\\"" (' + _COLOR_ALT + ') "\\"}"\n'
    'w ::= [a-zñáéíóú -]{1,24}\n')
TEACH_GBNF = (
    'root ::= "{\\"s\\":\\"" t "\\",\\"s_en\\":\\"" t "\\",\\"rel\\":[{\\"es\\":\\"" w "\\",\\"en\\":\\"" w "\\"},'
    '{\\"es\\":\\"" w "\\",\\"en\\":\\"" w "\\"}]}"\n'
    't ::= [^"\\\\\\n]{1,60}\n'
    'w ::= [^"\\\\\\n]{1,20}\n')

# Lean SEE (spec 5.3 variant chosen from measurements: 17 tokens instead of 28; Spanish noun + gender come from vocab.py)
LEAN_SYSTEM = ('You name the object a person holds up to a webcam. Reply with JSON only. '
               'en: common English noun, singular, or "none" if nothing is held. n: how many. color: main color.')
LEAN_GBNF = (
    'root ::= "{\\"en\\":\\"" w "\\",\\"n\\":" [1-5] ",\\"color\\":\\"" (' + _COLOR_ALT + ') "\\"}"\n'
    'w ::= [a-z -]{1,24}\n')
# only used when the object is not in vocab.py
ES_SYSTEM = 'Give the Spanish noun (singular, no article) and its gender (m or f) for an English noun. Reply with JSON only.'
ES_GBNF = ('root ::= "{\\"es\\":\\"" w "\\",\\"g\\":\\"" ("m" | "f") "\\"}"\n'
           'w ::= [a-zñáéíóú -]{1,24}\n')
