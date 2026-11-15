"""~120 common objects a judge might hold up: English (what the vision model says) -> Spanish noun + gender.
A name found here is trusted over the model's own Spanish. Human-review the Spanish before the demo."""
from .spanish import normalize

# en: (es, gender, plural override or None, plural_only)
_V = {
    # everyday carry
    "phone": ("teléfono", "m"), "key": ("llave", "f"), "wallet": ("cartera", "f"),
    "glasses": ("gafas", "f", None, True), "sunglasses": ("gafas de sol", "f", None, True),
    "watch": ("reloj", "m"), "badge": ("credencial", "f"), "headphones": ("auriculares", "m", None, True),
    "charger": ("cargador", "m"), "backpack": ("mochila", "f"), "bag": ("bolsa", "f"), "purse": ("bolso", "m"),
    "suitcase": ("maleta", "f"), "hat": ("sombrero", "m"), "cap": ("gorra", "f"), "mask": ("mascarilla", "f"),
    "lip balm": ("bálsamo labial", "m", "bálsamos labiales"), "lipstick": ("pintalabios", "m"),
    "coin": ("moneda", "f"), "card": ("tarjeta", "f"), "money": ("billete", "m"), "passport": ("pasaporte", "m"),
    "umbrella": ("paraguas", "m"), "ring": ("anillo", "m"), "necklace": ("collar", "m"), "bracelet": ("pulsera", "f"),
    "phone case": ("funda", "f"), "usb drive": ("memoria USB", "f", "memorias USB"),
    # desk
    "pen": ("bolígrafo", "m"), "pencil": ("lápiz", "m"), "marker": ("marcador", "m"), "notebook": ("cuaderno", "m"),
    "book": ("libro", "m"), "paper": ("papel", "m"), "envelope": ("sobre", "m"), "folder": ("carpeta", "f"),
    "scissors": ("tijeras", "f", None, True), "stapler": ("grapadora", "f"), "tape": ("cinta", "f"),
    "sticker": ("pegatina", "f"), "eraser": ("goma", "f"), "ruler": ("regla", "f"), "calculator": ("calculadora", "f"),
    "paper clip": ("clip", "m"), "magazine": ("revista", "f"), "newspaper": ("periódico", "m"), "map": ("mapa", "m"),
    # electronics
    "laptop": ("portátil", "m"), "mouse": ("ratón", "m"), "keyboard": ("teclado", "m"), "monitor": ("pantalla", "f"),
    "tablet": ("tableta", "f"), "camera": ("cámara", "f"), "speaker": ("altavoz", "m"), "remote": ("control remoto", "m", "controles remotos"),
    "cable": ("cable", "m"), "lamp": ("lámpara", "f"), "clock": ("reloj", "m"), "microphone": ("micrófono", "m"),
    # kitchen / food
    "cup": ("taza", "f"), "glass": ("vaso", "m"), "bottle": ("botella", "f"), "water bottle": ("botella de agua", "f"),
    "can": ("lata", "f"), "plate": ("plato", "m"), "bowl": ("tazón", "m"), "spoon": ("cuchara", "f"),
    "fork": ("tenedor", "m"), "knife": ("cuchillo", "m"), "napkin": ("servilleta", "f"), "straw": ("pajita", "f"),
    "thermos": ("termo", "m"), "snack": ("aperitivo", "m"), "banana": ("plátano", "m"), "apple": ("manzana", "f"),
    "orange": ("naranja", "f"), "lemon": ("limón", "m"), "tomato": ("tomate", "m"), "carrot": ("zanahoria", "f"),
    "bread": ("pan", "m"), "cookie": ("galleta", "f"), "chocolate": ("chocolate", "m"), "candy": ("caramelo", "m"),
    "egg": ("huevo", "m"),
    # home / personal
    "chair": ("silla", "f"), "table": ("mesa", "f"), "door": ("puerta", "f"), "window": ("ventana", "f"),
    "pillow": ("almohada", "f"), "blanket": ("manta", "f"), "towel": ("toalla", "f"), "soap": ("jabón", "m"),
    "toothbrush": ("cepillo de dientes", "m", "cepillos de dientes"), "comb": ("peine", "m"), "mirror": ("espejo", "m"),
    "hand sanitizer": ("gel desinfectante", "m", "geles desinfectantes"), "tissue": ("pañuelo", "m"),
    "candle": ("vela", "f"), "lighter": ("encendedor", "m"), "plant": ("planta", "f"), "flower": ("flor", "f"),
    "toy": ("juguete", "m"), "ball": ("pelota", "f"), "teddy bear": ("oso de peluche", "m", "osos de peluche"),
    "guitar": ("guitarra", "f"), "balloon": ("globo", "m"), "star": ("estrella", "f"),
    # clothes
    "shoe": ("zapato", "m"), "sock": ("calcetín", "m"), "shirt": ("camisa", "f"), "jacket": ("chaqueta", "f"),
    "sweater": ("suéter", "m"),
    # animals
    "cat": ("gato", "m"), "dog": ("perro", "m"),
}

SYN = {
    "mobile phone": "phone", "cell phone": "phone", "cellphone": "phone", "smartphone": "phone", "iphone": "phone",
    "android phone": "phone", "telephone": "phone", "mug": "cup", "coffee cup": "cup", "coffee mug": "cup",
    "tea cup": "cup", "teacup": "cup", "paper cup": "cup", "eyeglasses": "glasses", "spectacles": "glasses",
    "reading glasses": "glasses", "sun glasses": "sunglasses", "plastic bottle": "bottle", "drink bottle": "bottle",
    "earphones": "headphones", "earbuds": "headphones", "airpods": "headphones", "headset": "headphones",
    "computer mouse": "mouse", "notebook computer": "laptop", "macbook": "laptop", "laptop computer": "laptop",
    "ballpoint pen": "pen", "ballpoint": "pen", "marker pen": "marker", "sharpie": "marker", "highlighter": "marker",
    "credit card": "card", "debit card": "card", "bank card": "card", "id card": "badge", "name tag": "badge",
    "id badge": "badge", "lanyard": "badge", "keys": "key", "car key": "key", "key ring": "key", "keychain": "key",
    "chapstick": "lip balm", "tv remote": "remote", "remote control": "remote", "flash drive": "usb drive",
    "usb stick": "usb drive", "thumb drive": "usb drive", "wristwatch": "watch", "smartwatch": "watch",
    "baseball cap": "cap", "face mask": "mask", "bill": "money", "banknote": "money", "dollar bill": "money",
    "bookbag": "backpack", "rucksack": "backpack", "handbag": "purse", "tissues": "tissue", "kleenex": "tissue",
    "sanitizer": "hand sanitizer", "tooth brush": "toothbrush", "stuffed animal": "teddy bear", "teddy": "teddy bear",
    "screen": "monitor", "display": "monitor", "tape roll": "tape", "scotch tape": "tape", "post-it": "sticker",
    "sticky note": "sticker", "can of soda": "can", "soda can": "can", "tin": "can", "cell": "phone",
    "trainer": "shoe", "sneaker": "shoe", "sneakers": "shoe", "hoodie": "sweater", "jumper": "sweater",
    "airpod case": "headphones", "airpods case": "headphones", "earbud case": "headphones", "charging case": "headphones",
    "golf ball": "ball", "tennis ball": "ball", "basketball": "ball", "soccer ball": "ball", "football": "ball",
    "t-shirt": "shirt", "tshirt": "shirt", "coat": "jacket", "kitten": "cat", "puppy": "dog", "cutlery": "fork",
}
_NORM_V = {normalize(k): k for k in _V}
_NORM_SYN = {normalize(k): v for k, v in SYN.items()}


def canonical(en: str):
    """Model's English noun -> the vocab key, or None. Tries synonyms, then a crude singular."""
    e = normalize(en)
    if not e or e == "none":
        return None
    if e in _NORM_V:
        return _NORM_V[e]
    if e in _NORM_SYN and _NORM_SYN[e] in _V:
        return _NORM_SYN[e]
    for suf in ("es", "s"):
        if e.endswith(suf) and e[: -len(suf)] in _NORM_V:
            return _NORM_V[e[: -len(suf)]]
        if e.endswith(suf) and e[: -len(suf)] in _NORM_SYN and _NORM_SYN[e[: -len(suf)]] in _V:
            return _NORM_SYN[e[: -len(suf)]]
    # more specific than our list ("golf ball", "leather wallet"): fall back to the head noun (last word)
    parts = e.split(" ")
    if len(parts) > 1:
        return canonical(parts[-1])
    return None


def lookup(en: str):
    """-> dict(en, es, g, pl, only) or None."""
    key = canonical(en)
    if key is None:
        return None
    t = _V[key]
    return {"en": key, "es": t[0], "g": t[1], "pl": t[2] if len(t) > 2 else None, "only": bool(t[3]) if len(t) > 3 else False}


def same_object(a_en: str, b_en: str) -> bool:
    ca, cb = canonical(a_en), canonical(b_en)
    if ca and cb:
        return ca == cb
    return normalize(a_en) == normalize(b_en)


COMMON = ["phone", "key", "wallet", "water bottle", "pen", "laptop", "glasses", "watch", "cup", "notebook",
          "headphones", "charger", "mouse", "backpack", "book", "apple", "banana", "can", "card", "remote",
          "pencil", "scissors", "stapler", "sunglasses", "hat", "mask", "coin", "marker", "tablet", "camera",
          "bottle", "spoon", "fork", "plate", "bowl", "lamp", "clock", "chair", "umbrella", "shoe"]
COMMON_COLORS = {
    "phone": ["black", "white", "blue"], "key": ["silver", "gold", "black"], "wallet": ["black", "brown", "blue"],
    "water bottle": ["blue", "black", "silver"], "pen": ["black", "blue", "red"], "laptop": ["silver", "gray", "black"],
    "glasses": ["black", "brown", "gold"], "watch": ["black", "silver", "brown"], "cup": ["white", "blue", "red"],
    "notebook": ["black", "blue", "red"], "headphones": ["white", "black", "blue"], "charger": ["white", "black", "gray"],
    "mouse": ["black", "white", "gray"], "backpack": ["black", "blue", "gray"], "book": ["red", "blue", "green"],
    "apple": ["red", "green", "yellow"], "banana": ["yellow", "green", "brown"], "can": ["red", "silver", "blue"],
    "card": ["white", "blue", "black"], "remote": ["black", "gray", "white"], "pencil": ["yellow", "black", "red"],
    "scissors": ["black", "red", "blue"], "stapler": ["black", "gray", "red"], "sunglasses": ["black", "brown", "gold"],
    "hat": ["black", "blue", "red"], "mask": ["white", "blue", "black"], "coin": ["silver", "gold", "brown"],
    "marker": ["black", "red", "blue"], "tablet": ["black", "silver", "white"], "camera": ["black", "silver", "gray"],
    "bottle": ["blue", "green", "white"], "spoon": ["silver", "white", "black"], "fork": ["silver", "white", "black"],
    "plate": ["white", "blue", "black"], "bowl": ["white", "blue", "red"], "lamp": ["white", "black", "gray"],
    "clock": ["black", "white", "silver"], "chair": ["black", "brown", "red"], "umbrella": ["black", "blue", "red"],
    "shoe": ["black", "white", "brown"],
}
