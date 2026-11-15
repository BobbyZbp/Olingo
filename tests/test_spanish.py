import unittest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend import spanish as sp


class Phrases(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(sp.build_phrase("taza", "f", 1, "blue"), "una taza azul")
        self.assertEqual(sp.build_phrase("teléfono", "m", 1, "black"), "un teléfono negro")
        self.assertEqual(sp.build_phrase("llave", "f", 3, "black"), "unas llaves negras")
        self.assertEqual(sp.build_phrase("libro", "m", 2, "red"), "unos libros rojos")
        self.assertEqual(sp.build_phrase("taza", "f", 1, "none"), "una taza")

    def test_plurals(self):
        for w, p in [("lápiz", "lápices"), ("reloj", "relojes"), ("ratón", "ratones"), ("limón", "limones"),
                     ("paraguas", "paraguas"), ("gris", "grises"), ("mes", "meses"), ("cargador", "cargadores"), ("calcetín", "calcetines"),
                     ("jabón", "jabones"), ("suéter", "suéteres"), ("flor", "flores"), ("pan", "panes"),
                     ("tazón", "tazones"), ("autobús", "autobuses")]:
            self.assertEqual(sp.plural_word(w), p, w)

    def test_multiword(self):
        self.assertEqual(sp.plural_noun("botella de agua"), "botellas de agua")
        self.assertEqual(sp.plural_noun("bálsamo labial", "bálsamos labiales"), "bálsamos labiales")
        self.assertEqual(sp.build_phrase("botella de agua", "f", 2, "blue"), "unas botellas de agua azules")

    def test_plural_only(self):
        self.assertEqual(sp.build_phrase("gafas", "f", 1, "black", plural_only=True), "unas gafas negras")
        self.assertEqual(sp.build_phrase("tijeras", "f", 1, "none", plural_only=True), "unas tijeras")
        self.assertEqual(sp.deck_word("gafas", "f", plural_only=True), "las gafas")

    def test_colors(self):
        self.assertEqual(sp.color_es("red", "f"), "roja")
        self.assertEqual(sp.color_es("red", "m", 2), "rojos")
        self.assertEqual(sp.color_es("blue", "f", 2), "azules")
        self.assertEqual(sp.color_es("gray", "m", 2), "grises")
        self.assertEqual(sp.color_es("brown", "f", 2), "marrones")
        self.assertEqual(sp.color_es("green", "f", 2), "verdes")
        self.assertEqual(sp.color_es("white", "f", 2), "blancas")
        self.assertEqual(sp.color_es("none", "f"), "")

    def test_articles(self):
        self.assertEqual(sp.definite("f", 1), "la")
        self.assertEqual(sp.definite("m", 2), "los")
        self.assertEqual(sp.deck_word("taza", "f"), "la taza")


class Answers(unittest.TestCase):
    def test_color(self):
        self.assertTrue(sp.has_color("Es azul.", "blue"))
        self.assertTrue(sp.has_color("es roja", "red"))
        self.assertTrue(sp.has_color("son negras", "black"))
        self.assertFalse(sp.has_color("es verde", "blue"))

    def test_yes_no(self):
        self.assertEqual(sp.yes_no("Sí, es azul"), "yes")
        self.assertEqual(sp.yes_no("No, es verde"), "no")
        self.assertEqual(sp.yes_no("es azul"), None)

    def test_number(self):
        self.assertTrue(sp.has_number("tres", 3))
        self.assertTrue(sp.has_number("tengo 2", 2))
        self.assertTrue(sp.has_number("una", 1))
        self.assertFalse(sp.has_number("cuatro", 3))


class Alignment(unittest.TestCase):
    def test_perfect(self):
        heard = [{"w": "Una", "p": 0.9}, {"w": "taza", "p": 0.8}, {"w": "azul.", "p": 0.95}]
        r = sp.align("una taza azul", heard)
        self.assertEqual([x["p"] for x in r], [0.9, 0.8, 0.95])

    def test_accent_and_near_miss(self):
        heard = [{"w": "un", "p": 0.9}, {"w": "tasa", "p": 0.4}, {"w": "azul", "p": 0.9}]
        r = sp.align("una taza azul", heard)
        self.assertEqual(r[0]["p"], 0.0)            # 'una' vs 'un': 3 letters, no fuzzy
        self.assertEqual(r[1]["p"], 0.4)            # 'taza' vs 'tasa': 4 letters, distance 1
        self.assertEqual(r[2]["p"], 0.9)

    def test_missing_and_order(self):
        r = sp.align("unas llaves negras", [{"w": "llaves", "p": 0.7}])
        self.assertEqual([x["p"] for x in r], [0.0, 0.7, 0.0])

    def test_accents(self):
        r = sp.align("un teléfono negro", [{"w": "un", "p": 0.8}, {"w": "telefono", "p": 0.6}, {"w": "negro", "p": 0.9}])
        self.assertEqual([x["p"] for x in r], [0.8, 0.6, 0.9])

    def test_noun_pass(self):
        s = sp.align("una taza azul", [{"w": "una", "p": 0.4}, {"w": "taza", "p": 0.7}, {"w": "azul", "p": 0.3}])
        self.assertTrue(sp.noun_passed("una taza azul", s, "taza"))
        s2 = sp.align("una taza azul", [{"w": "una", "p": 0.9}, {"w": "taza", "p": 0.3}])
        self.assertFalse(sp.noun_passed("una taza azul", s2, "taza"))

    def test_clarity_colors(self):
        self.assertEqual(sp.clarity_color(0.8), "green")
        self.assertEqual(sp.clarity_color(0.6), "yellow")
        self.assertEqual(sp.clarity_color(0.2), "red")


if __name__ == "__main__":
    unittest.main()


class VocabFallbacks(unittest.TestCase):
    def test_head_noun_and_synonyms(self):
        from backend import vocab
        for en, key in [("golf ball", "ball"), ("leather wallet", "wallet"), ("airpod case", "headphones"),
                        ("keychain", "key"), ("airpods", "headphones"), ("mug", "cup"), ("iphone", "phone"), ("none", None), ("zzzz", None)]:
            self.assertEqual(vocab.canonical(en), key, en)
        self.assertTrue(vocab.same_object("golf ball", "ball"))


class ColourRedundancy(unittest.TestCase):
    def test_orange_orange(self):
        self.assertEqual(sp.build_phrase("naranja", "f", 1, "orange"), "una naranja")
        self.assertEqual(sp.build_phrase("taza", "f", 1, "orange"), "una taza naranja")
