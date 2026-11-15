"""Everything the call decides without a model: commands, routing, the opening question, scoring, word swap."""
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend import call, lang  # noqa: E402
from backend.speech import MockSpeech  # noqa: E402


class Rules(unittest.TestCase):
    def test_setup_answers(self):
        self.assertEqual(lang.parse_setup("I want to learn Chinese. I'm a beginner."), {"l2": "zh", "level": 1})
        self.assertEqual(lang.parse_setup("Spanish, intermediate I guess"), {"l2": "es", "level": 2})
        self.assertEqual(lang.parse_setup("我想学英语，我是初学者"), {"l2": "en", "level": 1})
        self.assertEqual(lang.parse_setup("Japanese please, I'm pretty advanced"), {"l2": "ja", "level": 3})
        self.assertEqual(lang.parse_setup("um, hello?"), {"l2": None, "level": None})

    def test_commands(self):
        self.assertEqual(call.intent("Bye!"), "bye")
        self.assertEqual(call.intent("Can you say that again?"), "again")
        self.assertEqual(call.intent("再说一遍"), "again")
        self.assertEqual(call.intent("Slower, please."), "slower")
        self.assertEqual(call.intent("more Chinese"), "more")
        self.assertEqual(call.intent("English please"), "less")
        self.assertEqual(call.intent("What is this again?"), "ask")
        self.assertEqual(call.intent("Yes please", offer=True), "yes")
        self.assertEqual(call.intent("Yes please"), "ask")
        self.assertEqual(call.intent("No thanks", offer=True), "no")
        self.assertEqual(call.intent("Hey! Yes, please.", offer=True), "yes")
        self.assertEqual(call.intent("You pick!"), "pick")
        self.assertEqual(call.intent("Pick up the phone"), "ask")
        self.assertEqual(call.intent("What's the weather like today?"), "offline")
        self.assertEqual(call.intent("Quel temps fait-il ?"), "offline")
        self.assertEqual(call.intent("今日の天気はどうですか"), "offline")
        self.assertEqual(call.intent("今天天气怎么样"), "offline")

    def test_right_sized_looking(self):
        self.assertTrue(call.SIMPLE_LOOK.search(call.norm("What is this?")))
        self.assertTrue(call.SIMPLE_LOOK.search(call.norm("Can you see what I'm holding?")) or call.SIMPLE_LOOK.search(call.norm("What am I holding?")))
        self.assertTrue(call.SIMPLE_LOOK.search("这是什么"))
        self.assertFalse(call.SIMPLE_LOOK.search(call.norm("What color is my shirt?")))
        self.assertFalse(call.SIMPLE_LOOK.search(call.norm("Why do people wear them?")))
        self.assertIn("object", call.VAGUE)
        self.assertTrue(call.ASKING.search(call.norm("What do you like to read?")))
        self.assertFalse(call.ASKING.search(call.norm("I like reading books.")))

    def test_commands_french_and_japanese(self):
        self.assertEqual(call.intent("Au revoir !"), "bye")
        self.assertEqual(call.intent("Tu peux répéter ?"), "again")
        self.assertEqual(call.intent("Plus lentement, s'il te plaît."), "slower")
        self.assertEqual(call.intent("Tu choisis."), "pick")
        self.assertEqual(call.intent("Euh, oui, d'accord.", offer=True), "yes")
        self.assertEqual(call.intent("Non, autre chose.", offer=True), "no")
        self.assertEqual(call.intent("Qu'est-ce que c'est ?"), "ask")
        self.assertEqual(call.intent("さようなら"), "bye")
        self.assertEqual(call.intent("もう一度お願いします"), "again")
        self.assertEqual(call.intent("もっとゆっくり"), "slower")
        self.assertEqual(call.intent("選んでください"), "pick")
        self.assertEqual(call.intent("はい、お願いします", offer=True), "yes")
        self.assertEqual(call.intent("いいえ、結構です", offer=True), "no")
        self.assertEqual(call.intent("これは何ですか"), "ask")
        self.assertEqual(call.route("C'est quoi ça ?")[0], 2)
        self.assertEqual(call.route("J'ai mangé des pâtes ce midi")[0], 1)
        self.assertEqual(call.route("これは何ですか")[0], 2)
        self.assertEqual(call.route("今日は天気がいいですね")[0], 1)

    def test_every_native_language_has_every_line(self):
        for l1 in lang.L1S:
            self.assertEqual(set(lang.FIXED[l1]), set(lang.FIXED["en"]), l1)
            self.assertEqual(set(lang.VARIED[l1]), set(lang.VARIED["en"]), l1)
            self.assertTrue(lang.FILLERS[l1] and lang.VOICES[l1])
            for code in lang.LANGS:
                self.assertTrue(lang.LANGS[code][l1], (code, l1))
            self.assertNotIn("{", lang.line("offer", l1, "zh") + lang.line("see_room", l1, "zh", "x") + lang.preview_line("aiden", l1, "es"))

    def test_routing(self):
        self.assertEqual(call.route("What am I holding?")[0], 2)
        self.assertEqual(call.route("这是什么")[0], 2)
        self.assertEqual(call.route("I had noodles for lunch")[0], 1)
        self.assertEqual(call.route("How do you say water?")[0], 1)
        self.assertEqual(call.route("How do you say this?")[0], 2)
        # it looks only when the words point at the picture; anything else is conversation (the 9B, without a frame)
        for t in ("Mostly science fiction", "They are black", "I like it", "That is true", "我每天都戴"):
            self.assertEqual(call.route(t)[0], 1, t)
        self.assertTrue(call.route("Mostly science fiction")[1].startswith("conversation"))
        for t in ("This is my cup", "Look at this", "What color is this?", "你看这个", "Do you like my shirt?"):
            self.assertEqual(call.route(t)[0], 2, t)

    def test_noise(self):
        self.assertTrue(call.is_noise({"text": "", "audio_s": 0}))
        self.assertTrue(call.is_noise({"text": "da " * 30, "audio_s": 4, "avg_p": 0.7}))
        self.assertTrue(call.is_noise({"text": "you", "audio_s": 0.4, "avg_p": 0.5}))
        self.assertTrue(call.is_noise({"text": "(upbeat music)", "audio_s": 2.0, "avg_p": 0.8}))
        self.assertTrue(call.is_noise({"text": "[BLANK_AUDIO]", "audio_s": 1.0, "avg_p": 0.9}))
        self.assertTrue(call.is_noise({"text": "what is this", "audio_s": 1.5, "avg_p": 0.2}))
        self.assertFalse(call.is_noise({"text": "What is this?", "audio_s": 1.2, "avg_p": 0.8}))
        self.assertFalse(call.is_noise({"text": "苹果", "audio_s": 0.8, "avg_p": 0.7}))

    def test_scoring_chinese(self):
        ok = call.score("杯子", {"text": "杯子", "words": [{"w": "杯", "p": 0.9}, {"w": "子", "p": 0.8}]}, "zh")
        self.assertEqual([c["p"] for c in ok["chars"]], [0.9, 0.8])
        homophone = call.score("杯子", {"text": "被子", "words": [{"w": "被", "p": 0.9}, {"w": "子", "p": 0.8}]}, "zh")
        self.assertAlmostEqual(homophone["chars"][0]["p"], 0.81)       # same sound, another character: nearly full marks
        trad = call.score("钥匙", {"text": "鑰匙", "words": [{"w": "鑰匙", "p": 0.8}]}, "zh")
        self.assertEqual(trad["score"], 0.8)
        self.assertEqual(call.score("杯子", {"text": "hello", "words": [{"w": "hello", "p": 0.9}]}, "zh")["score"], 0.0)

    def test_scoring_words(self):
        r = call.score("una taza azul", {"text": "una tasa azul", "words": [
            {"w": "una", "p": 0.9}, {"w": "tasa", "p": 0.6}, {"w": "azul", "p": 0.8}]}, "es")
        self.assertEqual([c["p"] for c in r["chars"]], [0.9, 0.6, 0.8])

    def test_pinyin(self):
        self.assertEqual(call.readings("你好，杯子!", "zh"), ["nǐ", "hǎo", "", "bēi", "zi", ""])
        self.assertEqual(call.reading("苹果", "zh"), "píng guǒ")

    def test_one_voice_for_the_whole_call(self):
        self.assertEqual(lang.voice_for("en", None, "zh"), "vivian")          # native in the language being learned
        self.assertEqual(lang.voice_for("zh", None, "en"), "aiden")
        self.assertEqual(lang.voice_for("en", None, "es"), "aiden")
        self.assertEqual(lang.voice_for("en", "serena", "zh"), "serena")      # the learner's own choice wins
        sp = MockSpeech()
        c = call.Call(sp, lambda: None, l1="en", l2="zh")
        c.running = True
        c.words = {"cup": {"l2": "杯子", "reading": "", "score": 0.9, "drilled": True}}
        c.reply("Where is your cup?", "en")
        self.assertEqual(sp.said_log, [("Where is your", "en"), ("杯子", "zh")])     # run by run, each in its language
        c.say("苹果", lang="zh")
        self.assertEqual(set(sp.voices), {"vivian"})                          # ... all by the same voice

    def test_names(self):
        self.assertEqual(lang.parse_name("My name is Bobby.", "en"), "Bobby")
        self.assertEqual(lang.parse_name("bobby", "en"), "Bobby")
        self.assertEqual(lang.parse_name("I think it is going fine today", "en"), "")
        self.assertEqual(lang.parse_name("我叫小明", "zh"), "小明")
        self.assertEqual(lang.parse_name("Je m'appelle Marie.", "fr"), "Marie")
        self.assertEqual(lang.parse_name("田中です", "ja"), "田中")

    def test_what_is_left_to_talk_about(self):
        c = call.Call(MockSpeech(), lambda: None, l1="en", l2="zh",
                      room=[{"l1": "wall", "l2": "墙"}, {"l1": "cup", "l2": "杯子"}, {"l1": "shirt", "l2": "衬衫"}])
        self.assertEqual(c.next_askable()["l1"], "cup")                 # a wall is not worth a lesson
        c.words["cup"] = {"l2": "杯子", "reading": "", "score": 0.9, "drilled": True}
        self.assertEqual(c.next_askable()["l1"], "shirt")
        c.declined.add("shirt")
        self.assertIsNone(c.next_askable())                              # everything in view has come up

    def test_shown_around_waits_until_the_picture_settles(self):
        c = call.Call(MockSpeech(), lambda: None)
        c.running, c.floor_open = True, True
        c.exploring = __import__("time").time()
        self.assertFalse(c.notify_scene(b"frame", 0.5, still_s=3.0))     # not before ten seconds have passed
        c.exploring -= 11
        self.assertFalse(c.notify_scene(b"frame", 0.5, still_s=0.4))     # ... and not while the camera still moves
        self.assertTrue(c.notify_scene(b"frame", 0.5, still_s=2.0))
        self.assertTrue(c._explore)

    def test_safe_phrases(self):
        self.assertEqual(call.frame_phrase("apple", "苹果", "en", "zh", 0), {"l2": "我的苹果", "l1": "my apple"})
        self.assertIsNone(call.frame_phrase("cup", "コップ", "en", "ja", 0))

    def test_word_swap(self):
        c = call.Call(MockSpeech(), lambda: None)
        c.words = {"cup": {"l2": "杯子", "reading": "", "score": 0.9, "drilled": True},
                   "apple": {"l2": "苹果", "reading": "", "score": 0.7, "drilled": True},
                   "noodles": {"l2": "面条", "reading": "", "score": 0.0, "drilled": False}}
        text, segs = c.swap("Do you keep your apples next to your cup? I like noodles.")
        self.assertEqual(text, "Do you keep your 苹果 next to your 杯子? I like noodles.")
        self.assertEqual([s["t"] for s in segs if s.get("l2")], ["苹果", "杯子"])
        self.assertEqual([s["g"] for s in segs if s.get("l2")], ["apple", "cup"])
        c.mix.append(c._units(segs))
        self.assertEqual(c.pct(), 15)


class SpeaksFirst(unittest.TestCase):
    """When something new is held up: speak first if the floor is silent, stop mid-sentence only for a big change."""
    def make(self):
        c = call.Call(MockSpeech(), lambda: b"jpeg", notice=True)
        c.running = True
        return c

    def test_off_unless_asked_for(self):
        c = call.Call(MockSpeech(), lambda: b"jpeg")
        c.running, c.floor_open = True, True
        self.assertFalse(c.notify_scene(b"frame", 0.5))        # movement alone is no reason to look

    def test_open_floor(self):
        c = self.make()
        c.floor_open = True
        self.assertTrue(c.notify_scene(b"frame", 0.11))
        self.assertFalse(c._cut)
        self.assertFalse(c.notify_scene(b"frame", 0.5))        # one at a time

    def test_interrupts_itself_only_for_a_big_change(self):
        c = self.make()
        c.speaking = 1
        self.assertFalse(c.notify_scene(b"frame", 0.11))       # a small change does not cut a sentence short
        self.assertTrue(c.notify_scene(b"frame", 0.30))
        self.assertTrue(c._cut)
        self.assertIsNone(c.say("this must wait"))             # everything else waits for the new thing

    def test_never_while_the_learner_talks_or_too_often(self):
        c = self.make()
        c.floor_open, c.hearing = True, True
        self.assertFalse(c.notify_scene(b"frame", 0.5))
        c.hearing = False
        c.streak = c.PRO_MAX
        self.assertFalse(c.notify_scene(b"frame", 0.5))
        c.streak = 0
        c.last_pro = __import__("time").time()
        self.assertFalse(c.notify_scene(b"frame", 0.5))


class NothingDropped(unittest.TestCase):
    """What the learner says is answered, also in the middle of a drill."""
    def make(self):
        c = call.Call(MockSpeech(), lambda: b"jpeg", l1="en", l2="zh")
        c.running = True
        return c

    def test_what_is_not_an_attempt(self):
        c = self.make()
        for text, lang in [("What does that mean?", "en"), ("bye", "en"), ("again", "en"), ("I do not know this one", "en"),
                           ("Can you say it slower please", "en")]:
            self.assertTrue(c.not_an_attempt({"text": text, "lang": lang}, "杯子"), text)
        # a sentence in any language, where one word was asked for (heard in a real call, scored as a bad repeat)
        self.assertTrue(c.not_an_attempt({"text": "I would like to study cat first", "lang": "es"}, "póster"))
        self.assertFalse(c.not_an_attempt({"text": "poster", "lang": "es"}, "póster"))
        self.assertFalse(c.not_an_attempt({"text": "el póster", "lang": "es"}, "póster"))
        for text, lang in [("杯子", "zh"), ("被子", "zh"), ("bei zi", "en"), ("cup", "en")]:
            self.assertFalse(c.not_an_attempt({"text": text, "lang": lang}, "杯子"), text)

    def test_a_question_in_a_drill_is_carried_to_the_loop(self):
        c = self.make()
        c.speech.type("What does that mean?")
        real = call.brain.phrases
        call.brain.phrases = lambda *a, **k: ([], {})
        try:
            self.assertFalse(c.teach("cup", "杯子"))
        finally:
            call.brain.phrases = real
        self.assertEqual(c.carry["text"], "What does that mean?")
        self.assertFalse(any(k == "score" for k in getattr(c, "_events", [])))


class MockCall(unittest.TestCase):
    """The loop itself, with typed input and a fake brain: name -> warm-up -> something it saw -> yes -> drill -> commands."""
    def test_flow(self):
        sp = MockSpeech()
        events = []
        real = (call.brain.answer, call.brain.phrases, call.brain.available, call.brain.same_question)
        real_accept = call.brain.accepts_offer
        call.brain.accepts_offer = lambda text, thing: (True, {"ms": 1})
        n = [0]

        def fake_answer(*a, **k):
            n[0] += 1
            say = f"Cups are useful, number {n[0]}."
            if k.get("on_say"):
                k["on_say"](say, True)
            return ({"say": say, "key_l1": "cup", "key": "杯子", "q": f"Do you like cup {n[0]}?"},
                    {"ms": 5, "prompt_ms": 1, "prompt_n": 10, "cache_n": 0, "predicted_n": 8, "tok_s": 50, "marks": {}})
        call.brain.answer = fake_answer
        call.brain.phrases = lambda *a, **k: ([{"l2": "一个杯子", "l1": "a cup"}], {"prompt_n": 5, "predicted_n": 5})
        call.brain.available = lambda t: True
        call.brain.same_question = lambda new, earlier: (False, {"ms": 1})
        try:
            c = call.Call(sp, lambda: b"jpeg", lambda kind, **d: events.append((kind, d)), l1="en", l2="zh", level=2,
                          room=[{"l1": "cup", "l2": "杯子"}])
            for line in ["Bobby", "I am fine", "yes", "杯子", "我的杯子", "一个杯子", "again", "you pick", "bye"]:
                sp.type(line)
            t = threading.Thread(target=c.run)
            t.start()
            t.join(timeout=20)
            self.assertFalse(t.is_alive())
        finally:
            call.brain.answer, call.brain.phrases, call.brain.available, call.brain.same_question = real
            call.brain.accepts_offer = real_accept
        kinds = [k for k, _ in events]
        prof = next(d for k, d in events if k == "profile")
        self.assertEqual((prof["l1"], prof["l2"], prof["level"], prof["voice"]), ("en", "zh", 2, "vivian"))
        self.assertEqual(c.name, "Bobby")
        self.assertIn("Nice to meet you, Bobby!", [t for t, _ in sp.said_log])
        self.assertIn("card", kinds)
        self.assertIn("score", kinds)
        self.assertIn("phrases", kinds)                       # intermediate: the word, then how it is used
        self.assertEqual(c.words["cup"]["l2"], "杯子")
        self.assertTrue(c.words["cup"]["drilled"])
        self.assertEqual([p["l2"] for p in c.words["cup"]["phrases"]], ["我的杯子", "一个杯子"])
        self.assertEqual([d["tier"] for k, d in events if k == "route"], [0, 1, 0, 1, 0, 0, 2, 0])
        said = [t for t, _ in sp.said_log]
        offers = [o.format(thing="cup") for o in call.L.OFFERS["en"]["see_room"]]
        self.assertTrue(any(o in said for o in offers))       # one of the wordings, not always the same one
        self.assertTrue(any(t.startswith(("In Chinese,", "The Chinese word", "Cup in Chinese", "We say")) for t in said))   # the word is taught in a sentence
        self.assertIn("杯子", said)
        self.assertIn("Now, how it is used.", said)
        self.assertTrue(any(t.startswith(('"my cup" is', 'To say "my cup"')) or t.endswith('means "my cup".') for t in said))   # explained first
        self.assertEqual(len([d for k, d in events if k == "score"]), 3)        # the word and both phrases were checked
        self.assertIn("Goodbye! See you next time.", said)
        self.assertEqual(c.deck()[0]["l1"], "cup")
        self.assertEqual(kinds[-1], "phase")


if __name__ == "__main__":
    unittest.main()
