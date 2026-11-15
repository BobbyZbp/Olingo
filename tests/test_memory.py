"""The call's memory: 30 seconds word for word, older turns compressed, questions never repeated (no model here)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import memory  # noqa: E402


class Sketch(unittest.TestCase):
    def test_same_topic_is_close_other_topic_is_far(self):
        a = memory.sketch("Do you like oranges?")
        self.assertGreater(float(a @ memory.sketch("Do you like your orange?")), 0.85)
        self.assertGreater(float(a @ memory.sketch("Do you enjoy eating oranges?")), 0.3)
        self.assertLess(float(a @ memory.sketch("Do you have a pet?")), 0.1)
        z = memory.sketch("你喜欢吃橘子吗？")
        self.assertGreater(float(z @ memory.sketch("你喜欢橘子吗？")), 0.6)
        self.assertLess(float(z @ memory.sketch("你的杯子是什么颜色的？")), 0.2)

    def test_gist(self):
        self.assertEqual(memory.gist("How is your day going?"), "day going")
        self.assertEqual(memory.gist("I went to the library and studied chemistry"), "went library studied chemistry")
        self.assertEqual(memory.gist("我今天去了图书馆，学了一天化学"), "我今天去了图书馆学了一天化学")


class Window(unittest.TestCase):
    def test_thirty_seconds_word_for_word_then_compressed(self):
        m = memory.Memory(30.0)
        m.add("user", "My name is Bobby.", 0)
        m.add("assistant", "Nice to meet you, Bobby! How is your day going?", 2)
        m.add("user", "I studied chemistry in the library all morning.", 10)
        m.add("assistant", "That sounds like hard work.", 12)
        m.add("user", "What is this?", 50)
        m.add("assistant", "That is an orange.", 52)
        w = m.window(55)
        self.assertEqual([x["content"] for x in w], ["What is this?", "That is an orange."])
        s = m.summary()
        self.assertIn("I studied chemistry in the library all morning", s["told"])
        self.assertIn("My name is Bobby", s["told"])
        self.assertIn("hard work", " ".join(s["said"]))               # its own lines: a few content words
        self.assertEqual(m.stats()["compressed"], 4)

    def test_the_last_exchange_survives_a_long_pause(self):
        m = memory.Memory(30.0)
        m.add("user", "Hello.", 0)
        m.add("assistant", "Do you like tea?", 1)
        self.assertEqual([x["content"] for x in m.window(500)], ["Hello.", "Do you like tea?"])

    def test_window_starts_with_the_learner(self):
        m = memory.Memory(30.0)
        m.add("assistant", "Hi! I'm Olingo.", 0)
        m.add("assistant", "What's your name?", 1)
        m.add("user", "Bobby.", 5)
        self.assertEqual(m.window(6), [{"role": "user", "content": "Bobby."}])


class Questions(unittest.TestCase):
    def test_closest_and_nearest(self):
        m = memory.Memory()
        for q in ["How is your day going?", "Do you like oranges?", "What color is your cup?"]:
            m.asked(q)
        sim, prev = m.closest("Do you like your orange?")
        self.assertEqual(prev, "Do you like oranges?")
        self.assertGreater(sim, 0.85)
        self.assertLess(m.closest("Do you have a pet?")[0], 0.2)
        self.assertEqual(len(m.nearest("Do you have a pet?")), 3)
        self.assertEqual(memory.Memory().closest("anything")[0], 0.0)


if __name__ == "__main__":
    unittest.main()
