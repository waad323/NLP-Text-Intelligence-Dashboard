"""Unit tests for text preprocessing pipeline."""

import unittest
from scripts.preprocess import clean_text


class TestPreprocessor(unittest.TestCase):

    def test_clean_text_normal(self):
        text = "Machine learning algorithms are discovering deep patterns in complex datasets!"
        cleaned = clean_text(text)
        self.assertIsInstance(cleaned, str)
        self.assertNotIn("!", cleaned)
        self.assertIn("machine", cleaned)
        self.assertIn("learning", cleaned)

    def test_html_tag_stripping(self):
        text = "<p>This is a <b>bold</b> statement with <br /><a href='http://example.com'>link</a>.</p>"
        cleaned = clean_text(text)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("http", cleaned)
        self.assertIn("bold", cleaned)

    def test_url_removal(self):
        text = "Visit https://openai.com or www.google.com for latest news."
        cleaned = clean_text(text)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("www", cleaned)
        self.assertNotIn("openai", cleaned)

    def test_empty_and_none(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text("   "), "")

    def test_numbers_removed(self):
        text = "The score was 100 out of 200 points in year 2026."
        cleaned = clean_text(text)
        self.assertNotIn("100", cleaned)
        self.assertNotIn("2026", cleaned)


if __name__ == '__main__':
    unittest.main()
