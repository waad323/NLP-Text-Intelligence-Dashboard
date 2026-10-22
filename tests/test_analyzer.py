"""Unit tests for extractive text analyzer and TF-IDF keyword extraction."""

import unittest
from modules.text_analyzer import analyze_text, clean_sentence_for_tfidf, compute_text_stats


class TestTextAnalyzer(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "Artificial intelligence is rapidly transforming global technological ecosystems. "
            "Machine learning algorithms extract complex statistical patterns from vast enterprise databases. "
            "Engineers leverage natural language processing for automated intelligence extraction. "
            "Modern deep learning architectures provide high-throughput predictive inference. "
            "Extractive summarization preserves original sentence integrity while reducing reading friction."
        )

    def test_analyze_text_basic(self):
        summary, keywords, stats = analyze_text(self.sample_text, num_sentences=2, num_keywords=3)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 20)
        self.assertIsInstance(keywords, list)
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["sentence_count"], 5)
        self.assertTrue(stats["word_count"] > 30)

    def test_short_text(self):
        short = "Single sentence input."
        summary, keywords, stats = analyze_text(short, num_sentences=3)
        self.assertEqual(summary, "Single sentence input.")

    def test_empty_text(self):
        summary, keywords, stats = analyze_text("")
        self.assertIn("No text", summary)
        self.assertEqual(keywords, [])
        self.assertEqual(stats["word_count"], 0)

    def test_clean_sentence(self):
        sent = "Check out https://nlp.org! Testing punctuation, uppercase ABC, and spaces."
        cleaned = clean_sentence_for_tfidf(sent)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertIn("testing", cleaned)


if __name__ == '__main__':
    unittest.main()
