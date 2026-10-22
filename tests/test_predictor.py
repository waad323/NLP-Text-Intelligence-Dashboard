"""Unit tests for ML inference pipelines (Sentiment & Topic classification)."""

import unittest
from modules.ml_predictor import get_predictor


class TestMLPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()

    def test_sentiment_positive(self):
        sample = "An absolutely magnificent film with brilliant acting and breathtaking cinematography. Highly recommended!"
        res = self.predictor.predict_sentiment(sample)
        self.assertEqual(res.label, "Positive")
        self.assertGreater(res.confidence, 0.5)
        self.assertGreater(res.positive_prob, res.negative_prob)

    def test_sentiment_negative(self):
        sample = "A completely terrible and boring disaster. The worst film I have ever watched with awful acting."
        res = self.predictor.predict_sentiment(sample)
        self.assertEqual(res.label, "Negative")
        self.assertGreater(res.confidence, 0.5)
        self.assertGreater(res.negative_prob, res.positive_prob)

    def test_category_tech(self):
        sample = "Apple and Microsoft announced new generative artificial intelligence chips with high GPU computing efficiency."
        res = self.predictor.predict_category(sample)
        self.assertEqual(res.label.lower(), "tech")
        self.assertGreater(res.confidence, 0.2)

    def test_category_sport(self):
        sample = "Arsenal defeated Chelsea in an intense Premier League match with two second-half goals by the striker."
        res = self.predictor.predict_category(sample)
        self.assertEqual(res.label.lower(), "sport")

    def test_empty_text_fallback(self):
        res_sent = self.predictor.predict_sentiment("")
        self.assertEqual(res_sent.label, "Neutral")
        res_cat = self.predictor.predict_category("")
        self.assertEqual(res_cat.label, "General")


if __name__ == '__main__':
    unittest.main()
