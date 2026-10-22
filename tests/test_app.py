"""Integration tests for Flask application routes and REST API."""

import unittest
from app import app


class TestAppRoutes(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_get_index(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"NLP Text Intelligence Studio", response.data)

    def test_health_endpoint(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data['status'], 'healthy')
        self.assertTrue(json_data['models']['sentiment_loaded'])
        self.assertTrue(json_data['models']['category_loaded'])

    def test_post_analyze(self):
        response = self.client.post('/analyze', data={
            'raw_text': 'Microsoft and Google released groundbreaking neural network models in Seattle today.'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Results", response.data)

    def test_post_analyze_empty(self):
        response = self.client.post('/analyze', data={'raw_text': ''})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please enter or paste text", response.data)

    def test_api_analyze(self):
        response = self.client.post('/api/analyze', json={
            'text': 'Tim Cook introduced the new Apple iPhone in Cupertino. Profits rose 20%.'
        })
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data['status'], 'success')
        self.assertIn('summary', json_data)
        self.assertIn('sentiment', json_data)
        self.assertIn('category', json_data)

    def test_download_pdf(self):
        response = self.client.post('/download_pdf', data={
            'raw_text': 'A comprehensive overview of recent technological transformations.'
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'application/pdf')
        self.assertGreater(len(response.data), 1000)


if __name__ == '__main__':
    unittest.main()
