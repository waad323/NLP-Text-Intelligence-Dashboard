"""NLP Text Intelligence Dashboard - Production Flask Application.

An interactive AI-powered web studio providing extractive summarization,
TF-IDF keyword extraction, IMDb sentiment analysis, BBC news categorization,
Named Entity Recognition (NER), and automated PDF intelligence report export.
"""

import io
import json
import logging
import os
import sys

from flask import Flask, jsonify, render_template, request, send_file

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.ml_predictor import get_predictor
from modules.ner_extractor import extract_entities
from modules.pdf_generator import generate_pdf_report
from modules.text_analyzer import analyze_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'nlp-intelligence-studio-key-2026')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max payload

# Load curated sample presets
SAMPLES_PATH = os.path.join(BASE_DIR, 'datasets', 'sample_texts.json')
SAMPLE_PRESETS = []
if os.path.exists(SAMPLES_PATH):
    try:
        with open(SAMPLES_PATH, 'r', encoding='utf-8') as f:
            SAMPLE_PRESETS = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load sample presets: {e}")


@app.route('/', methods=['GET'])
def index():
    """Renders the main dashboard interface with preset samples."""
    return render_template('index.html', results=None, samples=SAMPLE_PRESETS, error=None)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles text analysis submission from web UI form."""
    raw_text = request.form.get('raw_text', '').strip()
    if not raw_text:
        return render_template(
            'index.html',
            results=None,
            samples=SAMPLE_PRESETS,
            error="Please enter or paste text to perform NLP analysis.",
        )

    try:
        # 1. Extractive summarization, TF-IDF keywords, linguistic statistics
        summary, keywords, stats = analyze_text(raw_text, num_sentences=3, num_keywords=6)

        # 2. ML Inference (Sentiment and Topic Classification)
        predictor = get_predictor()
        sent_res = predictor.predict_sentiment(raw_text)
        cat_res = predictor.predict_category(raw_text)

        # 3. Named Entity Recognition
        entities = extract_entities(raw_text)

        results = {
            "original_text": raw_text,
            "summary": summary,
            "keywords": keywords,
            "sentiment": sent_res.label,
            "sentiment_confidence": round(sent_res.confidence * 100, 1),
            "sentiment_details": sent_res.to_dict(),
            "category": cat_res.label,
            "category_confidence": round(cat_res.confidence * 100, 1),
            "category_details": cat_res.to_dict(),
            "entities": entities,
            "stats": stats,
        }

        return render_template('index.html', results=results, samples=SAMPLE_PRESETS, error=None)

    except Exception as e:
        logger.exception("Error executing text analysis pipeline")
        return render_template(
            'index.html',
            results=None,
            samples=SAMPLE_PRESETS,
            error=f"Pipeline Processing Error: {str(e)}",
        )


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """REST API endpoint returning structured JSON NLP analytics."""
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({"error": "Missing or empty 'text' field."}), 400

    try:
        summary, keywords, stats = analyze_text(text)
        predictor = get_predictor()
        sent_res = predictor.predict_sentiment(text)
        cat_res = predictor.predict_category(text)
        entities = extract_entities(text)

        return jsonify({
            "status": "success",
            "summary": summary,
            "keywords": keywords,
            "sentiment": sent_res.to_dict(),
            "category": cat_res.to_dict(),
            "entities": entities,
            "statistics": stats,
        })
    except Exception as e:
        logger.exception("API Error")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    """Generates and streams publication-grade PDF intelligence report."""
    raw_text = request.form.get('raw_text', '').strip()
    if not raw_text:
        return "Source text is required to generate report.", 400

    try:
        summary, keywords, stats = analyze_text(raw_text)
        predictor = get_predictor()
        sent_res = predictor.predict_sentiment(raw_text)
        cat_res = predictor.predict_category(raw_text)
        entities = extract_entities(raw_text)

        buffer = io.BytesIO()
        generate_pdf_report(
            buffer=buffer,
            text=raw_text,
            sentiment=sent_res.label,
            keywords=keywords,
            entities=entities,
            summary=summary,
            category=cat_res.label,
            confidence_sent=sent_res.confidence * 100,
            confidence_cat=cat_res.confidence * 100,
            stats=stats,
        )
        buffer.seek(0)

        filename = f"NLP_Intelligence_Report_{cat_res.label.replace(' ', '_')}.pdf"
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf",
        )
    except Exception as e:
        logger.exception("PDF generation error")
        return f"PDF Generation Error: {str(e)}", 500


@app.route('/health', methods=['GET'])
def health():
    """Liveness probe and model readiness status."""
    predictor = get_predictor()
    return jsonify({
        "status": "healthy",
        "models": {
            "sentiment_loaded": predictor.sentiment_model is not None,
            "category_loaded": predictor.category_model is not None,
        },
        "sample_presets": len(SAMPLE_PRESETS),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() in ('true', '1')
    print(f"\n=======================================================")
    print(f"  NLP Text Intelligence Dashboard running on http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=debug)
