"""
Fertilizer Recommendation System — Flask API Server
====================================================
Serves the frontend and provides ML prediction endpoints.

Endpoints:
  GET  /             → Serves index.html
  POST /predict      → Returns fertilizer prediction from ML model
  GET  /model-info   → Returns model metadata (accuracy, features, etc.)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─── App Setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# ─── Load Model & Encoders ──────────────────────────────────────────────────
try:
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    fertilizer_encoder = joblib.load(os.path.join(MODELS_DIR, 'fertilizer_encoder.pkl'))
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    print("Model and encoders loaded successfully!")
    print(f"   Model: {metadata['best_model']} ({metadata['accuracy']}% accuracy)")
except FileNotFoundError:
    print("Warning: Model files not found! Run 'python train_model.py' first.")
    model = None
    label_encoders = None
    fertilizer_encoder = None
    metadata = None

# ─── Fertilizer Information Database ─────────────────────────────────────────
FERTILIZER_INFO = {
    'Urea': {
        'type': 'Nitrogenous',
        'npk_ratio': '46-0-0',
        'application_rate': '100–150 kg/ha',
        'application_method': 'Broadcasting / Side dressing',
        'best_season': 'Kharif & Rabi',
        'description': 'High-nitrogen fertilizer ideal for crops with heavy nitrogen demand.',
        'tips': [
            'Apply in split doses — 50% at sowing, 25% at tillering, 25% at panicle initiation',
            'Best applied when soil is moist for better absorption',
            'Avoid application during heavy rainfall to prevent nitrogen loss',
            'Store in a cool, dry place away from direct sunlight'
        ],
        'dosage': {
            'Rice':         {'per_acre': '40–55 kg', 'splits': 3, 'schedule': '50% basal, 25% at tillering, 25% at panicle', 'total_per_season': '120–165 kg/ha'},
            'Wheat':        {'per_acre': '45–60 kg', 'splits': 3, 'schedule': '50% basal, 25% at CRI stage, 25% at booting', 'total_per_season': '110–150 kg/ha'},
            'Maize':        {'per_acre': '50–65 kg', 'splits': 3, 'schedule': '33% basal, 33% at knee-high, 33% at tasseling', 'total_per_season': '125–160 kg/ha'},
            'Sugarcane':    {'per_acre': '55–70 kg', 'splits': 3, 'schedule': '33% at planting, 33% at 45 days, 33% at 90 days', 'total_per_season': '135–175 kg/ha'},
            'Cotton':       {'per_acre': '35–50 kg', 'splits': 3, 'schedule': '33% basal, 33% at squaring, 33% at flowering', 'total_per_season': '90–125 kg/ha'},
            'Tobacco':      {'per_acre': '20–30 kg', 'splits': 2, 'schedule': '60% basal, 40% at 30 days', 'total_per_season': '50–75 kg/ha'},
            'Millets':      {'per_acre': '25–35 kg', 'splits': 2, 'schedule': '50% basal, 50% at 25–30 days', 'total_per_season': '60–90 kg/ha'},
            'Pulses':       {'per_acre': '8–12 kg', 'splits': 1, 'schedule': 'Full dose as basal (Pulses fix their own N)', 'total_per_season': '20–30 kg/ha'},
            'Oil Seeds':    {'per_acre': '15–25 kg', 'splits': 2, 'schedule': '50% basal, 50% at 30 days', 'total_per_season': '40–60 kg/ha'},
            'Ground Nuts':  {'per_acre': '8–12 kg', 'splits': 1, 'schedule': 'Full dose as basal (legume — minimal N needed)', 'total_per_season': '20–30 kg/ha'},
        },
        'expected_outcomes': {
            'crop_yield': {'label': 'Increased Crop Yield', 'value': '20–35%', 'desc': 'Rapid vegetative growth and higher grain/fruit production due to optimal nitrogen supply'},
            'cost_reduction': {'label': 'Reduced Fertilizer Costs', 'value': '15–25%', 'desc': 'Precise dosing eliminates wastage from over-application of nitrogen fertilizers'},
            'environmental': {'label': 'Environmental Sustainability', 'value': 'High', 'desc': 'Split application reduces nitrogen leaching into groundwater and nitrous oxide emissions'},
            'soil_health': {'label': 'Improved Soil Health', 'value': 'Moderate', 'desc': 'Maintains soil nitrogen balance; pair with organic matter to prevent long-term soil degradation'},
            'decision_making': {'label': 'Enhanced Decision Making', 'value': 'ML-Optimized', 'desc': 'Data-driven nitrogen recommendation eliminates guesswork and adapts to specific field conditions'}
        }
    },
    'DAP': {
        'type': 'Phosphatic',
        'npk_ratio': '18-46-0',
        'application_rate': '75–125 kg/ha',
        'application_method': 'Basal application / Drilling',
        'best_season': 'Pre-sowing',
        'description': 'Primary source of phosphorus, essential for root development and flowering.',
        'tips': [
            'Apply as a basal dose during land preparation for best results',
            'Mix well with soil to ensure even nutrient distribution',
            'Ideal for crops with high phosphorus demand like wheat and pulses',
            'Avoid surface application — incorporate into the root zone'
        ],
        'dosage': {
            'Rice':         {'per_acre': '30–45 kg', 'splits': 1, 'schedule': 'Full dose as basal during puddling', 'total_per_season': '75–110 kg/ha'},
            'Wheat':        {'per_acre': '40–50 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '100–125 kg/ha'},
            'Maize':        {'per_acre': '35–45 kg', 'splits': 1, 'schedule': 'Full dose as basal at planting', 'total_per_season': '85–110 kg/ha'},
            'Sugarcane':    {'per_acre': '40–55 kg', 'splits': 1, 'schedule': 'Full dose in furrow at planting', 'total_per_season': '100–135 kg/ha'},
            'Cotton':       {'per_acre': '30–40 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '75–100 kg/ha'},
            'Tobacco':      {'per_acre': '25–35 kg', 'splits': 1, 'schedule': 'Full dose as basal before transplanting', 'total_per_season': '60–85 kg/ha'},
            'Millets':      {'per_acre': '20–30 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '50–75 kg/ha'},
            'Pulses':       {'per_acre': '30–45 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '75–110 kg/ha'},
            'Oil Seeds':    {'per_acre': '25–40 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '60–100 kg/ha'},
            'Ground Nuts':  {'per_acre': '35–50 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '85–125 kg/ha'},
        },
        'expected_outcomes': {
            'crop_yield': {'label': 'Increased Crop Yield', 'value': '15–30%', 'desc': 'Stronger root systems and improved flowering lead to higher overall productivity'},
            'cost_reduction': {'label': 'Reduced Fertilizer Costs', 'value': '20–30%', 'desc': 'Targeted phosphorus application avoids expensive over-fertilization'},
            'environmental': {'label': 'Environmental Sustainability', 'value': 'High', 'desc': 'Prevents phosphorus runoff that causes algal blooms in water bodies'},
            'soil_health': {'label': 'Improved Soil Health', 'value': 'High', 'desc': 'Enhances phosphorus availability and supports beneficial microbial activity in the root zone'},
            'decision_making': {'label': 'Enhanced Decision Making', 'value': 'ML-Optimized', 'desc': 'Soil-test-based phosphorus recommendation ensures each field gets exactly what it needs'}
        }
    },
    'MOP': {
        'type': 'Potassic',
        'npk_ratio': '0-0-60',
        'application_rate': '50–100 kg/ha',
        'application_method': 'Basal application',
        'best_season': 'Pre-sowing / Early growth',
        'description': 'Rich potassium source that strengthens stems and improves disease resistance.',
        'tips': [
            'Apply during soil preparation or early vegetative stage',
            'Essential for fruit-bearing crops and root vegetables',
            'Avoid over-application in saline soils',
            'Combine with nitrogen fertilizers for balanced nutrition'
        ],
        'dosage': {
            'Rice':         {'per_acre': '20–35 kg', 'splits': 2, 'schedule': '50% basal, 50% at panicle initiation', 'total_per_season': '50–85 kg/ha'},
            'Wheat':        {'per_acre': '15–25 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '40–60 kg/ha'},
            'Maize':        {'per_acre': '20–30 kg', 'splits': 2, 'schedule': '50% basal, 50% at tasseling', 'total_per_season': '50–75 kg/ha'},
            'Sugarcane':    {'per_acre': '30–40 kg', 'splits': 2, 'schedule': '50% at planting, 50% at earthing up', 'total_per_season': '75–100 kg/ha'},
            'Cotton':       {'per_acre': '20–30 kg', 'splits': 2, 'schedule': '50% basal, 50% at flowering', 'total_per_season': '50–75 kg/ha'},
            'Tobacco':      {'per_acre': '25–35 kg', 'splits': 1, 'schedule': 'Full dose as basal before transplanting', 'total_per_season': '60–85 kg/ha'},
            'Millets':      {'per_acre': '10–20 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '25–50 kg/ha'},
            'Pulses':       {'per_acre': '10–15 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '25–40 kg/ha'},
            'Oil Seeds':    {'per_acre': '10–20 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '25–50 kg/ha'},
            'Ground Nuts':  {'per_acre': '15–25 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '40–60 kg/ha'},
        },
        'expected_outcomes': {
            'crop_yield': {'label': 'Increased Crop Yield', 'value': '15–25%', 'desc': 'Stronger stems, better drought tolerance, and improved fruit quality increase marketable yield'},
            'cost_reduction': {'label': 'Reduced Fertilizer Costs', 'value': '10–20%', 'desc': 'Optimized potassium dosing prevents expensive over-application'},
            'environmental': {'label': 'Environmental Sustainability', 'value': 'Moderate', 'desc': 'Balanced potassium use reduces salt buildup and maintains soil structure'},
            'soil_health': {'label': 'Improved Soil Health', 'value': 'High', 'desc': 'Potassium improves water retention, enzyme activation, and overall soil fertility'},
            'decision_making': {'label': 'Enhanced Decision Making', 'value': 'ML-Optimized', 'desc': 'Precise K-level analysis prevents both deficiency symptoms and excess salt stress'}
        }
    },
    'NPK 10-26-26': {
        'type': 'Complex',
        'npk_ratio': '10-26-26',
        'application_rate': '125–175 kg/ha',
        'application_method': 'Basal application / Broadcasting',
        'best_season': 'Pre-sowing',
        'description': 'Balanced complex fertilizer with emphasis on phosphorus and potassium.',
        'tips': [
            'Apply during field preparation for uniform distribution',
            'Best suited for crops that need balanced P and K supplementation',
            'Follow up with urea top-dressing for complete nutrition',
            'Works well with drip irrigation systems'
        ],
        'dosage': {
            'Rice':         {'per_acre': '50–65 kg', 'splits': 1, 'schedule': 'Full dose as basal during puddling', 'total_per_season': '125–160 kg/ha'},
            'Wheat':        {'per_acre': '55–70 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '135–175 kg/ha'},
            'Maize':        {'per_acre': '50–65 kg', 'splits': 1, 'schedule': 'Full dose as basal at planting', 'total_per_season': '125–160 kg/ha'},
            'Sugarcane':    {'per_acre': '55–70 kg', 'splits': 1, 'schedule': 'Full dose in furrow at planting', 'total_per_season': '135–175 kg/ha'},
            'Cotton':       {'per_acre': '50–60 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '125–150 kg/ha'},
            'Tobacco':      {'per_acre': '40–50 kg', 'splits': 1, 'schedule': 'Full dose as basal before transplanting', 'total_per_season': '100–125 kg/ha'},
            'Millets':      {'per_acre': '35–45 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '85–110 kg/ha'},
            'Pulses':       {'per_acre': '40–55 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '100–135 kg/ha'},
            'Oil Seeds':    {'per_acre': '35–50 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '85–125 kg/ha'},
            'Ground Nuts':  {'per_acre': '45–55 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '110–135 kg/ha'},
        },
        'expected_outcomes': {
            'crop_yield': {'label': 'Increased Crop Yield', 'value': '25–40%', 'desc': 'Complete P and K supplementation supports root strength, flowering, and grain filling'},
            'cost_reduction': {'label': 'Reduced Fertilizer Costs', 'value': '20–35%', 'desc': 'Single balanced application reduces the need for multiple separate fertilizer purchases'},
            'environmental': {'label': 'Environmental Sustainability', 'value': 'High', 'desc': 'Reduced total applications means less soil compaction and lower carbon footprint from field operations'},
            'soil_health': {'label': 'Improved Soil Health', 'value': 'High', 'desc': 'Balanced nutrient replenishment prevents nutrient mining and maintains long-term fertility'},
            'decision_making': {'label': 'Enhanced Decision Making', 'value': 'ML-Optimized', 'desc': 'AI-driven NPK ratio selection matches exact soil deficiency patterns for maximum efficiency'}
        }
    },
    'NPK 20-20-20': {
        'type': 'Complex (Balanced)',
        'npk_ratio': '20-20-20',
        'application_rate': '100–150 kg/ha',
        'application_method': 'Fertigation / Foliar spray',
        'best_season': 'Throughout growth',
        'description': 'Perfectly balanced fertilizer suitable for all-round crop nutrition.',
        'tips': [
            'Ideal as a water-soluble fertilizer for fertigation systems',
            'Can be applied as foliar spray at 0.5–1% concentration',
            'Suitable for vegetable and horticultural crops',
            'Apply during active growth stages for maximum uptake'
        ],
        'dosage': {
            'Rice':         {'per_acre': '40–55 kg', 'splits': 3, 'schedule': '33% basal, 33% at tillering, 33% at booting', 'total_per_season': '100–135 kg/ha'},
            'Wheat':        {'per_acre': '40–55 kg', 'splits': 3, 'schedule': '33% basal, 33% at CRI, 33% at heading', 'total_per_season': '100–135 kg/ha'},
            'Maize':        {'per_acre': '45–60 kg', 'splits': 3, 'schedule': '33% basal, 33% at V6, 33% at tasseling', 'total_per_season': '110–150 kg/ha'},
            'Sugarcane':    {'per_acre': '50–65 kg', 'splits': 4, 'schedule': '25% at planting, then at 30/60/90 days', 'total_per_season': '125–160 kg/ha'},
            'Cotton':       {'per_acre': '40–55 kg', 'splits': 3, 'schedule': '33% basal, 33% at squaring, 33% at boll formation', 'total_per_season': '100–135 kg/ha'},
            'Tobacco':      {'per_acre': '30–40 kg', 'splits': 2, 'schedule': '50% basal, 50% at 30 days via fertigation', 'total_per_season': '75–100 kg/ha'},
            'Millets':      {'per_acre': '25–35 kg', 'splits': 2, 'schedule': '50% basal, 50% at 30 days', 'total_per_season': '60–85 kg/ha'},
            'Pulses':       {'per_acre': '25–35 kg', 'splits': 2, 'schedule': '50% basal, 50% at flowering', 'total_per_season': '60–85 kg/ha'},
            'Oil Seeds':    {'per_acre': '25–40 kg', 'splits': 2, 'schedule': '50% basal, 50% at flowering', 'total_per_season': '60–100 kg/ha'},
            'Ground Nuts':  {'per_acre': '30–40 kg', 'splits': 2, 'schedule': '50% basal, 50% at pegging', 'total_per_season': '75–100 kg/ha'},
        },
        'expected_outcomes': {
            'crop_yield': {'label': 'Increased Crop Yield', 'value': '20–35%', 'desc': 'Balanced nutrition throughout all growth stages ensures consistent, high-quality output'},
            'cost_reduction': {'label': 'Reduced Fertilizer Costs', 'value': '25–40%', 'desc': 'One-product fertigation replaces multiple separate fertilizer applications'},
            'environmental': {'label': 'Environmental Sustainability', 'value': 'Very High', 'desc': 'Foliar and drip application minimizes runoff and ensures near-complete nutrient uptake'},
            'soil_health': {'label': 'Improved Soil Health', 'value': 'High', 'desc': 'Even nutrient supply prevents deficiency stress and promotes healthy root microbiome'},
            'decision_making': {'label': 'Enhanced Decision Making', 'value': 'ML-Optimized', 'desc': 'Continuous monitoring recommendations adjust dosing through the crop lifecycle'}
        }
    },
    'SSP': {
        'type': 'Phosphatic',
        'npk_ratio': '0-16-0 (+11% S)',
        'application_rate': '200–300 kg/ha',
        'application_method': 'Basal application',
        'best_season': 'Pre-sowing',
        'description': 'Provides both phosphorus and sulphur, excellent for oilseed crops.',
        'tips': [
            'Rich source of both phosphorus and sulphur for oilseed crops',
            'Apply during final ploughing for thorough mixing',
            'Particularly beneficial for groundnut, mustard, and soybean',
            'Does not acidify soil — safe for all soil pH levels'
        ],
        'dosage': {
            'Rice':         {'per_acre': '80–110 kg', 'splits': 1, 'schedule': 'Full dose as basal during puddling', 'total_per_season': '200–275 kg/ha'},
            'Wheat':        {'per_acre': '85–120 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '210–300 kg/ha'},
            'Maize':        {'per_acre': '75–100 kg', 'splits': 1, 'schedule': 'Full dose as basal at planting', 'total_per_season': '185–250 kg/ha'},
            'Sugarcane':    {'per_acre': '90–120 kg', 'splits': 1, 'schedule': 'Full dose in furrow at planting', 'total_per_season': '225–300 kg/ha'},
            'Cotton':       {'per_acre': '70–95 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '175–235 kg/ha'},
            'Tobacco':      {'per_acre': '60–80 kg', 'splits': 1, 'schedule': 'Full dose as basal before transplanting', 'total_per_season': '150–200 kg/ha'},
            'Millets':      {'per_acre': '50–70 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '125–175 kg/ha'},
            'Pulses':       {'per_acre': '80–100 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '200–250 kg/ha'},
            'Oil Seeds':    {'per_acre': '90–120 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '225–300 kg/ha'},
            'Ground Nuts':  {'per_acre': '100–120 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '250–300 kg/ha'},
        },
        'expected_outcomes': {
            'crop_yield': {'label': 'Increased Crop Yield', 'value': '15–25%', 'desc': 'Dual phosphorus + sulphur supply boosts oil content and seed quality in oilseed crops'},
            'cost_reduction': {'label': 'Reduced Fertilizer Costs', 'value': '15–20%', 'desc': 'Combined P + S in one product eliminates the need for separate sulphur supplements'},
            'environmental': {'label': 'Environmental Sustainability', 'value': 'High', 'desc': 'Slow-release phosphorus reduces runoff risk compared to concentrated phosphatic fertilizers'},
            'soil_health': {'label': 'Improved Soil Health', 'value': 'Very High', 'desc': 'Sulphur improves soil structure, and SSP is pH-neutral — safe for all soil types'},
            'decision_making': {'label': 'Enhanced Decision Making', 'value': 'ML-Optimized', 'desc': 'Soil-specific P and S analysis ensures optimal nutrient matching for oilseed and pulse crops'}
        }
    },
    'Ammonium Sulphate': {
        'type': 'Nitrogenous (+Sulphur)',
        'npk_ratio': '21-0-0 (+24% S)',
        'application_rate': '75–125 kg/ha',
        'application_method': 'Top dressing / Side dressing',
        'best_season': 'During growth stages',
        'description': 'Dual-purpose fertilizer providing nitrogen and sulphur for crop growth.',
        'tips': [
            'Preferred for sulphur-loving crops like onions and garlic',
            'Can be applied as top dressing during vegetative growth',
            'Works well in alkaline soils as it has a slightly acidifying effect',
            'Avoid mixing with Calcium-based fertilizers'
        ],
        'dosage': {
            'Rice':         {'per_acre': '30–45 kg', 'splits': 2, 'schedule': '50% basal, 50% at tillering', 'total_per_season': '75–110 kg/ha'},
            'Wheat':        {'per_acre': '30–45 kg', 'splits': 2, 'schedule': '50% basal, 50% at CRI stage', 'total_per_season': '75–110 kg/ha'},
            'Maize':        {'per_acre': '35–50 kg', 'splits': 2, 'schedule': '50% basal, 50% at knee-high', 'total_per_season': '85–125 kg/ha'},
            'Sugarcane':    {'per_acre': '40–55 kg', 'splits': 2, 'schedule': '50% at planting, 50% at 45 days', 'total_per_season': '100–135 kg/ha'},
            'Cotton':       {'per_acre': '25–40 kg', 'splits': 2, 'schedule': '50% basal, 50% at squaring', 'total_per_season': '60–100 kg/ha'},
            'Tobacco':      {'per_acre': '30–45 kg', 'splits': 2, 'schedule': '50% basal, 50% at 30 days', 'total_per_season': '75–110 kg/ha'},
            'Millets':      {'per_acre': '20–30 kg', 'splits': 2, 'schedule': '50% basal, 50% at 25 days', 'total_per_season': '50–75 kg/ha'},
            'Pulses':       {'per_acre': '15–25 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '40–60 kg/ha'},
            'Oil Seeds':    {'per_acre': '25–35 kg', 'splits': 2, 'schedule': '50% basal, 50% at flowering', 'total_per_season': '60–85 kg/ha'},
            'Ground Nuts':  {'per_acre': '20–30 kg', 'splits': 1, 'schedule': 'Full dose as basal at sowing', 'total_per_season': '50–75 kg/ha'},
        },
        'expected_outcomes': {
            'crop_yield': {'label': 'Increased Crop Yield', 'value': '15–25%', 'desc': 'Combined N + S improves protein synthesis, chlorophyll production, and overall crop vigor'},
            'cost_reduction': {'label': 'Reduced Fertilizer Costs', 'value': '15–25%', 'desc': 'Dual-nutrient formula means fewer separate applications and lower labor costs'},
            'environmental': {'label': 'Environmental Sustainability', 'value': 'Moderate', 'desc': 'Slightly acidifying effect is beneficial in alkaline soils but should be monitored in acidic conditions'},
            'soil_health': {'label': 'Improved Soil Health', 'value': 'Moderate', 'desc': 'Sulphur enhances nutrient availability; acidifying effect can help unlock micronutrients in high-pH soils'},
            'decision_making': {'label': 'Enhanced Decision Making', 'value': 'ML-Optimized', 'desc': 'Intelligent pairing of N and S based on soil pH and crop requirements for maximum efficiency'}
        }
    }
}


# ─── Static File Serving ────────────────────────────────────────────────────
@app.route('/')
def serve_index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    # Security: only serve allowed frontend files, prevent directory traversal
    allowed_files = ['app.js', 'style.css', 'index.html']
    if filename in allowed_files:
        return send_from_directory(BASE_DIR, filename)
    return jsonify({'error': 'Forbidden access'}), 403


# ─── Prediction Endpoint ────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Run "python train_model.py" first.'
        }), 503

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'Invalid or missing JSON payload'}), 400

        # Validate required fields
        required = ['temperature', 'humidity', 'moisture', 'nitrogen',
                     'phosphorus', 'potassium', 'soil_type', 'crop_type']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

        # Parse numeric values
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        moisture = float(data['moisture'])
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        soil_type = data['soil_type']
        crop_type = data['crop_type']

        # Encode categorical features
        try:
            soil_encoded = label_encoders['Soil_Type'].transform([soil_type])[0]
        except ValueError:
            valid = list(label_encoders['Soil_Type'].classes_)
            return jsonify({'error': f'Invalid soil_type. Valid options: {valid}'}), 400

        try:
            crop_encoded = label_encoders['Crop_Type'].transform([crop_type])[0]
        except ValueError:
            valid = list(label_encoders['Crop_Type'].classes_)
            return jsonify({'error': f'Invalid crop_type. Valid options: {valid}'}), 400

        # Build feature array as a DataFrame matching training format
        features = pd.DataFrame(
            [[temperature, humidity, moisture, nitrogen, phosphorus, potassium, soil_encoded, crop_encoded]],
            columns=metadata['feature_columns']
        )

        # Predict
        prediction = model.predict(features)[0]
        fertilizer_name = fertilizer_encoder.inverse_transform([prediction])[0]

        # Get confidence (probability)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = round(float(np.max(probabilities)) * 100, 1)
        else:
            confidence = round(metadata['accuracy'], 1)

        # Look up fertilizer info
        info = FERTILIZER_INFO.get(fertilizer_name, {})

        # Look up crop-specific dosage
        dosage_info = info.get('dosage', {}).get(crop_type, {})

        return jsonify({
            'success': True,
            'prediction': {
                'fertilizer': fertilizer_name,
                'type': info.get('type', 'Unknown'),
                'npk_ratio': info.get('npk_ratio', 'N/A'),
                'application_rate': info.get('application_rate', 'N/A'),
                'application_method': info.get('application_method', 'N/A'),
                'best_season': info.get('best_season', 'N/A'),
                'description': info.get('description', ''),
                'confidence': confidence,
                'tips': info.get('tips', []),
                'expected_outcomes': info.get('expected_outcomes', {}),
                'dosage_recommendation': {
                    'crop': crop_type,
                    'per_acre': dosage_info.get('per_acre', 'N/A'),
                    'splits': dosage_info.get('splits', 1),
                    'schedule': dosage_info.get('schedule', 'N/A'),
                    'total_per_season': dosage_info.get('total_per_season', 'N/A'),
                    'general_rate': info.get('application_rate', 'N/A')
                }
            },
            'input': {
                'temperature': temperature,
                'humidity': humidity,
                'moisture': moisture,
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'soil_type': soil_type,
                'crop_type': crop_type
            }
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input values: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# ─── Model Info Endpoint ────────────────────────────────────────────────────
@app.route('/model-info', methods=['GET'])
def model_info():
    if metadata is None:
        return jsonify({'error': 'Model not trained yet.'}), 503

    return jsonify({
        'model_name': metadata['best_model'],
        'accuracy': metadata['accuracy'],
        'features': metadata['feature_columns'],
        'soil_types': metadata['soil_types'],
        'crop_types': metadata['crop_types'],
        'fertilizer_types': metadata['fertilizer_types'],
        'all_model_results': metadata['all_results']
    })


# ─── Run Server ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\nAgriSense - Fertilizer Recommendation Server")
    print("-" * 50)
    print("  http://localhost:5000")
    print("-" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
