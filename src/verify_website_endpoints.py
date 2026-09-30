"""
AdditiveLLM Website Endpoints Verification Script.
Tests all endpoints using FastAPI TestClient (direct ASGI invocation)
and prints full responses to verify backend functionality.
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from website.backend import app, load_model_and_tokenizers

def test_endpoints():
    print("=" * 60)
    print("AdditiveLLM API Endpoint Verification")
    print("=" * 60)

    # Ensure models are loaded
    load_model_and_tokenizers()
    client = TestClient(app)

    # 1. Test /api/health
    print("\n1. Testing GET /api/health...")
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health failed: {resp.text}"
    health_data = resp.json()
    print(f"   Status: {resp.status_code} OK")
    print(f"   Payload: {json.dumps(health_data, indent=2)}")

    # 2. Test /api/dashboard_data
    print("\n2. Testing GET /api/dashboard_data...")
    resp = client.get("/api/dashboard_data")
    assert resp.status_code == 200, f"Dashboard failed: {resp.text}"
    dash_data = resp.json()
    print(f"   Status: {resp.status_code} OK")
    print(f"   Keys returned: {list(dash_data.keys())}")
    print(f"   Multiseed Summary v2 rows: {len(dash_data.get('multiseed_summary_v2', []))}")
    print(f"   Overall Pooled v2 rows: {len(dash_data.get('overall_pooled_v2', []))}")
    print(f"   Tier C Diagnostic rows: {len(dash_data.get('tierc_diagnostic', []))}")
    print(f"   Tier C Root Cause rows: {len(dash_data.get('tierc_root_cause_analysis', []))}")

    # 3. Test POST /api/predict (Manual Mode - Conduction)
    print("\n3. Testing POST /api/predict (Manual Conduction Mode)...")
    payload = {
        "mode": "manual",
        "material": "Ti-6Al-4V",
        "power": 200.0,
        "velocity": 1000.0,
        "beam_diameter": 100.0,
        "hatch_spacing": 100.0,
        "layer_height": 30.0
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200, f"Predict failed: {resp.text}"
    p_data = resp.json()
    print(f"   Status: {resp.status_code} OK")
    print(f"   Banner: {p_data.get('banner', {}).get('title')}")
    print(f"   Primary Regime: {p_data.get('process_regime', {}).get('primary_regime')}")
    print(f"   None Prob: {p_data.get('defects', {}).get('none', {}).get('probability')}")
    print(f"   Aspect Ratios: {p_data.get('hydrodynamic_aspect_ratios')}")

    # 4. Test POST /api/predict (Prompt Mode - Keyhole)
    print("\n4. Testing POST /api/predict (Prompt Mode - Keyhole)...")
    payload_kh = {
        "mode": "prompt",
        "text": "SS316L processed at 380 W high laser power, scan velocity 600 mm/s, spot diameter 80 µm, hatch spacing 90 µm, and layer height 30 µm."
    }
    resp_kh = client.post("/api/predict", json=payload_kh)
    assert resp_kh.status_code == 200, f"Predict failed: {resp_kh.text}"
    p_kh = resp_kh.json()
    print(f"   Status: {resp_kh.status_code} OK")
    print(f"   Banner: {p_kh.get('banner', {}).get('title')}")
    print(f"   Banner Type: {p_kh.get('banner', {}).get('type')}")
    print(f"   Suggested Adjustment: {p_kh.get('banner', {}).get('suggested_adjustment')}")
    print(f"   Keyhole Prob: {p_kh.get('defects', {}).get('keyhole', {}).get('probability')}")

    # 5. Test POST /api/predict (Off-Domain Query)
    print("\n5. Testing POST /api/predict (Off-Domain Rejection)...")
    payload_off = {
        "mode": "prompt",
        "text": "what is the material used"
    }
    resp_off = client.post("/api/predict", json=payload_off)
    assert resp_off.status_code == 200, f"Predict failed: {resp_off.text}"
    p_off = resp_off.json()
    print(f"   Status: {resp_off.status_code} OK")
    print(f"   API Response Status: {p_off.get('status')}")
    print(f"   Category: {p_off.get('category')}")
    print(f"   Message: {p_off.get('message')}")
    assert p_off.get("status") == "rejected" or p_off.get("category") == "off-domain", "Failed off-domain check!"

    # 6. Test POST /api/predict (Partial Query)
    print("\n6. Testing POST /api/predict (Partial / Clarification Needed)...")
    payload_part = {
        "mode": "prompt",
        "text": "Ti-6Al-4V printed with laser power 250 W"
    }
    resp_part = client.post("/api/predict", json=payload_part)
    assert resp_part.status_code == 200, f"Predict failed: {resp_part.text}"
    p_part = resp_part.json()
    print(f"   Status: {resp_part.status_code} OK")
    print(f"   API Response Status: {p_part.get('status')}")
    print(f"   Category: {p_part.get('category')}")
    print(f"   Missing Fields: {p_part.get('missing_fields')}")
    assert p_part.get("status") == "clarification_needed", "Failed partial check!"

    print("\n" + "=" * 60)
    print("ALL API ENDPOINTS TESTED AND VERIFIED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_endpoints()
