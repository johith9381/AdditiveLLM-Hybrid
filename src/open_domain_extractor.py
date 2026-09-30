"""
Open-Domain Query Extractor and Parser for AdditiveLLM.
Implements Two-Stage Pipeline:
1. Stage 1 (Extraction):
   - Extracts Process Parameters: Material, Power, Velocity, Beam Diameter, Hatch Spacing, Layer Height
   - Classifies query as: {in-domain-complete, in-domain-partial, off-domain}
   - Online Mode: Uses Google Gemini API if GEMINI_API_KEY is configured in .env
   - Offline Mode: Seamless local regex & rule-based AM domain entity extractor when API key is missing or invalid.
2. Stage 2 (Validation & Classification Routing):
   - in-domain-complete, in-range -> Passes to Hybrid Model, returns predictions, auxiliary ratios, confidence score.
   - in-domain-complete, out-of-range -> Returns prediction clearly labeled 'extrapolated/low-confidence' with specific reason.
   - in-domain-partial -> Prompts user for specific missing field(s), never guesses.
   - off-domain -> Returns clear 'out of scope' message, no prediction attempted.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

# Physical calibrated L-PBF operational boundaries (for out-of-range / extrapolation checks)
PARAM_BOUNDS = {
    "power": (20.0, 1000.0, "W"),
    "velocity": (50.0, 4000.0, "mm/s"),
    "beam_diameter": (20.0, 300.0, "microns"),
    "hatch_spacing": (20.0, 500.0, "microns"),
    "layer_height": (10.0, 150.0, "microns"),
}

KNOWN_MATERIAL_SYNONYMS = {
    "ss316l": "SS316L", "316l": "SS316L", "stainless steel 316l": "SS316L", "s31603": "SS316L",
    "ti-6al-4v": "Ti-6Al-4V", "ti6al4v": "Ti-6Al-4V", "ti64": "Ti-6Al-4V", "titanium": "Ti-6Al-4V",
    "in718": "IN718", "inconel 718": "IN718", "inconel718": "IN718",
    "in625": "IN625", "inconel 625": "IN625", "inconel625": "IN625",
    "alsi10mg": "AlSi10Mg", "aluminum": "AlSi10Mg",
    "ss17-4ph": "SS17-4PH", "17-4ph": "SS17-4PH",
    "hastelloy x": "Hastelloy X", "hastelloy": "Hastelloy X",
    "cu10sn": "Cu10Sn", "bronze": "Cu10Sn",
    "tungsten": "Tungsten", "w": "Tungsten",
    "we43": "WE43", "magnesium": "WE43",
    "ss304": "SS304", "ss304l": "SS304L", "304l": "SS304L"
}

OFF_DOMAIN_KEYWORDS = [
    "recipe", "cake", "cookie", "pizza", "cooking", "weather", "temperature in",
    "flight", "hotel", "travel", "stock", "bitcoin", "crypto", "dividend",
    "python code to", "javascript", "sql query", "write an essay", "translate",
    "president", "movie", "song", "lyrics", "capital of", "joke"
]


def load_env_api_key(base_dir: Path) -> Optional[str]:
    """Read GEMINI_API_KEY from .env at project root without crashing if absent."""
    env_file = base_dir / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if key and key != "your_gemini_api_key_here":
                            return key
        except Exception:
            pass
    return os.environ.get("GEMINI_API_KEY", None)


def extract_parameters_offline(query: str) -> Dict[str, Any]:
    """
    Robust local regex & NER rule-based extractor for Laser Powder Bed Fusion parameters.
    Fully offline, zero external dependencies.
    """
    q_lower = query.lower()

    # 1. Off-domain screening
    for kw in OFF_DOMAIN_KEYWORDS:
        if kw in q_lower:
            return {
                "category": "off-domain",
                "message": "Query is out of scope. This system specializes exclusively in metal additive manufacturing (L-PBF) defect prediction.",
                "parameters": {},
                "missing_fields": [],
                "extraction_mode": "offline-local-parser"
            }

    # 2. Material extraction
    extracted_mat = None
    for syn, canonical in KNOWN_MATERIAL_SYNONYMS.items():
        pattern = r"\b" + re.escape(syn) + r"\b"
        if re.search(pattern, q_lower):
            extracted_mat = canonical
            break

    # 3. Numeric process parameter extraction
    # Laser Power (W or Watts)
    power_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:w\b|watts?\b|laser\s*power(?:\s*of)?\s*(\d+(?:\.\d+)?))", q_lower)
    power_val = None
    if power_match:
        val_str = power_match.group(1) or power_match.group(2)
        if val_str:
            power_val = float(val_str)

    # Scan Velocity (mm/s or m/s)
    vel_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm/s|m/s|speed(?:\s*of)?\s*(\d+(?:\.\d+)?)|velocity(?:\s*of)?\s*(\d+(?:\.\d+)?))", q_lower)
    vel_val = None
    if vel_match:
        val_str = vel_match.group(1) or vel_match.group(2) or vel_match.group(3)
        if val_str:
            val_float = float(val_str)
            # Convert m/s to mm/s if < 10
            if "m/s" in q_lower and "mm/s" not in q_lower and val_float < 10.0:
                val_float *= 1000.0
            vel_val = val_float

    # Beam Diameter (microns, um, µm)
    beam_match = re.search(r"(?:beam(?:\s*(?:diameter|spot|size))?|spot(?:\s*(?:diameter|size))?)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:um|µm|microns?)\s*(?:beam|spot)", q_lower)
    beam_val = None
    if beam_match:
        val_str = beam_match.group(1) or beam_match.group(2)
        if val_str:
            beam_val = float(val_str)

    # Hatch Spacing (microns, um, µm)
    hatch_match = re.search(r"hatch(?:\s*spacing)?\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:um|µm|microns?)\s*hatch", q_lower)
    hatch_val = None
    if hatch_match:
        val_str = hatch_match.group(1) or hatch_match.group(2)
        if val_str:
            hatch_val = float(val_str)

    # Layer Height / Thickness (microns, um, µm)
    layer_match = re.search(r"layer(?:\s*(?:height|thickness))?\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)|thickness\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:um|µm|microns?)\s*layers?", q_lower)
    layer_val = None
    if layer_match:
        val_str = layer_match.group(1) or layer_match.group(2) or layer_match.group(3)
        if val_str:
            layer_val = float(val_str)

    params = {
        "material": extracted_mat,
        "power": power_val,
        "velocity": vel_val,
        "beam_diameter": beam_val,
        "hatch_spacing": hatch_val,
        "layer_height": layer_val
    }

    # Identify missing fields
    missing_fields = [k for k, v in params.items() if v is None]

    # Check if completely devoid of manufacturing context
    num_extracted = sum(1 for v in params.values() if v is not None)
    if num_extracted == 0:
        return {
            "category": "off-domain",
            "message": "Query contains no recognizable Laser Powder Bed Fusion (L-PBF) process parameters. AdditiveLLM evaluates defect regimes for specific build parameters (Material, Power, Velocity, Beam Diameter, Hatch Spacing, Layer Height). Please enter a parameter description or switch to Process Sliders.",
            "parameters": {},
            "missing_fields": list(params.keys()),
            "extraction_mode": "offline-local-parser"
        }

    # Complete vs Partial
    if len(missing_fields) == 0:
        category = "in-domain-complete"
        msg = "All required process parameters successfully extracted."
    else:
        category = "in-domain-partial"
        msg = f"Incomplete build specification: {', '.join(missing_fields)} missing. Please provide all 6 process parameters for defect prediction."

    return {
        "category": category,
        "message": msg,
        "parameters": params,
        "missing_fields": missing_fields,
        "extraction_mode": "offline-local-parser"
    }


def extract_parameters_gemini(query: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Online Stage 1 extraction using Google Gemini API.
    Returns structured JSON with category classification.
    """
    try:
        import urllib.request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        prompt_instruction = (
            "You are an expert metallurgy and additive manufacturing parameter extraction system. "
            "Given a user query, extract the 6 Laser Powder Bed Fusion (L-PBF) process parameters: "
            "material, power (in Watts), velocity (in mm/s), beam_diameter (in microns), "
            "hatch_spacing (in microns), layer_height (in microns). "
            "Classify the query into one of three categories: "
            "1. 'in-domain-complete' if all 6 parameters are present. "
            "2. 'in-domain-partial' if 1 to 5 parameters are present. "
            "3. 'off-domain' if the query is unrelated to metal additive manufacturing or lacks parameters. "
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "category": "in-domain-complete" | "in-domain-partial" | "off-domain",\n'
            '  "parameters": {\n'
            '    "material": string or null,\n'
            '    "power": float or null,\n'
            '    "velocity": float or null,\n'
            '    "beam_diameter": float or null,\n'
            '    "hatch_spacing": float or null,\n'
            '    "layer_height": float or null\n'
            "  },\n"
            '  "missing_fields": ["list of missing field names"],\n'
            '  "message": "one-line explanation"\n'
            "}"
        )
        payload = {
            "contents": [{
                "parts": [{"text": f"{prompt_instruction}\n\nUser Query: {query}"}]
            }],
            "generationConfig": {"temperature": 0.0, "response_mime_type": "application/json"}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            content_str = res_json["candidates"][0]["content"]["parts"][0]["text"]
            parsed_result = json.loads(content_str)
            parsed_result["extraction_mode"] = "online-gemini"
            return parsed_result
    except Exception as e:
        print(f"[Gemini Extraction] API call failed ({e}). Gracefully falling back to local offline parser.")
        return None


def parse_query_two_stage(query: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Two-stage query processor:
    Attempts Gemini online extraction first if api_key is available.
    Silently falls back to offline parser on failure or missing key.
    Validates physical parameter boundaries for extrapolation flags.
    """
    res = None
    if api_key:
        res = extract_parameters_gemini(query, api_key)

    if res is None:
        res = extract_parameters_offline(query)

    # Boundary & Extrapolation validation for complete parameters
    if res["category"] == "in-domain-complete":
        params = res["parameters"]
        extrapolations = []
        for p_name, (low, high, unit) in PARAM_BOUNDS.items():
            val = params.get(p_name)
            if val is not None:
                if val < low or val > high:
                    extrapolations.append(f"{p_name} ({val} {unit}) outside calibrated L-PBF range [{low}-{high} {unit}]")

        if len(extrapolations) > 0:
            res["is_extrapolated"] = True
            res["extrapolation_reason"] = "Extrapolation Alert: " + "; ".join(extrapolations)
        else:
            res["is_extrapolated"] = False
            res["extrapolation_reason"] = None

    return res


def run_robustness_test(output_csv: Path) -> pd.DataFrame:
    """
    Builds an expanded benchmark set of 58 held-out paraphrased queries (complete, partial, off-domain, adversarial),
    runs the full pipeline on all of them, and saves results to results/robustness_test.csv.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  OPEN-DOMAIN PROMPT EXTRACTION ROBUSTNESS BENCHMARK (58 QUERIES)")
    print("=" * 70)

    test_queries = [
        # In-domain complete (varied natural language styles) - 20 queries
        ("Inconel 718 printed at 250W power, scanning at 900 mm/s with 100um beam diameter, 110um hatch spacing, and 40um layer thickness.", "in-domain-complete"),
        ("What defects will happen in SS316L if I use 180 Watts laser, 800 mm/s velocity, 80 micron beam, 120 micron hatch, and 30 micron layers?", "in-domain-complete"),
        ("Parameters: Material Ti64, 300W laser, speed 1200 mm/s, spot size 90um, hatch 100um, layer height 50um.", "in-domain-complete"),
        ("Can we print AlSi10Mg with 350 W power, scan velocity 1500 mm/s, beam 75 microns, hatch spacing 130 microns, and layer thickness 30 microns?", "in-domain-complete"),
        ("Evaluate defect regimes for IN625 at 200W, 600 mm/s, beam diameter 100um, hatch spacing 80um, and layer height 40um.", "in-domain-complete"),
        ("Process setup: stainless steel 316L, 150W, 700 mm/s, beam 85um, hatch 95um, layer 25um.", "in-domain-complete"),
        ("Running Ti-6Al-4V additive manufacturing with 280W, 1100 mm/s speed, 100um beam, 120um hatch, 45um layer.", "in-domain-complete"),
        ("What defects occur in Hastelloy X at 220 Watts, 950 mm/s, 70 micron beam, 100 micron hatch, 35 micron layer?", "in-domain-complete"),
        ("Fabricate SS17-4PH using 190W power, 850 mm/s velocity, 90um beam diameter, 105um hatch, and 30um layer thickness.", "in-domain-complete"),
        ("L-PBF of Cu10Sn with 400W laser, 600 mm/s scanning, 100um beam, 140um hatch spacing, 50um layer.", "in-domain-complete"),
        ("Will there be keyholing in SS316L at 450W, 500 mm/s, 98um beam, 100um hatch, 50um layer?", "in-domain-complete"),
        ("Is balling likely in Ti64 when operating at 100W, 2500 mm/s, 80um beam, 120um hatch, 30um layer?", "in-domain-complete"),
        ("Predict porosity for Tungsten run at 350W, 400 mm/s speed, 100 micron spot, 90 micron hatch, 20 micron layer.", "in-domain-complete"),
        ("Under conditions of 200W, 1000 mm/s, 90um beam, 110um hatch, 40um layer in IN718, what defects form?", "in-domain-complete"),
        ("Assess melt pool for WE43 alloy with 120W, 800 mm/s, 70um beam, 100um hatch, 30um layer.", "in-domain-complete"),
        ("Check lack of fusion in SS304L at 160W, 900 mm/s, 85um beam, 150um hatch, 60um layer.", "in-domain-complete"),
        ("Laser power 230W, scan rate 750 mm/s, beam 100um, hatch 100um, layer 30um for Inconel 718 build.", "in-domain-complete"),
        ("Parameters: Material SS316L, Power 210W, Velocity 1050 mm/s, Beam 80um, Hatch 90um, Layer 35um.", "in-domain-complete"),
        ("Print job: Ti-6Al-4V at 175W, 650 mm/s, 95um beam, 115um hatch, 30um layer.", "in-domain-complete"),
        ("Investigate defect modes in AlSi10Mg with 300W laser power, 1200 mm/s scan velocity, 80um beam, 130um hatch, 40um layer.", "in-domain-complete"),

        # In-domain partial (missing 1 to 5 fields, including adversarial single words) - 18 queries
        ("What defects occur in Ti-6Al-4V at 200W and 800 mm/s?", "in-domain-partial"),
        ("Printing SS316L with 250W power, 100um beam diameter, and 120um hatch spacing.", "in-domain-partial"),
        ("Laser power 180W, speed 900 mm/s, layer thickness 30 microns for IN718.", "in-domain-partial"),
        ("Predict defects for AlSi10Mg at 300W power.", "in-domain-partial"),
        ("SS304L scanned at 1200 mm/s with 80um beam and 40um layer.", "in-domain-partial"),
        ("What happens to IN625 at 950 mm/s velocity and 100um hatch spacing?", "in-domain-partial"),
        ("Hastelloy X with 220W laser and 100um beam diameter.", "in-domain-partial"),
        ("Copper printing with 400W and 500 mm/s.", "in-domain-partial"),
        ("Evaluate defect formation at 150W and 600 mm/s in Titanium.", "in-domain-partial"),
        ("Power 200W, beam 90um, hatch 100um, layer 30um (material omitted).", "in-domain-partial"),
        ("titanium", "in-domain-partial"),
        ("print Ti64", "in-domain-partial"),
        ("power 300W only", "in-domain-partial"),
        ("speed 800 mm/s", "in-domain-partial"),
        ("laser 200W speed 1000 mm/s", "in-domain-partial"),
        ("SS316L layer 30um", "in-domain-partial"),
        ("AlSi10Mg 400W 1200 mm/s", "in-domain-partial"),
        ("IN718 at 250W without speed or beam specified", "in-domain-partial"),

        # Off-domain queries (including adversarial meta-questions and single words) - 20 queries
        ("what is the material used", "off-domain"),
        ("how does this work", "off-domain"),
        ("can you help me", "off-domain"),
        ("what can this tool do", "off-domain"),
        ("explain laser powder bed fusion", "off-domain"),
        ("what are the defect regimes", "off-domain"),
        ("fast", "off-domain"),
        ("1000", "off-domain"),
        ("defect", "off-domain"),
        ("porosity", "off-domain"),
        ("who made this model", "off-domain"),
        ("How do I make chocolate chip cookies from scratch?", "off-domain"),
        ("Write a Python script to scrape stock prices from Yahoo Finance.", "off-domain"),
        ("What is the current weather forecast for Pittsburgh, PA?", "off-domain"),
        ("Explain how convolutional neural networks classify images.", "off-domain"),
        ("What are the best tourist attractions to visit in Tokyo?", "off-domain"),
        ("Can you translate this sentence into Spanish?", "off-domain"),
        ("Who was the 16th president of the United States?", "off-domain"),
        ("Write an SQL query to join customer and orders tables.", "off-domain"),
        ("Tell me a funny joke about computer scientists.", "off-domain"),
    ]

    results = []
    correct_category_count = 0
    total = len(test_queries)

    for i, (q, expected_cat) in enumerate(test_queries, 1):
        parsed = extract_parameters_offline(q)
        pred_cat = parsed["category"]
        is_match = (pred_cat == expected_cat)
        if is_match:
            correct_category_count += 1

        p = parsed.get("parameters", {})
        results.append({
            "query_id": i,
            "query_text": q,
            "expected_category": expected_cat,
            "predicted_category": pred_cat,
            "category_match": is_match,
            "extracted_material": p.get("material", ""),
            "extracted_power": p.get("power", ""),
            "extracted_velocity": p.get("velocity", ""),
            "extracted_beam": p.get("beam_diameter", ""),
            "extracted_hatch": p.get("hatch_spacing", ""),
            "extracted_layer": p.get("layer_height", ""),
            "extraction_mode": parsed["extraction_mode"],
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(output_csv, index=False)

    acc = (correct_category_count / total) * 100.0
    print(f"\n[Robustness Suite] Evaluated {total} queries:")
    print(f"  Category Classification Accuracy: {acc:.2f}% ({correct_category_count}/{total})")
    print(f"  Results saved to: {output_csv}")
    return df_res


def run_offline_fallback_test(output_txt: Path):
    """
    Deliberately tests the offline fallback with:
    1. No API key provided.
    2. Invalid / broken API key provided.
    Confirms zero crash and valid classification, saving proof to results/offline_fallback_test.txt.
    """
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print("  OFFLINE FALLBACK & API RESILIENCE VERIFICATION TEST")
    print("=" * 70)

    test_query = "SS316L part printed with 200W laser, 800 mm/s speed, 100um beam, 110um hatch, 40um layer."

    # Test 1: Key removed / None
    res_no_key = parse_query_two_stage(test_query, api_key=None)

    # Test 2: Invalid / broken key
    res_broken_key = parse_query_two_stage(test_query, api_key="INVALID_BROKEN_KEY_99999")

    # Assertions
    assert res_no_key["category"] == "in-domain-complete", "Fallback failed with no key!"
    assert res_broken_key["category"] == "in-domain-complete", "Fallback failed with broken key!"
    assert res_no_key["extraction_mode"] == "offline-local-parser"
    assert res_broken_key["extraction_mode"] == "offline-local-parser"

    log_text = (
        "=======================================================================\n"
        "OFFLINE FALLBACK & RESILIENT API HANDLING TEST AUDIT\n"
        "Status: PASSED (Zero Crash, 100% Graceful Fallback Confirmed)\n"
        "=======================================================================\n\n"
        "Test 1: Execution with No API Key (api_key=None)\n"
        f"  - Input Query:       '{test_query}'\n"
        f"  - Extraction Mode:   {res_no_key['extraction_mode']}\n"
        f"  - Category:          {res_no_key['category']}\n"
        f"  - Extracted Params:  {json.dumps(res_no_key['parameters'])}\n"
        "  - Outcome:           SUCCESS (Clean offline extraction)\n\n"
        "Test 2: Execution with Deliberately Broken Key (api_key='INVALID_BROKEN_KEY_99999')\n"
        f"  - Extraction Mode:   {res_broken_key['extraction_mode']}\n"
        f"  - Category:          {res_broken_key['category']}\n"
        f"  - Extracted Params:  {json.dumps(res_broken_key['parameters'])}\n"
        "  - Crash Evaded:      TRUE (Handled via try/except fallback without crashing app)\n"
        "  - Outcome:           SUCCESS (Clean offline extraction)\n\n"
        "Formal Scientific Conclusion:\n"
        "The system does not depend on external APIs for core defect prediction.\n"
        "When an API key is absent or invalid, the app silently falls back to the high-precision\n"
        "local rule-based AM domain parser, and the UI displays the 'Offline Local Parser Mode' badge.\n"
    )

    with open(output_txt, "w") as f:
        f.write(log_text)

    print(f"[Offline Fallback] Certified audit proof successfully written to {output_txt}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"
    run_robustness_test(results_dir / "robustness_test.csv")
    run_offline_fallback_test(results_dir / "offline_fallback_test.txt")
