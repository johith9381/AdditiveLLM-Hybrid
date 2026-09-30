"""
AdditiveLLM Persistent Web Dashboard and Live Inference Server.
Features:
- Multi-page visualization of paper tables (Tables 1, 2, 4) and figures (Figs. 5-12)
- Dedicated interactive sections for AUDIT 1 (Frozen Weights) and AUDIT 2 (Data Leakage)
- Guided user interface for live defect classification (Baseline and Prompt modes)
- Fully disk-persistent: loads all evaluation metrics and artifacts directly from disk.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

app = FastAPI(
    title="AdditiveLLM Reproduction & Audit Dashboard",
    description="Interactive evaluation, live inference, and audit interface for AdditiveLLM",
    version="1.0.0"
)

# Health endpoint
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AdditiveLLM Dashboard",
        "reports_available": REPORTS_DIR.exists(),
        "figures_available": FIGURES_DIR.exists(),
    }

# API routes for dynamic data loading from disk
@app.get("/api/audit1")
def get_audit1():
    path = REPORTS_DIR / "audit1_frozen_weights.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"error": "Audit 1 results not yet generated."}

@app.get("/api/audit2")
def get_audit2():
    path = REPORTS_DIR / "audit2_data_leakage.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"error": "Audit 2 results not yet generated."}

@app.get("/api/tables/{table_name}")
def get_table(table_name: str):
    path = TABLES_DIR / f"{table_name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"error": f"Table {table_name} not found."}

@app.get("/api/checkpoints")
def get_checkpoints_report():
    path = REPORTS_DIR / "checkpoint_availability_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"error": "Checkpoint availability report not found."}

@app.get("/figures/{filename}")
def get_figure_file(filename: str):
    path = FIGURES_DIR / filename
    if path.exists():
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": "Figure not found"})

class InferenceRequest(BaseModel):
    mode: str  # "baseline" or "prompt"
    model_name: str  # "distilbert", "scibert", "t5", "llama"
    material: str
    power: float
    velocity: float
    beam_diameter: float
    hatch_spacing: float
    layer_height: float
    template_idx: Optional[int] = 0

@app.post("/api/predict")
def predict_defects(req: InferenceRequest):
    """
    Live prediction using the loaded model classification head and physics formulas.
    """
    from pipeline.preprocessing import (
        format_baseline_text,
        check_keyhole,
        check_lack_of_fusion,
        check_balling,
        format_prompt_text
    )

    # Format input string
    if req.mode == "baseline":
        formatted_input = format_baseline_text(
            req.material, req.power, req.velocity,
            req.beam_diameter, req.hatch_spacing, req.layer_height
        )
    else:
        # Load sample prompt template
        val_prompts_file = BASE_DIR / "data" / "templates" / "validation_prompts.json"
        template = "What defects would occur in <material> L-PBF at <power>, <beam_diameter> beam, <velocity> speed, <hatch_spacing> hatch, <layer_height> layer?"
        if val_prompts_file.exists():
            with open(val_prompts_file) as f:
                prompts = json.load(f)
                template = prompts.get(str(req.template_idx), template)
        
        params = {
            "material": str(req.material),
            "power": f"{req.power} W",
            "velocity": f"{req.velocity} mm/s",
            "beam_diameter": f"{req.beam_diameter} microns",
            "hatch_spacing": f"{req.hatch_spacing} microns",
            "layer_height": f"{req.layer_height} microns",
        }
        formatted_input = format_prompt_text(template, params)

    # Try live PyTorch model inference if weights are cached
    model_prediction = None
    try:
        from pipeline.inference_engine import run_live_inference
        model_prediction = run_live_inference(req.model_name, req.mode, formatted_input)
    except Exception as e:
        model_prediction = {"status": "fallback", "note": f"Model inference engine fallback: {str(e)}"}

    # Analytical physics baseline comparison
    # We estimate melt pool geometry using standard Rosenthal/Eagar-Tsai approximations for AM
    # or directly apply known ratio thresholds
    # E.g. width ~ beam_diameter * sqrt(power / velocity)
    analytical_kh = req.power / (req.velocity * (req.beam_diameter + 1e-5)) > 0.05
    analytical_lof = (req.hatch_spacing / (req.beam_diameter * 1.5)) ** 2 + (req.layer_height / (req.beam_diameter * 1.2)) ** 2 > 1.0
    analytical_bal = (req.velocity / (req.power + 1e-5)) > 1.2

    return {
        "input_parameters": {
            "material": req.material,
            "power_w": req.power,
            "velocity_mm_s": req.velocity,
            "beam_diameter_um": req.beam_diameter,
            "hatch_spacing_um": req.hatch_spacing,
            "layer_height_um": req.layer_height,
            "mode": req.mode,
            "model_selected": req.model_name
        },
        "formatted_input_string": formatted_input,
        "model_prediction": model_prediction,
        "physics_criteria_analysis": {
            "keyhole_risk": bool(analytical_kh),
            "lack_of_fusion_risk": bool(analytical_lof),
            "balling_risk": bool(analytical_bal),
            "defect_free": bool(not (analytical_kh or analytical_lof or analytical_bal))
        }
    }

# Main HTML Single Page Dashboard
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_file = BASE_DIR / "dashboard" / "templates" / "index.html"
    if html_file.exists():
        with open(html_file) as f:
            return f.read()
    return "<h1>AdditiveLLM Dashboard loading... Please ensure templates/index.html is created.</h1>"
