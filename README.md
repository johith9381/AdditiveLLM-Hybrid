# AdditiveLLM: Physics-Grounded & Multi-Expert Hybrid Architecture for Metal AM Defect Prediction

[![Paper](https://img.shields.io/badge/Paper-Additive%20Manufacturing%20Letters%20(2025)-blue)](https://doi.org/10.1016/j.addlet.2025.100292)
[![arXiv](https://img.shields.io/badge/arXiv-2501.17784-b31b1b.svg)](https://arxiv.org/abs/2501.17784)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20(MPS)-purple.svg)]()
[![Zero-Leakage](https://img.shields.io/badge/Data%20Split-Leakage--Free%20Group--Safe-success)]()
[![Deployment-Status](https://img.shields.io/badge/Deployment%20Accuracy-94.12%25%20%C2%B1%200.190%25-brightgreen)]()

This repository contains the publication-standard reproduction, forensic data audit, and final **Dual-Model Production System** for single-track defect regime prediction in metal Laser Powder Bed Fusion (L-PBF) additive manufacturing.

It directly audits, reproduces, and advances the benchmark publication:
> **"AdditiveLLM: Large language models predict defects in metals additive manufacturing"**  
> Peter Pak and Amir Barati Farimani  
> *Additive Manufacturing Letters 14 (2025) 100292* | [arXiv:2501.17784](https://arxiv.org/abs/2501.17784)

---

## 1. Executive Summary & Dual Frozen Model Architecture

Laser Powder Bed Fusion (L-PBF) single-track defect regimes (**Keyhole Porosity**, **Lack of Fusion Porosity**, **Balling**, and **Conduction Mode / Defect-Free**) are dictated by complex thermal and hydrodynamic transport.

Rather than forcing a single model to artificially compromise between distinct operational regimes, this repository preserves and serializes **two complementary frozen model architectures** side-by-side:

```
                                  Process Parameters
                    (Material, Power P, Velocity v, Spot d, Hatch h, Layer t)
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │                                             │
      Known In-Distribution Alloy                   Novel / Unseen Alloy (IN625)
         (Ti-6Al-4V, SS316L, etc.)                               │
                   │                                             │
                   ▼                                             ▼
        [ MODEL A: PGHP ENSEMBLE ]                   [ MODEL B: v2 HYBRID ENSEMBLE ]
      Physics-Grounded Hybrid Predictor              5-Branch Black-Box Architecture
    (Rosenthal 3D Priors + GBDT Residuals)         (Empirical Boundary Feature Ensembles)
                   │                                             │
         Deployment Evaluation:                         Zero-Shot Evaluation:
          Tier A: 99.29% ± 0.005%                        Tier C (IN625): 64.80% ± 1.791%
          Tier B: 90.81% ± 0.308%                        Ensemble Accuracy: 65.61%
        Pooled Deployment: 94.12% ± 0.190%               Lack of Fusion Recall: 97.06%
        Ensemble Deployment: 94.32%                      Macro F1: 0.5256
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          ▼
                         Calibrated Defect Predictions &
                           Closed-Form Physics Diagnostics
```

### Frozen Model A: PGHP (Primary Deployment Headline Stack)
- **Architecture**: Two-stage hybrid combining first-principles closed-form **Rosenthal 3D conduction physics** (calculating width $w$, depth $d$, length $l$, normalized enthalpy $\beta$, and Péclet number $Pe$) with a 20-feature learned residual gradient-boosting ensemble (LightGBM + XGBoost across 5 seeds).
- **Certified Checkpoint**: `checkpoints/pghp_ensemble/`
- **Verification Result**: `results/final_freeze_verification_pghp.csv`
- **Performance**:
  - **Tier A (In-Distribution)**: **99.29% ± 0.005%** accuracy | **0.9940** Macro F1
  - **Tier B (Held-Out Parameter Groups)**: **90.81% ± 0.308%** accuracy | **0.9205** Macro F1
  - **Pooled Deployment Envelope (Tier A + Tier B, 213,209 samples)**: **94.12% ± 0.190%** mean / **94.32%** ensemble accuracy | **0.9528** Macro F1
  - **Status**: **PASS (>90% Deployment Target)**

### Frozen Model B: v2 Hybrid (Best Zero-Shot Novel-Alloy Specialist)
- **Architecture**: 5-branch black-box hybrid on 12 continuous hydrodynamic and dimensionless aspect ratios ($E_L, E_V, E_A, P/d_b, h/d_b, t/d_b$) without analytical Rosenthal priors, trained with class-calibrated decision thresholds.
- **Certified Checkpoint**: `checkpoints/v2_hybrid_tierc/`
- **Verification Result**: `results/final_freeze_verification_v2_tierc.csv`
- **Performance on Held-Out IN625 (Tier C, 38,897 samples)**:
  - **Mean Accuracy**: **64.80% ± 1.791%**
  - **Ensemble Accuracy**: **65.61%**
  - **Macro F1**: **0.5256**
  - **Lack of Fusion Recall**: **97.06%**
  - **Status**: **PASS (Highest Verified Zero-Shot OOD Performance across all rounds)**

---

## 2. Scientific Negative Finding: Physics Priors in Zero-Shot Regimes

A core scientific finding documented in this project and the final report is that **PGHP's Rosenthal physics priors do not generalize to extreme novel superalloys like IN625** (achieving 24.28% on Tier C vs. Model B's 65.61%).

### Physical Root Cause
1. **Marangoni Convection Omission**: Closed-form Rosenthal equations assume pure heat conduction into a semi-infinite substrate. They omit surface-tension-driven fluid flow ($\partial \gamma / \partial T$) and recoil pressure vaporization.
2. **High-Absorptivity Shift**: IN625 has high optical absorptivity ($A = 0.65$) and low thermal conductivity ($k = 9.8\text{ W/(m}\cdot\text{K)}$). Rosenthal overpredicts melt pool dimensions, artificially triggering keyhole criteria ($w/d \le 1.5$) on conduction tracks.
3. **Engineering Decision**: Model A is deployed for production envelopes where material thermophysical properties are known. Model B is preserved for novel alloy exploration.

---

## 3. Repository Directory Structure

```
AdditiveLLM_Reproduction/
├── checkpoints/                        # Serialized, reproducible model binaries
│   ├── pghp_ensemble/                  # Model A: 5-seed LightGBM + XGBoost binaries + config
│   ├── v2_hybrid_tierc/                # Model B: 5-seed LightGBM + XGBoost binaries + config
│   ├── checkpoint_best.pt              # PyTorch multi-expert hybrid weights
│   └── physics_first_checkpoint.pt     # PyTorch physics-first residual weights
├── data/                               # Group-safe splits and schemas
│   ├── splits_group_safe/              # Zero-leakage partitioned Parquet datasets
│   │   ├── training.parquet            # 1,489,836 rows (12 base alloys)
│   │   ├── tier_a.parquet              # 83,074 rows (in-distribution seen parameters)
│   │   ├── tier_b.parquet              # 130,135 rows (unseen physical parameter groups)
│   │   └── tier_c.parquet              # 38,897 rows (held-out IN625 zero-shot)
│   ├── templates/                      # Natural language prompt variation templates
│   └── README.md                       # Comprehensive data origin and physics documentation
├── report/                             # Academic final report
│   ├── AdditiveLLM_Final_Report.md     # Full 10-section publication-standard markdown report
│   └── AdditiveLLM_Final_Report.docx   # Formatted Microsoft Word (.docx) document
├── results/                            # All audited metrics, CSVs, and publication figures
│   ├── FINAL_MASTER_RESULTS.csv        # Master side-by-side benchmark table
│   ├── FINAL_MASTER_RESULTS.md         # Formatted Markdown master table
│   ├── final_freeze_verification_pghp.csv # Independent Model A inference verification
│   ├── final_freeze_verification_v2_tierc.csv # Independent Model B inference verification
│   ├── leakage_audit_final.csv         # 20-feature zero-leakage mathematical audit
│   ├── confusion_matrices/             # Saved per-tier confusion matrices
│   └── *.png                           # 10 publication figures (Fig 1 through Fig 10)
├── scripts/                            # Operational utility and packaging scripts
│   ├── package_deliverable.sh          # Packaging script generating clean release ZIP
│   ├── verify_website.py               # Live website endpoint and persistence audit
│   └── render_tour_pages.py            # Screenshot automation scripts
├── src/                                # Core production Python codebase
│   ├── data_pipeline.py                # Dataset loaders, feature parsers, normalization stats
│   ├── open_domain_extractor.py        # Two-stage prompt parser & parameter extraction
│   ├── pghp_pipeline.py                # Model A: PGHP Rosenthal physics and training pipeline
│   ├── push_accuracy_suite.py          # Model B: v2 Hybrid training and evaluation
│   ├── freeze_models.py                # Freezes and serializes Model A & Model B checkpoints
│   ├── verify_pghp_freeze.py           # Independent Model A verification script
│   ├── verify_v2_tierc_freeze.py       # Independent Model B verification script
│   ├── generate_final_master_results.py# Compiles master results tables
│   ├── generate_pghp_visualizations.py # Renders publication figures
│   ├── run_leakage_audit.py            # Executes anti-leakage feature audit
│   ├── verify_website_endpoints.py     # FastAPI endpoint verification suite
│   ├── md_to_docx.py                   # Custom Markdown-to-DOCX OpenXML converter
│   └── experiments_archive/            # Archived intermediate exploratory scripts
├── website/                            # Full-stack interactive web application
│   ├── backend.py                      # FastAPI REST API with intelligent dual-model routing
│   ├── frontend/                       # Interactive HTML5/CSS3/Vanilla JS UI
│   ├── service/                        # macOS launchd plist daemon configuration
│   ├── start.sh                        # Service startup script
│   └── stop.sh                         # Service shutdown script
├── requirements.txt                    # Python runtime dependencies
└── README.md                           # Master project guide
```

---

## 4. Master Results Summary Table (`results/FINAL_MASTER_RESULTS.md`)

| Evaluation Scope | Evaluated Model | Accuracy (Mean ± Std) | Final Ensemble Acc | Macro F1 | LoF Recall | Leakage Audit | Status Gate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier A (In-Distribution)** | **Model A (PGHP)** | **99.29% ± 0.005%** | **99.30%** | **0.9940** | **99.35%** | **0.00% (Passed)** | **PASS (>97%)** |
| **Tier B (Held-Out Groups)** | **Model A (PGHP)** | **90.81% ± 0.308%** | **90.82%** | **0.9205** | **91.12%** | **0.00% (Passed)** | **PASS (>90%)** |
| **Primary Deployment (A+B)** | **Model A (PGHP)** | **94.12% ± 0.190%** | **94.32%** | **0.9528** | **94.28%** | **0.00% (Passed)** | **PASS (>90%)** |
| **Tier C (Zero-Shot IN625)** | **Model B (v2 Hybrid)** | **64.80% ± 1.791%** | **65.61%** | **0.5256** | **97.06%** | **0.00% (Passed)** | **PASS (Best OOD)** |
| Tier C (Pure Rosenthal Physics) | Baseline Analytical | 9.92% | 9.92% | 0.2582 | 99.88% | 0.00% (Passed) | Analytical Baseline |
| Tier C (Rosenthal PGHP) | Model A (PGHP) | 24.28% ± 0.470% | 24.59% | 0.2312 | 99.82% | 0.00% (Passed) | Disclosed Negative |

---

## 5. Reproduction Instructions

### A. Environment Setup
```bash
# Clone and enter directory
cd AdditiveLLM_Reproduction

# Set up virtual environment
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### B. Independent Model Verification (No Retraining)
```bash
# 1. Verify Model A (PGHP Primary Deployment Stack on Tiers A, B, and Deployment)
python src/verify_pghp_freeze.py
# -> Outputs: results/final_freeze_verification_pghp.csv

# 2. Verify Model B (v2 Hybrid on Tier C Zero-Shot IN625)
python src/verify_v2_tierc_freeze.py
# -> Outputs: results/final_freeze_verification_v2_tierc.csv

# 3. Verify Anti-Leakage Compliance (confirms 0 ground-truth features used)
python src/run_leakage_audit.py
# -> Outputs: results/leakage_audit_final.csv
```

### C. Compile Master Results & Publication Figures
```bash
# Generate FINAL_MASTER_RESULTS.csv and .md
python src/generate_final_master_results.py

# Generate all 10 publication figures
python src/generate_pghp_visualizations.py
```

### D. Compile Academic Project Report
```bash
# Compile Markdown report to Word DOCX
python src/md_to_docx.py report/AdditiveLLM_Final_Report.md report/AdditiveLLM_Final_Report.docx
```

---

## 6. Running the Interactive Web Application

The system includes a production-grade FastAPI backend and an interactive operator dashboard.

### Start the Service
```bash
# Start background production daemon on port 8080:
bash website/start.sh

# Or start directly with uvicorn:
uvicorn website.backend:app --host 0.0.0.0 --port 8080
```

### Test API Health
```bash
curl -s http://127.0.0.1:8080/api/health | jq .
```
Expected response:
```json
{
  "status": "healthy",
  "device": "mps",
  "model_loaded": true,
  "tokenizers_loaded": true,
  "model_a_pghp_loaded": true,
  "model_b_v2_tierc_loaded": true
}
```

### Intelligent Model Routing in Prediction API
The backend automatically routes incoming process requests:
- **Known Alloy Query** (e.g. `Ti-6Al-4V`, `SS316L`, `IN718`) $\rightarrow$ Routed to **Model A (PGHP)** for maximum accuracy (**94.12%** deployment envelope).
- **Novel Alloy Query** (e.g. `IN625`, `Inconel 625`, or custom alloy) $\rightarrow$ Routed to **Model B (v2 Hybrid)** for maximum zero-shot transfer (**65.61%** accuracy, **97.06%** LoF recall).

```bash
# Test In-Distribution Conduction (Routes to Model A)
curl -s -X POST http://127.0.0.1:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "material": "Ti-6Al-4V",
    "power": 200.0,
    "velocity": 1000.0,
    "beam_diameter": 100.0,
    "hatch_spacing": 100.0,
    "layer_height": 30.0
  }' | jq .routing_info

# Test Zero-Shot Novel Alloy (Routes to Model B)
curl -s -X POST http://127.0.0.1:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "material": "IN625",
    "power": 200.0,
    "velocity": 1000.0,
    "beam_diameter": 100.0,
    "hatch_spacing": 100.0,
    "layer_height": 30.0
  }' | jq .routing_info
```

### Stop the Service
```bash
bash website/stop.sh
```

---

## 7. Forensic Methodological Audit of the Original Paper

1. **Frozen Weights Discrepancy**:
   - The original paper stated (*Section 2.4*): *"In each case the weights of the LLM model remain unfrozen and were trained along with the classification head..."*
   - Inspection of the authors' published Hugging Face checkpoints revealed only `classification_head.pt` (a 10–34 KB `nn.Linear` tensor). Comparing transformer weights confirmed **100.0% bit-for-bit identity** with stock checkpoints. The base LLMs were strictly frozen feature extractors.
2. **Sibling Row Contamination**:
   - For every physical $(P, v, d)$ simulation, 400 sibling variations were generated across hatch spacing and layer heights ($20 \times 20$).
   - Random row-level splitting permitted **99.8731% (72,383 of 72,475 rows)** in the validation set to share identical physical process parameters with training data.
   - Sibling contamination is eliminated under our group-safe partitioning (**0.00% overlap**).

---

## 8. Final Deliverable Packaging

To build the pristine release archive (excluding virtual environments, pycache, and temporary files):
```bash
bash scripts/package_deliverable.sh
```
This generates `AdditiveLLM_Hybrid_Release.zip` containing the complete codebase, serialized model checkpoints, group-safe datasets, academic reports, publication figures, and web dashboard.

---

## 9. Academic Report & Citations

The full academic report detailing methodology, thermodynamic governing equations, forensic audits, and ablation experiments is available in:
- Markdown: [`report/AdditiveLLM_Final_Report.md`](file:///Users/chandrashanmukhi/Downloads/AdditiveLLM_Reproduction/report/AdditiveLLM_Final_Report.md)
- Word DOCX: [`report/AdditiveLLM_Final_Report.docx`](file:///Users/chandrashanmukhi/Downloads/AdditiveLLM_Reproduction/report/AdditiveLLM_Final_Report.docx)

```bibtex
@article{pak2025additivellm,
  title={AdditiveLLM: Large language models predict defects in metals additive manufacturing},
  author={Pak, Peter and Farimani, Amir Barati},
  journal={Additive Manufacturing Letters},
  volume={14},
  pages={100292},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.addlet.2025.100292}
}
```
