# AdditiveLLM Dataset Documentation: Origins, Physics, and Splits

## 1. Dataset Origin and Simulation Foundation

The dataset utilized in this project reproduces and extends the additive manufacturing defect benchmark introduced by **Pak & Barati Farimani (2025)** (*Additive Manufacturing Letters*, "AdditiveLLM: A Multimodal Large Language Model for Additive Manufacturing").

The underlying data originates from high-fidelity 3D Computational Fluid Dynamics (CFD) simulations conducted in **FLOW-3D**, capturing thermo-fluid melt pool dynamics in Laser Powder Bed Fusion (L-PBF). The simulation database models:
- Free-surface fluid flow governed by Navier-Stokes equations with surface tension and Marangoni convection ($d\gamma/dT$).
- Laser-matter interaction with Fresnel optical absorption and temperature-dependent thermophysical properties.
- Melt pool geometry outputs: Width ($w$, $\mu$m), Depth ($d$, $\mu$m), and Length ($l$, $\mu$m).

---

## 2. Process Parameters and Defect Physics

Each simulation scenario is parameterized by five machine process variables and the alloy chemistry:
1. **Laser Power ($P$)**: Range $50 - 500\text{ W}$ (Normalizing prior: $\mu = 180.0, \sigma = 100.0$)
2. **Scan Velocity ($v$)**: Range $200 - 2500\text{ mm/s}$ (Normalizing prior: $\mu = 1000.0, \sigma = 600.0$)
3. **Beam Spot Diameter ($d_b$)**: Range $40 - 150\ \mu\text{m}$ (Normalizing prior: $\mu = 90.0, \sigma = 30.0$)
4. **Hatch Spacing ($h$)**: Range $40 - 250\ \mu\text{m}$ (Normalizing prior: $\mu = 120.0, \sigma = 60.0$)
5. **Layer Height ($t$)**: Range $20 - 80\ \mu\text{m}$ (Normalizing prior: $\mu = 40.0, \sigma = 20.0$)

### Hydrodynamic Defect Criteria (Eqs. 1–3)
The multi-label defect targets ($y \in \{0, 1\}^4$) are governed by first-principles physical thresholds:
- **Keyhole Porosity ($y_1$)**:
  $$\frac{w}{d} \le 1.5$$
  Caused by excessive linear energy density ($E_L = P/v$) driving recoil pressure vaporization and deep vapor depression collapse.
- **Lack of Fusion Porosity ($y_2$)**:
  $$\left(\frac{h}{w}\right)^2 + \left(\frac{t}{d}\right)^2 > 1.0$$
  Caused by insufficient melt pool cross-sectional overlap, leaving un-melted powder particles between adjacent scan tracks or layers.
- **Balling Porosity ($y_3$)**:
  $$\frac{l}{w} \ge \pi \approx 3.14159$$
  Triggered by Plateau-Rayleigh hydrodynamic capillary instability when the liquid melt pool cylinder elongates beyond its circumferential wavelength.
- **Defect-Free / Conduction Mode ($y_0$)**:
  Satisfies $w/d > 1.5$, $(h/w)^2 + (t/d)^2 \le 1.0$, and $l/w < \pi$.

---

## 3. Group-Safe Split Methodology & Anti-Leakage Protocol

In the original benchmark, random row-level splitting permitted "sibling leakage"—instances where identical physical process parameter tuples $(P, v, d_b, h, t)$ appeared in both training and test sets under superficial text variations.

To enforce strict zero-leakage academic standards, this repository uses **Group-Safe Split Partitioning** (`data/splits_group_safe/`):
- Splitting key: A cryptographic hash of the continuous physical parameter tuple `(Material, Power, Velocity, Beam Diameter, Hatch Spacing, Layer Height)`.
- Sibling overlap between train and test: **Strictly 0.00%** (verified by `src/run_leakage_audit.py`).

### Split Manifest
| Split File | Number of Samples | Scope & Physical Description | Material Distribution | Leakage Status |
| :--- | :--- | :--- | :--- | :--- |
| `training.parquet` | **1,489,836** | Group-partitioned training set | 12 Base Alloys (Ti-6Al-4V, SS316L, IN718, AlSi10Mg, etc.) | 0.00% Sibling Overlap |
| `tier_a.parquet` | **83,074** | Tier A: In-Distribution test set (seen physical parameter ranges & seen alloys) | 12 Base Alloys | 0.00% Sibling Overlap |
| `tier_b.parquet` | **130,135** | Tier B: Held-Out Physical Groups (unseen $(P, v, d_b)$ combinations) | 12 Base Alloys | 0.00% Sibling Overlap |
| `tier_c.parquet` | **38,897** | Tier C: Held-Out Novel Material (strict zero-shot transfer) | **Inconel 625 (IN625)** (Excluded entirely from train/A/B) | 0.00% Sibling Overlap |
| **Primary Deployment** | **213,209** | **Tier A + Tier B Pooled** (Industry Deployment Envelope) | 12 Base Alloys | 0.00% Sibling Overlap |
| **Total Database** | **1,741,942** | Complete curated group-safe dataset | All 13 Alloys | Certified Clean |

---

## 4. Verification and File Integrity

All split files are stored in Apache Parquet format with snappy compression. Feature verification can be executed via:
```bash
python src/run_leakage_audit.py
```
This confirms that none of the 20 input features in Model A or 12 input features in Model B leak ground-truth melt pool dimensions $(w, d, l)$ or labels.
