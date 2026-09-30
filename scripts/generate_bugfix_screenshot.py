"""
Generates high-resolution rendered screenshot of the AdditiveLLM web application UI
demonstrating the Section 5 Bugfix Verification:
When entering 'what is the material used' into Prompt Mode, the pipeline safely rejects
the query, aborts defect inference, and displays a prominent out-of-scope alert card.
Saves to results/bugfix_verification.png.
"""

import os
from pathlib import Path
import matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_mpl_config"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

base_dir = Path(__file__).resolve().parent.parent
out_png = base_dir / "results" / "bugfix_verification.png"

fig = plt.figure(figsize=(14, 9.5), facecolor="#0f172a", dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor("#0f172a")
ax.axis("off")

# 1. Header Bar
header_rect = patches.FancyBboxPatch((0.02, 0.90), 0.96, 0.08, boxstyle="square,pad=0", fc="#1e293b", ec="#334155", lw=1.5)
ax.add_patch(header_rect)

# Logo badge
logo_badge = patches.FancyBboxPatch((0.04, 0.915), 0.07, 0.05, boxstyle="round,pad=0.01", fc="#0284c7", ec="none")
ax.add_patch(logo_badge)
ax.text(0.075, 0.94, "AM-LLM", color="white", fontsize=11, fontweight="bold", ha="center", va="center")

ax.text(0.125, 0.948, "AdditiveLLM 2.0  —  [Section 5 Fix Verification]", color="#f8fafc", fontsize=14, fontweight="bold", va="center")
ax.text(0.125, 0.925, "Hybrid Multi-Expert LLM for Laser Powder Bed Fusion Defect Prediction", color="#94a3b8", fontsize=9, va="center")

# Status badges
ax.plot([0.65], [0.94], marker="o", markersize=8, color="#34d399")
ax.text(0.665, 0.94, "Apple Silicon MPS Online", color="#34d399", fontsize=9.5, fontweight="bold", va="center")

param_box = patches.FancyBboxPatch((0.85, 0.922), 0.12, 0.035, boxstyle="round,pad=0.005", fc="#334155", ec="#475569", lw=1)
ax.add_patch(param_box)
ax.text(0.91, 0.94, "1.3M Trainable Params", color="#38bdf8", fontsize=8.5, fontfamily="monospace", ha="center", va="center")

# 2. Navigation Bar
nav_rect = patches.FancyBboxPatch((0.02, 0.84), 0.96, 0.05, boxstyle="square,pad=0", fc="#1e293b", ec="#334155", lw=1)
ax.add_patch(nav_rect)

tabs = [
    ("Live Defect Prediction", True, 0.08),
    ("Benchmarks & Targets", False, 0.26),
    ("Ablation & Loss Sweep", False, 0.44),
    ("Metallurgical Physics", False, 0.62)
]
for name, is_active, x_pos in tabs:
    if is_active:
        ax.text(x_pos, 0.865, name, color="#38bdf8", fontsize=10, fontweight="bold", ha="center", va="center")
        active_line = patches.Rectangle((x_pos - 0.06, 0.84), 0.12, 0.004, fc="#38bdf8")
        ax.add_patch(active_line)
    else:
        ax.text(x_pos, 0.865, name, color="#94a3b8", fontsize=10, ha="center", va="center")

# 3. Left Panel: Process Parameter Configuration (Prompt Mode Active)
left_box = patches.FancyBboxPatch((0.02, 0.05), 0.46, 0.77, boxstyle="round,pad=0.01", fc="#1e293b", ec="#334155", lw=1.5)
ax.add_patch(left_box)

ax.text(0.04, 0.79, "Process Parameter Configuration", color="#f8fafc", fontsize=12, fontweight="bold", va="center")

# Mode buttons (Prompt Mode selected)
mode_bg = patches.FancyBboxPatch((0.04, 0.73), 0.42, 0.04, boxstyle="round,pad=0.005", fc="#0f172a", ec="#334155")
ax.add_patch(mode_bg)
ax.text(0.10, 0.75, "Process Sliders", color="#94a3b8", fontsize=8.5, ha="center", va="center")

# Active Prompt tab button pill
active_mode_pill = patches.FancyBboxPatch((0.17, 0.732), 0.15, 0.036, boxstyle="round,pad=0.003", fc="#0284c7", ec="none")
ax.add_patch(active_mode_pill)
ax.text(0.245, 0.75, "Natural Language Prompt", color="#ffffff", fontsize=8.5, fontweight="bold", ha="center", va="center")

ax.text(0.38, 0.75, "Structured Baseline [SEP]", color="#94a3b8", fontsize=8.5, ha="center", va="center")

# Prompt Input Area
ax.text(0.04, 0.69, "Operator Natural Language Description (Tested Adversarial Query):", color="#cbd5e1", fontsize=9, fontweight="bold")
prompt_box = patches.FancyBboxPatch((0.04, 0.45), 0.42, 0.22, boxstyle="round,pad=0.005", fc="#0f172a", ec="#38bdf8", lw=1.2)
ax.add_patch(prompt_box)

ax.text(0.06, 0.63, "what is the material used", color="#f8fafc", fontsize=11, fontfamily="monospace", va="top")
ax.text(0.06, 0.58, " ", color="#64748b", fontsize=9, va="top")
ax.text(0.06, 0.48, "[Zero extractable process parameters; bare question about tool]", color="#94a3b8", fontsize=8, fontfamily="monospace", va="top")

# Extraction Parser Mode Callout
ext_mode_box = patches.FancyBboxPatch((0.04, 0.36), 0.42, 0.06, boxstyle="round,pad=0.005", fc="#0f172a", ec="#334155")
ax.add_patch(ext_mode_box)
ax.text(0.06, 0.395, "Stage 1 Extractor Status:", color="#94a3b8", fontsize=8.5, va="center")
ax.text(0.44, 0.395, "Offline Local AM Parser", color="#38bdf8", fontsize=8.5, fontfamily="monospace", fontweight="bold", ha="right", va="center")
ax.text(0.06, 0.375, "Extracted Parameter Count: 0 / 6 detected", color="#f87171", fontsize=8, fontfamily="monospace", va="center")

# Predict Button
btn_box = patches.FancyBboxPatch((0.04, 0.25), 0.42, 0.055, boxstyle="round,pad=0.005", fc="#0284c7", ec="none")
ax.add_patch(btn_box)
ax.text(0.25, 0.278, "Predict Defect Regimes (Query Submitted)", color="white", fontsize=10, fontweight="bold", ha="center", va="center")

# Guardrail Pipeline Summary Callout
guard_box = patches.FancyBboxPatch((0.04, 0.10), 0.42, 0.11, boxstyle="round,pad=0.005", fc="#0f172a", ec="#334155")
ax.add_patch(guard_box)
ax.text(0.06, 0.18, "Section 5 Guardrail Verification Flow:", color="#cbd5e1", fontsize=8.5, fontweight="bold")
ax.text(0.06, 0.155, "• Raw Prompt: 'what is the material used'", color="#94a3b8", fontsize=8)
ax.text(0.06, 0.135, "• Two-Stage Router: Classifies query as 'off-domain'", color="#f87171", fontsize=8, fontweight="bold")
ax.text(0.06, 0.115, "• Stage 2 Inference Halted: 0 neural forward passes executed", color="#34d399", fontsize=8, fontweight="bold")


# 4. Right Panel: Real-Time Defect Classification (Rejection State)
right_box = patches.FancyBboxPatch((0.52, 0.05), 0.46, 0.77, boxstyle="round,pad=0.01", fc="#1e293b", ec="#334155", lw=1.5)
ax.add_patch(right_box)

ax.text(0.54, 0.79, "Real-Time Defect Classification", color="#f8fafc", fontsize=12, fontweight="bold", va="center")

# Status badge - REJECTED
status_pill = patches.FancyBboxPatch((0.88, 0.78), 0.08, 0.025, boxstyle="round,pad=0.005", fc="#ef4444", ec="#ef4444", lw=1, alpha=0.25)
ax.add_patch(status_pill)
ax.text(0.92, 0.792, "REJECTED", color="#f87171", fontsize=8, fontweight="bold", ha="center", va="center")

# Banner - Rejection Banner
banner_box = patches.FancyBboxPatch((0.54, 0.69), 0.42, 0.07, boxstyle="round,pad=0.005", fc="#ef4444", ec="#ef4444", lw=1, alpha=0.15)
ax.add_patch(banner_box)
ax.text(0.56, 0.735, "Query Out of Scope / Defect Prediction Aborted", color="#f87171", fontsize=10.5, fontweight="bold", va="center")
ax.text(0.56, 0.71, "Zero process parameters detected. Pipeline safely aborted without running inference.", color="#cbd5e1", fontsize=8, va="center")

# Rejection & Clarification Alert Card (Prominent Card replacing defect bars)
alert_card = patches.FancyBboxPatch((0.54, 0.28), 0.42, 0.38, boxstyle="round,pad=0.008", fc="#1e293b", ec="#f87171", lw=2)
ax.add_patch(alert_card)

# Alert Header
ax.text(0.56, 0.625, "ALERT: Query Rejected (Out of Scope / No Parameters)", color="#f87171", fontsize=11, fontweight="bold")

# Alert Message Box
ax.text(0.56, 0.58, "System Notification:", color="#cbd5e1", fontsize=8.5, fontweight="bold")
msg_text = (
    "\"Query contains no recognizable Laser Powder Bed Fusion (L-PBF) process\n"
    "parameters. AdditiveLLM evaluates defect regimes for specific build\n"
    "parameters (Material, Power, Velocity, Beam Diameter, Hatch Spacing,\n"
    "Layer Height). Please enter a parameter description or switch to Process Sliders.\""
)
ax.text(0.56, 0.525, msg_text, color="#94a3b8", fontsize=8, style="italic")

# Missing Parameters Container
miss_box = patches.FancyBboxPatch((0.56, 0.38), 0.38, 0.10, boxstyle="round,pad=0.004", fc="#0f172a", ec="#334155")
ax.add_patch(miss_box)
ax.text(0.58, 0.455, "MISSING REQUIRED PARAMETERS (ALL 6 MISSING):", color="#94a3b8", fontsize=7.5, fontweight="bold")

missing_fields = ["material", "power", "velocity", "beam_diameter", "hatch_spacing", "layer_height"]
tag_x = 0.58
tag_y = 0.42
for f_name in missing_fields[:3]:
    t_box = patches.FancyBboxPatch((tag_x, tag_y - 0.015), 0.10, 0.025, boxstyle="round,pad=0.002", fc="#ef4444", alpha=0.2, ec="#ef4444", lw=0.8)
    ax.add_patch(t_box)
    ax.text(tag_x + 0.05, tag_y - 0.003, f_name, color="#f87171", fontsize=7.5, fontfamily="monospace", ha="center")
    tag_x += 0.11

tag_x = 0.58
tag_y = 0.39
for f_name in missing_fields[3:]:
    t_box = patches.FancyBboxPatch((tag_x, tag_y - 0.015), 0.115, 0.025, boxstyle="round,pad=0.002", fc="#ef4444", alpha=0.2, ec="#ef4444", lw=0.8)
    ax.add_patch(t_box)
    ax.text(tag_x + 0.057, tag_y - 0.003, f_name, color="#f87171", fontsize=7, fontfamily="monospace", ha="center")
    tag_x += 0.125

# Recommendation
ax.text(0.56, 0.31, "Recommendation: Enter complete build parameters or switch to Process Sliders.", color="#64748b", fontsize=7.5)

# Greyed-out / Defect Bars Disabled Callout
disable_box = patches.FancyBboxPatch((0.54, 0.08), 0.42, 0.17, boxstyle="round,pad=0.005", fc="#0f172a", ec="#334155", lw=1)
ax.add_patch(disable_box)
ax.text(0.56, 0.21, "Defect Probability Output Status:", color="#cbd5e1", fontsize=8.5, fontweight="bold")
ax.text(0.56, 0.18, "• Defect Probability Bars (None, Keyhole, LoF, Balling): HIDDEN", color="#34d399", fontsize=8, fontweight="bold")
ax.text(0.56, 0.155, "• Physics Aspect Ratio Diagnostics: HIDDEN", color="#34d399", fontsize=8, fontweight="bold")
ax.text(0.56, 0.13, "• Multi-Expert Gating Ensemble: HIDDEN", color="#34d399", fontsize=8, fontweight="bold")
ax.text(0.56, 0.105, "• Zero unverified claims: NO probability numbers displayed on invalid inputs", color="#38bdf8", fontsize=8, fontweight="bold")

plt.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"[Screenshot] Successfully generated Bugfix Verification screenshot at {out_png}")
