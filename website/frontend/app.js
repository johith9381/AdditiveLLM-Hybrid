/**
 * AdditiveLLM Frontend Interactive Controller.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const modeBtns = document.querySelectorAll(".mode-btn");

  const manualGroup = document.getElementById("manualInputsGroup");
  const promptGroup = document.getElementById("promptInputGroup");
  const baselineGroup = document.getElementById("baselineInputGroup");

  const presetSelect = document.getElementById("presetSelect");
  const materialSelect = document.getElementById("materialSelect");
  const powerInput = document.getElementById("powerInput");
  const velocityInput = document.getElementById("velocityInput");
  const beamInput = document.getElementById("beamInput");
  const hatchInput = document.getElementById("hatchInput");
  const layerInput = document.getElementById("layerInput");

  const powerVal = document.getElementById("powerVal");
  const velocityVal = document.getElementById("velocityVal");
  const beamVal = document.getElementById("beamVal");
  const hatchVal = document.getElementById("hatchVal");
  const layerVal = document.getElementById("layerVal");
  const ledDisplay = document.getElementById("ledDisplay");

  const promptTextArea = document.getElementById("promptTextArea");
  const baselineTextInput = document.getElementById("baselineTextInput");
  const predictForm = document.getElementById("predictForm");
  const btnSubmit = document.getElementById("btnSubmit");
  const btnText = document.getElementById("btnText");
  const btnSpinner = document.getElementById("btnSpinner");

  let activeMode = "manual";

  // Tab Switching
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const tabId = `tab-${btn.dataset.tab}`;
      const targetTab = document.getElementById(tabId);
      if (targetTab) targetTab.classList.add("active");

      if (btn.dataset.tab === "benchmarks" || btn.dataset.tab === "ablation") {
        fetchDashboardData();
      }
    });
  });

  // Mode Switching
  modeBtns.forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      modeBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeMode = btn.dataset.mode;

      manualGroup.classList.add("hidden");
      promptGroup.classList.add("hidden");
      baselineGroup.classList.add("hidden");

      if (activeMode === "manual") {
        manualGroup.classList.remove("hidden");
      } else if (activeMode === "prompt") {
        promptGroup.classList.remove("hidden");
        syncPromptText();
      } else if (activeMode === "baseline") {
        baselineGroup.classList.remove("hidden");
        syncBaselineText();
      }
    });
  });

  // Update Slider Labels & Linear Energy Density
  function updateSliderDisplays() {
    const p = parseFloat(powerInput.value);
    const v = parseFloat(velocityInput.value);
    const b = parseFloat(beamInput.value);
    const h = parseFloat(hatchInput.value);
    const l = parseFloat(layerInput.value);

    powerVal.textContent = `${p} W`;
    velocityVal.textContent = `${v} mm/s`;
    beamVal.textContent = `${b} µm`;
    hatchVal.textContent = `${h} µm`;
    layerVal.textContent = `${l} µm`;

    const led = p / (v + 1e-3);
    ledDisplay.textContent = `${led.toFixed(3)} J/mm`;

    if (activeMode === "prompt") syncPromptText();
    if (activeMode === "baseline") syncBaselineText();
  }

  function syncPromptText() {
    const mat = materialSelect.value;
    const p = powerInput.value;
    const v = velocityInput.value;
    const b = beamInput.value;
    const h = hatchInput.value;
    const l = layerInput.value;
    promptTextArea.value = `In laser powder bed fusion additive manufacturing of ${mat}, the process operates at a laser power of ${p} W and scan speed of ${v} mm/s with a beam diameter of ${b} microns, hatch spacing of ${h} microns, and layer thickness of ${l} microns.`;
  }

  function syncBaselineText() {
    const mat = materialSelect.value;
    const p = powerInput.value;
    const v = velocityInput.value;
    const b = beamInput.value;
    const h = hatchInput.value;
    const l = layerInput.value;
    baselineTextInput.value = `${mat}[SEP]${parseFloat(p).toFixed(1)} W[SEP]${parseFloat(v).toFixed(1)} mm/s[SEP]${parseFloat(b).toFixed(1)} microns[SEP]${parseFloat(h).toFixed(1)} microns[SEP]${parseFloat(l).toFixed(1)} microns`;
  }

  [powerInput, velocityInput, beamInput, hatchInput, layerInput].forEach(slider => {
    slider.addEventListener("input", updateSliderDisplays);
  });
  materialSelect.addEventListener("change", updateSliderDisplays);

  // Preset Configurations
  const presets = {
    conduction: { material: "Ti-6Al-4V", power: 200, velocity: 1000, beam: 100, hatch: 100, layer: 30 },
    keyhole: { material: "Ti-6Al-4V", power: 380, velocity: 450, beam: 80, hatch: 100, layer: 30 },
    balling: { material: "SS316L", power: 250, velocity: 2200, beam: 90, hatch: 90, layer: 30 },
    lof: { material: "AlSi10Mg", power: 90, velocity: 1600, beam: 120, hatch: 150, layer: 50 },
  };

  presetSelect.addEventListener("change", () => {
    const key = presetSelect.value;
    if (presets[key]) {
      const cfg = presets[key];
      materialSelect.value = cfg.material;
      powerInput.value = cfg.power;
      velocityInput.value = cfg.velocity;
      beamInput.value = cfg.beam;
      hatchInput.value = cfg.hatch;
      layerInput.value = cfg.layer;
      updateSliderDisplays();
      submitInference();
    }
  });

  // Example Prompt Chips (Click to load & automatically run inference)
  const exampleChips = document.querySelectorAll(".example-chip");
  exampleChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const promptText = chip.getAttribute("data-prompt");
      if (promptText) {
        promptTextArea.value = promptText;
        // Switch to prompt mode
        modeBtns.forEach(b => b.classList.remove("active"));
        const promptBtn = document.querySelector('.mode-btn[data-mode="prompt"]');
        if (promptBtn) promptBtn.classList.add("active");
        activeMode = "prompt";

        manualGroup.classList.add("hidden");
        baselineGroup.classList.add("hidden");
        promptGroup.classList.remove("hidden");

        submitInference();
      }
    });
  });

  // Handle Form Submission
  predictForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitInference();
  });

  async function submitInference() {
    btnSubmit.disabled = true;
    btnSpinner.classList.remove("hidden");
    btnText.textContent = "Computing Hybrid Inference...";

    let payload = {};

    if (activeMode === "manual") {
      payload = {
        mode: "manual",
        material: materialSelect.value,
        power: parseFloat(powerInput.value),
        velocity: parseFloat(velocityInput.value),
        beam_diameter: parseFloat(beamInput.value),
        hatch_spacing: parseFloat(hatchInput.value),
        layer_height: parseFloat(layerInput.value),
      };
    } else if (activeMode === "prompt") {
      payload = {
        mode: "prompt",
        text: promptTextArea.value.trim(),
      };
    } else if (activeMode === "baseline") {
      payload = {
        mode: "baseline",
        text: baselineTextInput.value.trim(),
      };
    }

  const API_BASE = window.location.protocol === "file:" ? "http://localhost:8080" : "";

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Server error ${res.status}: ${errText}`);
      }
      const data = await res.json();
      renderInferenceResults(data);
    } catch (err) {
      console.error("Inference request failed:", err);
      alert(`Inference failed: ${err.message || err}`);
    } finally {
      btnSubmit.disabled = false;
      btnSpinner.classList.add("hidden");
      btnText.textContent = "Predict Defect Regimes";
    }
  }

  function renderInferenceResults(data) {
    const clarificationAlert = document.getElementById("clarificationAlert");
    const alertHeading = document.getElementById("alertHeading");
    const alertMessage = document.getElementById("alertMessage");
    const alertIcon = document.getElementById("alertIcon");
    const alertMissingTags = document.getElementById("alertMissingTags");
    const alertMissingContainer = document.getElementById("alertMissingFieldsContainer");
    const predictionGroup = document.getElementById("predictionResultsGroup");

    const banner = document.getElementById("regimeBanner");
    const badge = document.getElementById("regimeBadge");
    const title = document.getElementById("regimeTitle");
    const desc = document.getElementById("regimeDesc");

    // Case 1: Rejected / Off-Domain Query
    if (data.status === "rejected" || data.category === "off-domain") {
      predictionGroup.classList.add("hidden");
      clarificationAlert.classList.remove("hidden");
      clarificationAlert.className = "alert-card rejected";

      alertIcon.textContent = "🛑";
      alertHeading.textContent = "Query Rejected (Out of Scope / No Parameters)";
      alertMessage.textContent = data.message;

      if (data.missing_fields && data.missing_fields.length > 0) {
        alertMissingContainer.classList.remove("hidden");
        alertMissingTags.innerHTML = data.missing_fields.map(f => `<span class="missing-tag">${f}</span>`).join("");
      } else {
        alertMissingContainer.classList.add("hidden");
      }

      badge.textContent = "REJECTED";
      badge.className = "regime-badge rejected";
      banner.className = "regime-banner banner-rejected";
      title.textContent = "Query Out of Scope / Defect Prediction Aborted";
      desc.textContent = "Zero process parameters detected. Defect prediction pipeline safely aborted without running inference.";
      return;
    }

    // Case 2: Incomplete / Partial Query
    if (data.status === "clarification_needed" || data.category === "in-domain-partial") {
      predictionGroup.classList.add("hidden");
      clarificationAlert.classList.remove("hidden");
      clarificationAlert.className = "alert-card partial";

      alertIcon.textContent = "⚠️";
      alertHeading.textContent = "Clarification Required (Incomplete Build Parameters)";
      alertMessage.textContent = data.message;

      alertMissingContainer.classList.remove("hidden");
      alertMissingTags.innerHTML = (data.missing_fields || []).map(f => `<span class="missing-tag">${f}</span>`).join("");

      badge.textContent = "PARTIAL";
      badge.className = "regime-badge partial";
      banner.className = "regime-banner banner-partial";
      title.textContent = "Incomplete L-PBF Process Parameters";
      desc.textContent = "The system requires all 6 process parameters to evaluate defect physics. Stage 2 inference halted.";
      return;
    }

    // Case 3: In-domain complete -> Render predictions
    clarificationAlert.classList.add("hidden");
    predictionGroup.classList.remove("hidden");

    const defects = data.defects;
    const regime = data.process_regime;
    const diag = data.physics_meltpool_diagnostics;
    const uncertainty = data.ensemble_uncertainty;

    // 1. Prominent Status Banner
    if (data.banner) {
      title.textContent = data.banner.title;
      badge.textContent = regime.risk_level;
      badge.className = `regime-badge ${data.banner.type === 'defect_free' ? 'low' : 'high'}`;
      banner.className = `regime-banner banner-${data.banner.type === 'defect_free' ? 'low' : 'high'}`;
      desc.textContent = data.banner.description + (data.banner.suggested_adjustment ? " " + data.banner.suggested_adjustment : "");
    } else {
      title.textContent = regime.primary_regime;
      badge.textContent = regime.risk_level;
      badge.className = `regime-badge ${regime.risk_level.toLowerCase()}`;
      banner.className = `regime-banner banner-${regime.risk_level === 'LOW' ? 'low' : 'high'}`;
      desc.textContent = "Process operating parameters evaluated.";
    }

    // 2. Defect Probabilities
    updateDefectBar("None", defects.none.probability);
    updateDefectBar("Keyhole", defects.keyhole.probability, defects.keyhole.physics_indicator);
    updateDefectBar("Lof", defects.lack_of_fusion.probability, defects.lack_of_fusion.physics_indicator);
    updateDefectBar("Balling", defects.balling.probability, defects.balling.physics_indicator);

    // 3. Physics Aspect Ratios & Energy Densities
    const wdDisplay = document.getElementById("wdDisplay");
    const wdStatus = document.getElementById("wdStatus");
    if (wdDisplay) wdDisplay.textContent = diag.predicted_width_to_depth_ratio.toFixed(2);
    if (wdStatus) {
      if (diag.keyhole_geometry_flag) {
        wdStatus.textContent = "Keyhole Vaporization (<= 1.5)";
        wdStatus.className = "diag-status danger";
      } else {
        wdStatus.textContent = "Conduction (> 1.5)";
        wdStatus.className = "diag-status safe";
      }
    }

    const lwDisplay = document.getElementById("lwDisplay");
    const lwStatus = document.getElementById("lwStatus");
    if (lwDisplay) lwDisplay.textContent = diag.predicted_length_to_width_ratio.toFixed(2);
    if (lwStatus) {
      if (diag.balling_geometry_flag) {
        lwStatus.textContent = "Rayleigh Instability (>= 3.14)";
        lwStatus.className = "diag-status danger";
      } else {
        lwStatus.textContent = "Stable (< 3.14)";
        lwStatus.className = "diag-status safe";
      }
    }

    const hbDisplay = document.getElementById("hbDisplay");
    if (hbDisplay && data.hydrodynamic_aspect_ratios) {
      hbDisplay.textContent = data.hydrodynamic_aspect_ratios.hatch_to_beam_aspect_ratio.toFixed(2);
    }

    const vedDisplay = document.getElementById("vedDisplay");
    if (vedDisplay && data.hydrodynamic_aspect_ratios) {
      vedDisplay.textContent = `${data.hydrodynamic_aspect_ratios.volumetric_energy_density_J_per_mm3} J/mm³`;
    }

    const ledDisplay2 = document.getElementById("ledDisplay2");
    if (ledDisplay2 && data.hydrodynamic_aspect_ratios) {
      ledDisplay2.textContent = `${data.hydrodynamic_aspect_ratios.linear_energy_density_J_per_mm.toFixed(3)} J/mm`;
    }

    // 4. Update 5-Branch Prediction Breakdown
    if (data.branch_predictions) {
      const b1 = data.branch_predictions.branch1_distilbert_lora;
      const b2 = data.branch_predictions.branch2_scibert_lora;
      const b3 = data.branch_predictions.branch3_materials_lora;
      const b4 = data.branch_predictions.branch4_ft_transformer;
      const b5 = data.branch_predictions.branch5_gbdt_expert;

      const formatTop = (b) => {
        if (!b) return "N/A";
        const maxP = Math.max(b.predicted_none, b.predicted_keyhole, b.predicted_lof, b.predicted_balling);
        let name = "None";
        if (maxP === b.predicted_keyhole) name = "Keyhole";
        else if (maxP === b.predicted_lof) name = "Lack of Fusion";
        else if (maxP === b.predicted_balling) name = "Balling";
        return `${name} (${(maxP * 100).toFixed(1)}%)`;
      };

      const b1El = document.getElementById("b1Prediction");
      const b2El = document.getElementById("b2Prediction");
      const b3El = document.getElementById("b3Prediction");
      const b4El = document.getElementById("b4Prediction");
      const b5El = document.getElementById("b5Prediction");

      if (b1El) b1El.textContent = formatTop(b1);
      if (b2El) b2El.textContent = formatTop(b2);
      if (b3El) b3El.textContent = formatTop(b3);
      if (b4El) b4El.textContent = formatTop(b4);
      if (b5El) b5El.textContent = formatTop(b5);
    }

    // 4b. Update Two-Stage Physics Residual Decomposition (AdditiveLLM v3)
    if (data.residual_decomposition) {
      const s1 = data.residual_decomposition.stage1_analytical_physics;
      const s2 = data.residual_decomposition.stage2_residual_correction;

      if (s1 && s1.probabilities) {
        const s1None = document.getElementById("s1None");
        const s1Keyhole = document.getElementById("s1Keyhole");
        const s1Lof = document.getElementById("s1Lof");
        const s1Balling = document.getElementById("s1Balling");
        if (s1None && s1.probabilities.None !== undefined) s1None.textContent = `${(s1.probabilities.None * 100).toFixed(1)}%`;
        if (s1Keyhole && s1.probabilities.Keyhole !== undefined) s1Keyhole.textContent = `${(s1.probabilities.Keyhole * 100).toFixed(1)}%`;
        if (s1Lof && s1.probabilities["Lack of Fusion"] !== undefined) s1Lof.textContent = `${(s1.probabilities["Lack of Fusion"] * 100).toFixed(1)}%`;
        if (s1Balling && s1.probabilities.Balling !== undefined) s1Balling.textContent = `${(s1.probabilities.Balling * 100).toFixed(1)}%`;
      }

      if (s1 && s1.dimensionless_features) {
        const s1Beta = document.getElementById("s1Beta");
        const s1Pe = document.getElementById("s1Pe");
        if (s1Beta && s1.dimensionless_features.normalized_enthalpy_beta !== undefined) {
          s1Beta.textContent = s1.dimensionless_features.normalized_enthalpy_beta.toFixed(2);
        }
        if (s1Pe && s1.dimensionless_features.peclet_number_Pe !== undefined) {
          s1Pe.textContent = s1.dimensionless_features.peclet_number_Pe.toFixed(2);
        }
      }

      if (s2 && s2.residual_deltas) {
        const s2None = document.getElementById("s2None");
        const s2Keyhole = document.getElementById("s2Keyhole");
        const s2Lof = document.getElementById("s2Lof");
        const s2Balling = document.getElementById("s2Balling");
        const fmtDelta = (val) => (val !== undefined ? (val >= 0 ? `+${val.toFixed(3)}` : `${val.toFixed(3)}`) : "-");
        if (s2None) s2None.textContent = fmtDelta(s2.residual_deltas.None);
        if (s2Keyhole) s2Keyhole.textContent = fmtDelta(s2.residual_deltas.Keyhole);
        if (s2Lof) s2Lof.textContent = fmtDelta(s2.residual_deltas["Lack of Fusion"]);
        if (s2Balling) s2Balling.textContent = fmtDelta(s2.residual_deltas.Balling);
      }
    }

    // 5. Update Empirical Confidence Callout
    if (data.confidence) {
      const confDisp = document.getElementById("confidenceDisplay");
      const confTier = document.getElementById("confidenceTier");
      const confDesc = document.getElementById("confidenceDesc");
      if (confDisp) confDisp.textContent = `${data.confidence.percentage}%`;
      if (confTier) confTier.textContent = data.confidence.tier;
      if (confDesc) confDesc.textContent = data.confidence.explanation;
    }
  }

  function updateDefectBar(name, prob, indicatorText) {
    const pct = (prob * 100).toFixed(1);
    const probEl = document.getElementById(`prob${name}`);
    const barEl = document.getElementById(`bar${name}`);
    const indEl = document.getElementById(`indicator${name}`);

    if (probEl) probEl.textContent = `${pct}%`;
    if (barEl) barEl.style.width = `${pct}%`;
    if (indEl && indicatorText) indEl.textContent = `Indicator: ${indicatorText}`;
  }

  // Figure toggle handler
  const figBtns = document.querySelectorAll("#figToggleGroup .mode-btn");
  const figContainer = document.getElementById("figDisplayContainer");
  if (figBtns && figContainer) {
    figBtns.forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        figBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const figType = btn.dataset.fig;

        if (figType === "overall") {
          figContainer.innerHTML = `
            <div class="fig-box" style="grid-column: span 2;">
              <h4>Format A: Multi-Tier Accuracy & F1 Comparison (Tiers A, B, C)</h4>
              <img src="/results/all_tiers_comparison.png" alt="Multi-Tier Comparison" class="fig-img" onerror="this.src='/static/placeholder.png';">
            </div>
          `;
        } else if (figType === "pooled") {
          figContainer.innerHTML = `
            <div class="fig-box">
              <h4>Format B: Overall Pooled Confusion Matrix (Baseline)</h4>
              <img src="/results/confusion_matrices/confusion_matrix_overall_pooled_baseline.png" alt="Pooled CM" class="fig-img" onerror="this.src='/static/placeholder.png';">
            </div>
            <div class="fig-box">
              <h4>Format B: Overall Pooled Calibration Reliability</h4>
              <img src="/results/calibration_overall_pooled_baseline.png" alt="Pooled Calibration" class="fig-img" onerror="this.src='/static/placeholder.png';">
            </div>
          `;
        } else if (figType === "tiera") {
          figContainer.innerHTML = `
            <div class="fig-box">
              <h4>Tier A (In-Distribution) Confusion Matrix</h4>
              <img src="/results/confusion_matrices/confusion_matrix_tier_a_baseline.png" alt="Tier A CM" class="fig-img" onerror="this.src='/static/placeholder.png';">
            </div>
            <div class="fig-box">
              <h4>Tier A Calibration Reliability Diagram</h4>
              <img src="/results/calibration_tier_a_baseline.png" alt="Tier A Calibration" class="fig-img" onerror="this.src='/static/placeholder.png';">
            </div>
          `;
        } else if (figType === "tierb") {
          figContainer.innerHTML = `
            <div class="fig-box">
              <h4>Tier B (Held-Out Groups) Confusion Matrix</h4>
              <img src="/results/confusion_matrices/confusion_matrix_tier_b_baseline.png" alt="Tier B CM" class="fig-img" onerror="this.src='/static/placeholder.png';">
            </div>
            <div class="fig-box">
              <h4>Tier B Calibration Reliability Diagram</h4>
              <img src="/results/calibration_tier_b_baseline.png" alt="Tier B Calibration" class="fig-img" onerror="this.src='/static/placeholder.png';">
            </div>
          `;
        } else if (figType === "tierc") {
          figContainer.innerHTML = `
            <div class="fig-box">
              <h4>Tier C (Held-Out Alloy IN625 OOD) Confusion Matrix</h4>
              <img src="/results/confusion_matrices/confusion_matrix_tier_c_baseline.png" alt="Tier C CM" class="fig-img" onerror="this.src='/static/placeholder.png';">
            </div>
            <div class="fig-box">
              <h4>Tier C Calibration Reliability Diagram</h4>
              <img src="/results/calibration_tier_c_baseline.png" alt="Tier C Calibration" class="fig-img" onerror="this.src='/static/placeholder.png';">
            </div>
          `;
        }
      });
    });
  }

  // Fetch Dashboard Benchmarks and Ablation Data
  async function fetchDashboardData() {
    const API_BASE = window.location.protocol === "file:" ? "http://localhost:8080" : "";
    try {
      const res = await fetch(`${API_BASE}/api/dashboard_data`);
      if (!res.ok) return;
      const data = await res.json();

      // 1. Overall Pooled Headline Cards (Upgraded Stack v3 / v2)
      const pooledSource = (data.overall_pooled_v3 && data.overall_pooled_v3.length > 0) 
        ? data.overall_pooled_v3 
        : ((data.overall_pooled_v2 && data.overall_pooled_v2.length > 0) ? data.overall_pooled_v2 : data.overall_pooled);
      if (pooledSource && pooledSource.length > 0) {
        const basePooled = pooledSource.find(r => r.Format === "Baseline") || pooledSource[0];
        const promPooled = pooledSource.find(r => r.Format === "Prompt");
        const elAcc = document.getElementById("metricPooledAcc");
        const elF1 = document.getElementById("metricPooledF1");
        if (elAcc && basePooled) {
          const bAcc = basePooled.Pooled_Accuracy_Pct !== undefined ? basePooled.Pooled_Accuracy_Pct.toFixed(1) : "";
          const pAcc = (promPooled && promPooled.Pooled_Accuracy_Pct !== undefined) ? promPooled.Pooled_Accuracy_Pct.toFixed(1) : "";
          elAcc.textContent = `${bAcc}% (Base) / ${pAcc}% (Prom)`;
        }
        if (elF1 && basePooled) {
          const bF1 = basePooled.Pooled_Macro_F1 !== undefined ? basePooled.Pooled_Macro_F1.toFixed(3) : "";
          const pF1 = (promPooled && promPooled.Pooled_Macro_F1 !== undefined) ? promPooled.Pooled_Macro_F1.toFixed(3) : "";
          elF1.textContent = `${bF1} (Base) / ${pF1} (Prom)`;
        }
      }

      // 2. Format A 3-Tier Mini Summaries
      const multiSource = (data.multiseed_summary_v3 && data.multiseed_summary_v3.length > 0)
        ? data.multiseed_summary_v3
        : ((data.multiseed_summary_v2 && data.multiseed_summary_v2.length > 0) ? data.multiseed_summary_v2 : data.multiseed_summary);
      if (multiSource && multiSource.length > 0) {
        const tA = multiSource.find(r => r.Tier && r.Tier.includes("Tier A") && r.Format === "Baseline");
        const tB = multiSource.find(r => r.Tier && r.Tier.includes("Tier B") && r.Format === "Baseline");
        const tC = multiSource.find(r => r.Tier && r.Tier.includes("Tier C") && r.Format === "Baseline");
        const elA = document.getElementById("tierAMetric");
        const elB = document.getElementById("tierBMetric");
        const elC = document.getElementById("tierCMetric");
        if (elA && tA) elA.textContent = `${tA.Mean_Accuracy_Pct || tA.Accuracy_Pct}% / ${(tA.Mean_Macro_F1 || tA.Macro_F1).toFixed(3)} F1`;
        if (elB && tB) elB.textContent = `${tB.Mean_Accuracy_Pct || tB.Accuracy_Pct}% / ${(tB.Mean_Macro_F1 || tB.Macro_F1).toFixed(3)} F1`;
        if (elC && tC) elC.textContent = `${tC.Mean_Accuracy_Pct || tC.Accuracy_Pct}% / ${(tC.Mean_Macro_F1 || tC.Macro_F1).toFixed(3)} F1`;
      }

      // 3. Render All-Tier Benchmarks Table (Format A & B)
      const tbodyBench = document.getElementById("benchmarkTableBody");
      if (tbodyBench && multiSource && multiSource.length > 0) {
        tbodyBench.innerHTML = "";
        multiSource.forEach(row => {
          const tr = document.createElement("tr");
          const acc = row.Mean_Accuracy_Pct !== undefined ? row.Mean_Accuracy_Pct : row.Accuracy_Pct;
          const f1 = row.Mean_Macro_F1 !== undefined ? row.Mean_Macro_F1 : row.Macro_F1;
          const prec = row.Mean_Precision !== undefined ? row.Mean_Precision : row.Macro_Precision;
          const rec = row.Mean_Recall !== undefined ? row.Mean_Recall : row.Macro_Recall;
          const ece = row.Mean_ECE !== undefined ? row.Mean_ECE : row.ECE;
          const n_samp = row.Sample_Count || "5 Seeds";

          let classification = "In-Distribution Benchmark";
          if (row.Tier && row.Tier.includes("Tier B")) classification = "Held-Out Process Groups";
          if (row.Tier && row.Tier.includes("Tier C")) classification = "Held-Out Material OOD (Case B Ceiling)";

          tr.innerHTML = `
            <td><strong>${row.Tier}</strong></td>
            <td>${row.Format}</td>
            <td><strong style="color:#38bdf8;">${acc.toFixed(2)}%</strong></td>
            <td>${prec ? prec.toFixed(4) : '-'}</td>
            <td>${rec ? rec.toFixed(4) : '-'}</td>
            <td><strong style="color:#34d399;">${f1.toFixed(4)}</strong></td>
            <td>${ece ? ece.toFixed(4) : '-'}</td>
            <td>${n_samp}</td>
            <td><span style="color:${row.Tier && row.Tier.includes('Tier C') ? '#fbbf24' : '#34d399'}; font-weight:600;">${classification}</span></td>
          `;
          tbodyBench.appendChild(tr);
        });

        // Append Format B Overall Pooled Rows if present
        if (pooledSource && pooledSource.length > 0) {
          pooledSource.forEach(pRow => {
            const tr = document.createElement("tr");
            tr.style.background = "rgba(56, 189, 248, 0.08)";
            tr.innerHTML = `
              <td><strong style="color:#38bdf8;">Overall Pooled (Format B)</strong></td>
              <td>${pRow.Format}</td>
              <td><strong style="color:#38bdf8;">${pRow.Pooled_Accuracy_Pct.toFixed(2)}%</strong></td>
              <td>${pRow.Pooled_Macro_Precision.toFixed(4)}</td>
              <td>${pRow.Pooled_Macro_Recall.toFixed(4)}</td>
              <td><strong style="color:#34d399;">${pRow.Pooled_Macro_F1.toFixed(4)}</strong></td>
              <td>${pRow.Pooled_ECE.toFixed(4)}</td>
              <td>${pRow.Total_Test_Samples_Pooled}</td>
              <td><strong style="color:#38bdf8;">FORMAT B POOLED</strong></td>
            `;
            tbodyBench.appendChild(tr);
          });
        }
      }

      // 4. Render Multi-Seed Robustness Table
      const tbodyMulti = document.getElementById("multiseedTableBody");
      if (tbodyMulti && multiSource && multiSource.length > 0) {
        tbodyMulti.innerHTML = "";
        multiSource.forEach(row => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><strong>${row.Tier}</strong></td>
            <td>${row.Format}</td>
            <td><strong style="color:#38bdf8;">${row.Accuracy_Pct_Mean_Std}</strong></td>
            <td><strong style="color:#34d399;">${row.Macro_F1_Mean_Std}</strong></td>
            <td>${row.Macro_Precision_Mean_Std}</td>
            <td>${row.Macro_Recall_Mean_Std}</td>
            <td>${row.ECE_Mean_Std}</td>
          `;
          tbodyMulti.appendChild(tr);
        });
      }

      // 5. Render Controlled Baselines Table
      const tbodyCtrl = document.getElementById("controlledBaselinesTableBody");
      if (tbodyCtrl && data.controlled_comparison && data.controlled_comparison.length > 0) {
        tbodyCtrl.innerHTML = "";
        data.controlled_comparison.forEach(row => {
          const tr = document.createElement("tr");
          const isHybrid = row.Model.includes("Hybrid");
          if (isHybrid) tr.style.background = "rgba(52, 211, 153, 0.08)";
          tr.innerHTML = `
            <td><strong>${row.Tier}</strong></td>
            <td>${row.Format}</td>
            <td><strong>${row.Model}</strong></td>
            <td><strong style="color:${isHybrid ? '#34d399' : '#94a3b8'};">${row.Accuracy_Pct_Mean_Std}</strong></td>
            <td><strong style="color:${isHybrid ? '#34d399' : '#94a3b8'};">${row.Macro_F1_Mean_Std}</strong></td>
            <td>${row.ECE_Mean_Std}</td>
            <td>${row.Hybrid_Advantage_Over_Paper || "-"}</td>
          `;
          tbodyCtrl.appendChild(tr);
        });
      }

      // 6. Render Tier C Interventions Table
      const tbodyTierc = document.getElementById("tiercInterventionsTableBody");
      if (tbodyTierc && data.tierc_interventions && data.tierc_interventions.length > 0) {
        tbodyTierc.innerHTML = "";
        data.tierc_interventions.forEach(row => {
          const tr = document.createElement("tr");
          const isSig = row.Statistically_Significant_p05;
          const strat = row.Intervention_Strategy || row.Intervention;
          const mech = row.Mechanism_Hypothesis || row.Format;
          const accStr = row.Accuracy_Pct_Mean_Std || (row.Mean_Accuracy_Pct !== undefined ? `${row.Mean_Accuracy_Pct}% ± ${row.Std_Accuracy_Pct}%` : "-");
          const f1Str = row.Macro_F1_Mean_Std || (row.Mean_Macro_F1 !== undefined ? `${row.Mean_Macro_F1} ± ${row.Std_Macro_F1}` : "-");
          const concl = row.Physical_Conclusion || row.Verdict || "-";
          tr.innerHTML = `
            <td><strong>${strat}</strong></td>
            <td>${mech}</td>
            <td><strong>${accStr}</strong></td>
            <td><strong>${f1Str}</strong></td>
            <td>${row.Paired_t_PValue || "-"}</td>
            <td>${row.Wilcoxon_PValue || "-"}</td>
            <td><span style="color:${isSig ? '#34d399' : '#fbbf24'}; font-weight:600;">${concl}</span></td>
          `;
          tbodyTierc.appendChild(tr);
        });
      }

      // 7. Render Tier C Ceiling Diagnostic Table
      const tbodyDiag = document.getElementById("tiercDiagTableBody");
      if (tbodyDiag && data.tierc_diagnostic && data.tierc_diagnostic.length > 0) {
        tbodyDiag.innerHTML = "";
        data.tierc_diagnostic.forEach(row => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><strong>${row.Alloy}</strong></td>
            <td><strong>${row.Zero_Shot_Accuracy_Pct}%</strong></td>
            <td>${row.Zero_Shot_Macro_F1.toFixed(4)}</td>
            <td><strong>${row.Included_Accuracy_Pct}%</strong></td>
            <td>${row.Included_Macro_F1.toFixed(4)}</td>
            <td style="color:${row.Delta_Accuracy_Pct >= 0 ? '#34d399' : '#f87171'}; font-weight:600;">${row.Delta_Accuracy_Pct >= 0 ? '+' : ''}${row.Delta_Accuracy_Pct}%</td>
            <td style="color:${row.Delta_Macro_F1 >= 0 ? '#34d399' : '#f87171'}; font-weight:600;">${row.Delta_Macro_F1 >= 0 ? '+' : ''}${row.Delta_Macro_F1.toFixed(4)}</td>
            <td><span class="regime-badge high" style="padding:2px 8px; font-size:0.75rem;">${row.Diagnosis}</span></td>
          `;
          tbodyDiag.appendChild(tr);
        });
      }

      // 8. Render Tier C Root Cause Analysis Table
      const tbodyRca = document.getElementById("tiercRcaTableBody");
      if (tbodyRca && data.tierc_root_cause_analysis && data.tierc_root_cause_analysis.length > 0) {
        tbodyRca.innerHTML = "";
        data.tierc_root_cause_analysis.forEach(row => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><strong>${row.Alloy}</strong></td>
            <td>${row.Total_Dataset_Samples.toLocaleString()}</td>
            <td>${row.Prevalence_None_Pct}%</td>
            <td>${row.Prevalence_Keyhole_Pct}%</td>
            <td><strong style="color:${row.Prevalence_Lack_of_Fusion_Pct > 50 ? '#fbbf24' : '#94a3b8'};">${row.Prevalence_Lack_of_Fusion_Pct}%</strong></td>
            <td>${row.Prevalence_Balling_Pct}%</td>
            <td>${row.Mean_Linear_Energy_Density_J_per_mm.toFixed(3)}</td>
            <td><strong style="color:${row.Thermal_Conductivity_k_W_mK > 100 ? '#f87171' : '#cbd5e1'};">${row.Thermal_Conductivity_k_W_mK}</strong></td>
            <td>${row.Optical_Absorptivity_A}</td>
            <td style="font-size:0.8rem;"><strong style="color:#fde68a;">${row.Primary_Root_Cause}</strong><br><span style="color:#94a3b8;">${row.Scientific_Remedy}</span></td>
          `;
          tbodyRca.appendChild(tr);
        });
      }

      // 9. Render Ablation Table
      if (data.ablation_study && data.ablation_study.length > 0) {
        const tbodyAbl = document.getElementById("ablationTableBody");
        if (tbodyAbl) {
          tbodyAbl.innerHTML = "";
          data.ablation_study.forEach(row => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
              <td><strong>${row.configuration}</strong></td>
              <td>${row.accuracy_pct.toFixed(2)}%</td>
              <td>${row.exact_match_ratio_pct.toFixed(2)}%</td>
              <td>${row.macro_f1.toFixed(4)}</td>
              <td>${row.inference_latency_per_sample_ms.toFixed(2)} ms</td>
            `;
            tbodyAbl.appendChild(tr);
          });
        }
      }

      // 10. Render Direct Paper Comparison Table (results/paper_comparison_v3.csv)
      const tbodyPaperComp = document.getElementById("paperComparisonTableBody");
      const paperCompData = data.paper_comparison_v3 || [];
      if (tbodyPaperComp && paperCompData.length > 0) {
        tbodyPaperComp.innerHTML = "";
        paperCompData.forEach(row => {
          const tr = document.createElement("tr");
          const isWinner = row.Winner && (row.Winner.includes("v3") || row.Winner.includes("AdditiveLLM"));
          const isGateMetric = row.Metric && row.Metric.includes("Gating");
          const passGate = row.Pass_Win_Condition;

          let gateBadge = `<span style="color:#34d399; font-weight:600;">PASS</span>`;
          if (!passGate && isGateMetric) {
            gateBadge = `<span style="color:#f87171; font-weight:700;">FAIL (Gating Shortfall)</span>`;
          } else if (!passGate) {
            gateBadge = `<span style="color:#fbbf24; font-weight:600;">Unmet</span>`;
          }

          if (isGateMetric && !passGate) {
            tr.style.background = "rgba(239, 68, 68, 0.08)";
          }

          tr.innerHTML = `
            <td><strong>${row.Metric}</strong></td>
            <td><strong style="color:#94a3b8;">${row.Paper_Headline}</strong></td>
            <td style="font-size:0.75rem; color:#64748b;">${row.Paper_Citation}</td>
            <td>${row.AdditiveLLM_v2 || "-"}</td>
            <td><strong style="color:${isWinner ? '#38bdf8' : '#cbd5e1'}; font-size:1.05rem;">${row.AdditiveLLM_v3}</strong></td>
            <td><strong style="color:${isWinner ? '#34d399' : '#f87171'};">${row.Winner}</strong></td>
            <td>${gateBadge}</td>
          `;
          tbodyPaperComp.appendChild(tr);
        });
      }

      // 11. Render Shortfall Diagnostic Table (results/v3_shortfall_diagnostic.csv)
      const tbodyShortfall = document.getElementById("shortfallDiagnosticTableBody");
      const shortfallData = data.v3_shortfall_diagnostic || [];
      if (tbodyShortfall && shortfallData.length > 0) {
        tbodyShortfall.innerHTML = "";
        shortfallData.forEach(row => {
          const tr = document.createElement("tr");
          const isShortfall = row.Status && row.Status.includes("SHORTFALL");
          const isRecovered = row.Status && (row.Status.includes("RECOVERED") || row.Status.includes("EXCEEDS"));
          const statusColor = isShortfall ? "#f87171" : (isRecovered ? "#34d399" : "#fbbf24");

          tr.innerHTML = `
            <td style="font-size:0.8rem; color:#94a3b8;">${row.Diagnostic_Dimension}</td>
            <td><strong>${row.Tier_or_Component}</strong></td>
            <td>${row.v2_Macro_F1 || "-"}</td>
            <td><strong style="color:#38bdf8;">${row.v3_Macro_F1}</strong></td>
            <td style="color:${row.Delta_F1 && row.Delta_F1.startsWith('+') ? '#34d399' : '#f87171'}; font-weight:600;">${row.Delta_F1}</td>
            <td style="font-size:0.8rem; color:#64748b;">${row.Target_Threshold}</td>
            <td><span style="color:${statusColor}; font-weight:700; font-size:0.8rem;">${row.Status}</span></td>
            <td style="font-size:0.75rem; color:#cbd5e1; max-width:280px;">${row.Root_Cause_Analysis}</td>
          `;
          tbodyShortfall.appendChild(tr);
        });
      }
    } catch (err) {
      console.warn("Could not load dashboard data yet:", err);
    }
  }

  // Initial updates & initial run
  updateSliderDisplays();
  submitInference();
});
