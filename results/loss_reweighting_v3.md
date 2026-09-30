# Cluster-Aware Focal Loss & Class Distribution Reweighting (v3)

## 1. Executive Summary & Problem Formulation

Standard cross-entropy loss with global class weights assumes that defect distributions are uniform or identically skewed across all metal alloys. In multi-material Laser Powder Bed Fusion (L-PBF), this assumption fails catastrophically:
- Majority alloys (**SS316L** and **Ti-6Al-4V**, representing >85% of standard training datasets) exhibit a strong **None / Desirable** dominance (61.0% to 82.1% None, 17.9% to 34.1% Lack of Fusion, <1.5% Keyhole).
- In contrast, out-of-distribution superalloy **IN625** exhibits an extreme **inverted distribution**: **91.97% Lack of Fusion** and only **6.69% None**.

Under standard global loss weighting, the overwhelming volume of majority-alloy instances forces gradients to minimize the false-positive rate on "None", penalizing predictions of Lack of Fusion unless process parameters are drastically suboptimal. When applied to IN625, this induced prior leads to widespread false negatives on Lack of Fusion, severely crippling Tier C macro F1 (which weights all 4 classes equally).

To rectify this without destroying Tier A/B accuracy, we implement a **Cluster-Aware Focal Loss** where loss weighting is conditioned on alloy clusters.

---

## 2. Mathematical Formulation

For sample $i$ with true one-hot/multi-label target $y_{i, c} \in \{0, 1\}$ and predicted probability $p_{i, c} \in [0, 1]$ for defect class $c \in \{\text{None}, \text{Keyhole}, \text{LoF}, \text{Balling}\}$:

$$\mathcal{L}_{\text{Cluster-Focal}} = -\sum_{c=1}^4 \alpha_{c, m} \left(1 - p_{i, c}^{(t)}\right)^\gamma \log\left(p_{i, c}^{(t)}\right)$$

where:
- $p_{i, c}^{(t)} = p_{i, c}$ if $y_{i, c} = 1$, and $(1 - p_{i, c})$ if $y_{i, c} = 0$.
- $\gamma \ge 0$ is the focal focusing parameter (set to $\gamma = 2.0$), down-weighting easy, well-classified examples.
- $\alpha_{c, m}$ is the cluster-conditioned class weight for material cluster $m = \text{cluster}(i)$:
  $$\alpha_{c, m} = \frac{1}{\left(f_{c, m} + \epsilon\right)^\nu}$$
  with class frequency $f_{c, m}$ within cluster $m$, smoothing exponent $\nu = 0.5$, and $\epsilon = 10^{-4}$.

---

## 3. Physical & Distribution-Aware Alloy Clustering

Alloys are partitioned into distinct operational clusters:

| Cluster | Cluster Name | Constituent Materials in Dataset | Thermophysical Profile | Label Distribution Characteristic |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Ferrous & Nickel Matrix** | `SS316L`, `IN718`, `Hastelloy X`, `SS17-4PH`, `IN738LC`, `Invar36`, `SS304` | Moderate conductivity ($k \approx 9\text{--}18\text{ W/m}\cdot\text{K}$), $A \approx 0.39\text{--}0.44$, $T_m \approx 1540\text{--}1700\text{ K}$ | Balanced-to-Desirable (None 61–65%, LoF 34–37%, Keyhole 14–22%) |
| **2** | **Titanium Alloys** | `Ti-6Al-4V`, `Ti-49Al-2Cr-2Nb`, `Ti6242`, `Ti-45Al` | Low conductivity ($k \approx 6.7\text{ W/m}\cdot\text{K}$), $A \approx 0.38$, High melting point ($T_m \approx 1900\text{ K}$) | Heavily Desirable-Dominated (None 82.1%, LoF 17.9%, Keyhole 1.4%) |
| **3** | **Extreme Class-Inversion Nickel** | `IN625` | Moderate conductivity ($k = 9.8\text{ W/m}\cdot\text{K}$), Elevated absorptivity ($A = 0.65$), $T_m = 1593\text{ K}$ | **Severe Inversion: 91.97% Lack of Fusion**, 6.69% None, 6.19% Keyhole, 46.39% Balling |
| **4** | **Thermal Outliers & Reflective Alloys** | `AlSi10Mg`, `Cu10Sn`, `HCP Cu`, `Al-2.5Fe` | Extreme conductivity ($k = 64\text{--}150\text{ W/m}\cdot\text{K}$), Low absorptivity ($A = 0.18\text{--}0.22$), Low $T_m$ ($850\text{--}1200\text{ K}$) | High Keyhole/LoF sensitivity; rapid heat dissipation into substrate |

---

## 4. Explicit Justification for IN625 Cluster Split (Cluster 3)

> [!IMPORTANT]
> **Why IN625 is Separated into Cluster 3**:
> Metallurgically, Inconel 625 is a nickel-chromium-molybdenum superalloy closely related to Inconel 718 and Hastelloy X. Its bulk thermophysical properties ($k = 9.8\text{ W/m}\cdot\text{K}$, $T_m = 1593\text{ K}$) are very similar to Cluster 1.
> 
> However, IN625 is explicitly placed in its own cluster (**Cluster 3**) **because of its extreme label-distribution inversion** (91.97% Lack of Fusion vs 6.69% None). In standard training datasets, nickel and ferrous alloys operate in parameter regimes where "None" is the prevailing mode (61–65%). In contrast, experimental builds of IN625 in the benchmark dataset were conducted in parameter windows deliberately probing the onset of incomplete fusion.
> 
> Grouping IN625 with Cluster 1 based solely on metallurgy would force IN625 to inherit Cluster 1's class weights ($\alpha_{\text{None}} \approx 1.2, \alpha_{\text{LoF}} \approx 1.6$), severely under-weighting Lack of Fusion for IN625. Assigning IN625 its own cluster provides the exact compensatory weighting ($\alpha_{\text{LoF}, m=3} \approx 1.04, \alpha_{\text{None}, m=3} \approx 3.87$), penalizing false None predictions on IN625 without altering the loss landscape for SS316L or Ti-6Al-4V.

---

## 5. Control Comparison in the Ablation Suite

To isolate whether the performance gain is driven by the physical clustering or specifically by isolating IN625's inverted distribution, the ablation suite evaluates:
1. **Control 1: Standard Global Cross-Entropy** (Flat class weighting across the entire dataset).
2. **Control 2: 3-Cluster Thermophysical Baseline** (IN625 merged into Cluster 1 based purely on thermophysical properties, with no distribution-aware override).
3. **Proposed 4-Cluster Distribution-Aware Focal Loss** (IN625 isolated into Cluster 3).

All three conditions are evaluated across identical splits and random seeds in [`results/loss_ablation_v3.csv`](file:///Users/chandrashanmukhi/Downloads/AdditiveLLM_Reproduction/results/loss_ablation_v3.csv).
