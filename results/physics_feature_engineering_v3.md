# Physics-Informed Feature Engineering & Dimensionless Scaling (v3)

## 1. Executive Summary & Root-Cause Targeting

In metal Laser Powder Bed Fusion (L-PBF), defect regimes (**Keyhole**, **Lack of Fusion**, **Balling**, **Conduction / Desirable**) are fundamentally governed by conservation of thermal energy and hydrodynamic fluid dynamics, rather than arbitrary textual tokens.

The v2 pipeline revealed that out-of-distribution (OOD) transfer to Tier C alloys (`IN625`, `AlSi10Mg`) suffered from two physical root causes:
1. **Severe Class Inversion in IN625**: 91.97% Lack of Fusion vs 6.69% None (opposite of SS316L/Ti-6Al-4V where None is 61–82%). Pure text or categorical embeddings inherit the majority alloy prior, causing models to predict None where Lack of Fusion actually dominates.
2. **Thermal Dissipation Outlier in AlSi10Mg**: Thermal conductivity $k = 150.0\text{ W/m}\cdot\text{K}$ (9x higher than SS316L, 22x higher than Ti-6Al-4V) and low optical absorptivity ($A = 0.18$). Heat dissipates into the powder bed almost an order of magnitude faster, dramatically altering melt pool geometry for identical laser parameters.

By engineering continuous thermophysical properties and dimensionless hydrodynamic ratios directly into the feature representation, the network learns defect regimes across the universal physical parameter space.

---

## 2. Thermophysical Property Features

All thermophysical properties are sourced from peer-reviewed metallurgical literature and materials handbooks (cataloged in [`results/physics_properties_sources.csv`](file:///Users/chandrashanmukhi/Downloads/AdditiveLLM_Reproduction/results/physics_properties_sources.csv)):

| Feature Name | Symbol | Units | Physical Rationale & Impact on Defect Formation |
| :--- | :---: | :---: | :--- |
| **Thermal Conductivity** | $k$ | $\text{W}/(\text{m}\cdot\text{K})$ | Dictates conductive heat dissipation rate from the melt pool into substrate. AlSi10Mg ($150.0$) vs SS316L ($16.3$). |
| **Optical Absorptivity** | $A$ | dimensionless | Effective fraction of laser power coupled into the powder bed at $\lambda \approx 1064\text{--}1070\text{ nm}$. IN625 ($0.65$) vs AlSi10Mg ($0.18$). |
| **Mean Melting Point** | $T_m$ | $\text{K}$ | Mean of solidus ($T_s$) and liquidus ($T_l$). Defines temperature threshold required for phase transition from powder to liquid. |
| **Density** | $\rho$ | $\text{kg}/\text{m}^3$ | Inertial resistance and volumetric mass density of the solid alloy. |
| **Specific Heat** | $C_p$ | $\text{J}/(\text{kg}\cdot\text{K})$ | Volumetric thermal capacity $\rho C_p$ required to raise alloy temperature per Kelvin. |
| **Thermal Diffusivity** | $\alpha = \frac{k}{\rho C_p}$ | $\text{m}^2/\text{s}$ | Thermophysical rate of temperature diffusion through the material. |

---

## 3. Hydrodynamic Aspect Ratios & Paper Equations (1)–(3)

Equations (1)–(3) from Pak & Barati Farimani (*Additive Manufacturing Letters*, 2025) define the fundamental physical boundaries between regimes:

### 3.1 Width-to-Depth Ratio (Keyhole Porosity Criterion, Eq. 1)
$$r_1 = \frac{w}{d}$$
- **Physical Boundary**: When $w/d < 1.5$ (melt pool depth exceeds $1.5 \times$ width), intense laser recoil pressure creates a narrow vapor depression (keyhole cavity). As the beam traverses, surface tension induces keyhole collapse, entrapping shielding gas voids.
- **Engineered Feature**: $w/d$ ratio and threshold violation distance $\max(0, 1.5 - w/d)$.

### 3.2 Geometrical Overlap (Lack of Fusion Porosity Criterion, Eq. 2)
$$\left(\frac{h}{w}\right)^2 + \left(\frac{t}{d}\right)^2 \le 1$$
- **Physical Boundary**: Where $h$ is hatch spacing, $t$ is layer thickness, $w$ is track width, and $d$ is penetration depth. If $\left(\frac{h}{w}\right)^2 + \left(\frac{t}{d}\right)^2 > 1$, adjacent scan tracks and successive powder layers fail to metallurgically coalesce, leaving un-melted powder cavities.
- **Engineered Feature**: $\Omega_{\text{LoF}} = \left(\frac{h}{w}\right)^2 + \left(\frac{t}{d}\right)^2$ and threshold exceedance $\max(0, \Omega_{\text{LoF}} - 1.0)$.

### 3.3 Length-to-Width Ratio (Balling Porosity Criterion, Eq. 3)
$$r_2 = \frac{l}{w}$$
- **Physical Boundary**: Rayleigh-Plateau hydrodynamic capillary instability dictates that a free liquid cylinder is unstable to varicose perturbations when its wavelength exceeds its circumference. In laser melting, when track length exceeds $\pi \times$ width ($l/w > \pi \approx 3.14$), the continuous cylindrical melt bead breaks up into isolated spherical droplets (balling).
- **Engineered Feature**: $l/w$ ratio and instability excess $\max(0, l/w - \pi)$.

---

## 4. Energy Densities & Dimensionless Scaling Numbers

### 4.1 Volumetric & Linear Energy Density
- **Linear Energy Density (LED)**:
  $$\text{LED} = \frac{P}{v} \quad [\text{J}/\text{mm}]$$
- **Volumetric Energy Density (VED)**:
  $$\text{VED} = \frac{P}{v \cdot h \cdot t} \quad [\text{J}/\text{mm}^3]$$
- **Areal Energy Density (AED)**:
  $$\text{AED} = \frac{P}{v \cdot h} \quad [\text{J}/\text{mm}^2]$$

### 4.2 Normalized Enthalpy ($\beta$) — Formula and Literature Source
To normalize laser power across diverse alloys with vastly different melting temperatures and thermal properties, we compute the normalized enthalpy $\beta$:
$$\beta = \frac{A \cdot P}{\pi \rho C_p (T_m - T_0) \sqrt{\alpha v d^3}}$$
- **Literature Source**:
  1. Rubenchik, A. M., King, W. E., & Wu, S. S. *"Scaling laws for the depth of keyhole and conduction mode laser melts"*, **Journal of Materials Processing Technology**, 256, 172-181 (2018). [DOI: 10.1016/j.jmatprotec.2018.01.037](https://doi.org/10.1016/j.jmatprotec.2018.01.037)
  2. King, W. E., Barth, H. D., Castillo, V. M., Gallegos, G. F., Gibbs, J. W., Hahn, D. E., Kamath, C., & Rubenchik, A. M. *"Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing"*, **Journal of Materials Processing Technology**, 214(12), 2915-2925 (2014). [DOI: 10.1016/j.jmatprotec.2014.06.005](https://doi.org/10.1016/j.jmatprotec.2014.06.005)
- **Physical Rationale**: Keyhole transition is characterized by $\beta > \beta_{\text{crit}} \approx 6.0\text{--}8.0$, providing an alloy-independent threshold that normalizes both IN625 and AlSi10Mg.

### 4.3 Péclet Number ($Pe$) — Formula and Literature Source
The thermal Péclet number describes the ratio of advective heat transport (due to scan velocity) to conductive heat dissipation:
$$Pe = \frac{v \cdot d}{\alpha} = \frac{v \cdot d \cdot \rho C_p}{k}$$
- **Literature Source**:
  1. Rosenthal, D. *"The theory of moving sources of heat and its application to metal treatments"*, **Transactions of ASME**, 68, 849-866 (1946).
  2. DebRoy, T. et al. *"Additive manufacturing of metallic components – Process, structure and properties"*, **Progress in Materials Science**, 92, 112-224 (2018). [DOI: 10.1016/j.pmatsci.2017.10.001](https://doi.org/10.1016/j.pmatsci.2017.10.001)
- **Physical Rationale**: For high thermal conductivity alloys like AlSi10Mg ($k=150$), $Pe$ is low, indicating conduction dominates; for titanium and nickel alloys, $Pe$ is higher, indicating advective elongation of the melt pool.

---

## 5. Isolation of $\beta$ and $Pe$ in the Ablation Suite

Per user instructions, $\beta$ and $Pe$ are **not bundled** into a single feature claim. In [`results/physics_feature_ablation.csv`](file:///Users/chandrashanmukhi/Downloads/AdditiveLLM_Reproduction/results/physics_feature_ablation.csv), they are evaluated in two distinct rows:
1. **Row 1 (Original Specification Features)**: $k, A, T_s, T_l, \rho, C_p$, $w/d$, LoF ratio, $l/w$, VED, LED.
2. **Row 2 ($\beta$ and $Pe$ Added on Top)**: Evaluates whether normalized enthalpy and Péclet number provide statistically separable, non-redundant lift beyond VED and LED.
