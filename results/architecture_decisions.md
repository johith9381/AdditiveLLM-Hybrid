# AdditiveLLM: Architecture Decisions & Tier C Generalization Rationale

## 1. Executive Summary of Architectural Evolution

| Evolution Stage | Core Design | Trainable Params | Tier A Acc / F1 | Tier B Acc / F1 | Tier C Acc / F1 | Limitation Identified |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Stage 1 (Paper Baseline)** | Frozen DistilBERT / Llama Linear Probing | 1.2M - 1.2B | 88.4% / 0.796 | N/A (Leaked) | N/A (Leaked) | Severe 99.87% sibling row contamination |
| **Stage 2 (4-Branch Gated)** | DistilBERT + SciBERT + MatSci + Categorical MLP | 1.30M | 97.14% / 0.9457 | 97.29% / 0.9063 | 57.44% / 0.1973 | Overfitted to discrete material embeddings |
| **Stage 3 (Physics-First)** | Continuous Physics + Dual Transformer + Hydrodynamic Residuals | 518K | 97.42% / 0.9514 | 97.06% / 0.9019 | 60.37% / 0.2393 | Improved, but limited by alloy diversity in raw data |

---

## 2. Key Architectural Decisions

### Decision 1: Elimination of Learned Categorical Material Embeddings
- **Rationale**: The material dependency ablation revealed that zeroing learned material embeddings caused a severe performance drop on Tier C ($\Delta F1 = +0.0866$), proving the classification head had learned to rely on alloy identity lookup rather than physical parameter scaling.
- **Implementation**: The categorical embedding layer was entirely eliminated. Process parameters $(P, v, d, h, t)$ are normalized and passed through a continuous physical feature encoder (`ContinuousProcessPhysicsEncoder`).

### Decision 2: Explicit Hydrodynamic Geometry Engine & Residual Violation Coupling
- **Rationale**: In metal laser powder bed fusion, defect boundaries are governed by physical dimensionless aspect ratios:
  - Keyhole mode: $w/d \le 1.5$ (vaporization-induced deep depression).
  - Balling mode: $l/w \ge \pi$ (Plateau-Rayleigh capillary elongation instability).
  - Conduction mode: $w/d > 1.5$ and $l/w < \pi$ (stable melt pool).
- **Implementation**: The network predicts auxiliary geometry ratios $[r_1 = w/d, r_2 = l/w]$ via multi-task loss ($\lambda = 0.5$). These predicted ratios are injected directly into the defect logits via physical ReLU residual activations:
  $$P(	ext{Keyhole}) = \sigma(W_{	ext{cls}} \cdot h + \gamma_1 \cdot 	ext{ReLU}(1.5 - \hat{r}_1))$$
  $$P(	ext{Balling}) = \sigma(W_{	ext{cls}} \cdot h + \gamma_2 \cdot 	ext{ReLU}(\hat{r}_2 - \pi))$$

### Decision 3: Empirical Uncertainty & Reliability Signal
- **Rationale**: Operators require a transparent confidence metric that monotonically correlates with true physical accuracy.
- **Implementation**: Distance from the 0.5 decision boundary across defect classes ($C = rac{1}{4} \sum |p_i - 0.5| 	imes 2$) was empirically validated. Predictions in the high-confidence bucket ($>0.85$) achieve **>98% empirical accuracy**, while borderline predictions ($<0.60$) reflect physical boundary transitions.

---

## 3. Honest Scientific Statement on Tier C Generalization

> [!IMPORTANT]
> **Data-Availability Limitation**: While the Physics-First architecture successfully improves Tier C Baseline performance from 57.54% to 60.37% ($p = 0.0096$) and Prompt performance to 70.06% ($p = 0.0022$), Tier C remains fundamentally lower than Tiers A and B (~97% accuracy).
> 
> This gap is **thermodynamic and data-constrained**: the dataset contains only 13 commercial alloys. Thermophysical properties—specifically optical absorptivity $A$ (which ranges from 0.12 in copper to 0.45 in tungsten) and thermal conductivity $k$ (which varies by 3,000%)—cannot be inferred zero-shot without broad alloy distribution in training data. True material-level out-of-distribution transfer requires expanding the experimental library across diverse alloy families, not further model complexity.
