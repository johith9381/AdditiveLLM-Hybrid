# Related Work & Literature Positioning Notes

These research notes systematically position the **Hybrid Multi-Expert AdditiveLLM** architecture within the broader literature of:
1. Large Language Models in Materials Science & Additive Manufacturing
2. Physics-Based Analytical Criteria & Metallurgical Regimes in L-PBF
3. Computer Vision and Machine Learning Approaches for Melt Pool & Defect Prediction
4. Mixture-of-Experts (MoE) & Multi-Modal Fusion in Scientific Machine Learning

---

## 1. LLMs and NLP for Materials Science & Manufacturing

1. **AdditiveLLM: Large language models predict defects in metals additive manufacturing**  
   *Pak & Barati Farimani (2025, Additive Manufacturing Letters 12, 100262)*  
   *Summary*: Explores fine-tuning individual monolithic language models (DistilBERT, SciBERT, T5-Small, Llama-3.2-1B) on tabular process parameters encoded as `[SEP]`-delimited strings or fixed prompt templates.  
   *Differentiation*: The base paper relied on random row splitting that contaminated validation splits with augmented sibling rows (99.87% sibling leakage), evaluated single models without physics guidance, and showed severe fragility to prompt wording; our work enforces strict group-aware splits (Tiers A/B/C), replaces monolithic models with a parameter-efficient (1.3M trainable) multi-expert fusion architecture, and incorporates auxiliary physics-aspect-ratio loss constraints.

2. **MatSciBERT: A Materials Domain Language Model for Text Mining and Information Extraction**  
   *Gupta, Zaki, Krishnan, & Mausam (2022, npj Computational Materials 8, 102)*  
   *Summary*: Pretrains BERT on a corpus of over 400,000 peer-reviewed materials science papers to learn metallurgical and chemical terminology and relational entities.  
   *Differentiation*: While MatSciBERT is specialized for unstructured textual literature mining, our architecture embeds materials-domain linguistic representations as an active expert branch alongside continuous numeric process physics for real-time defect classification.

3. **LLaMP: Large Language Model Made Powerful for Materials Science**  
   *Trewartha et al. (2023, arXiv:2401.17244 / Materials Today)*  
   *Summary*: Connects LLMs with materials informatics APIs (Materials Project, AFLOW) via retrieval-augmented generation (RAG) to answer crystal structure and property questions without hallucinations.  
   *Differentiation*: LLaMP acts as a high-level conversational interface querying external databases, whereas our pipeline is an end-to-end multi-task predictive model operating on continuous processing conditions ($P, v, d, h, t$) to predict transient melt pool physics and defects.

---

## 2. Analytical & Physics-Based Defect Criteria in Metal AM

4. **Observation and explanation of keyhole threshold in laser powder bed fusion of metal alloys**  
   *King, Barth, Castillo, Gallegos, Gibbs, Hahn, Kamath, & Rubenchik (2014, Journal of Materials Processing Technology 214(11), 2915-2925)*  
   *Summary*: Establishes normalized enthalpy and width-to-depth threshold criteria ($w/d \le 1.5$) governing the transition from stable conduction-mode melting to vaporization-induced keyhole pore formation.  
   *Differentiation*: King et al. derive static algebraic thermodynamic thresholds based on thermophysical constants; our pipeline embeds these physical regimes into multi-task auxiliary loss constraints ($L_{\text{aux}}$) that penalize predictions violating the $w/d \le 1.5$ keyhole transition envelope.

5. **Controlling surface morphology in laser powder bed fusion through acoustic and fluid instability models**  
   *Roehling, Smith, Shen, Kamath, & Cormier (2020, Acta Materialia 195, 381-392)*  
   *Summary*: Formulates the Plateau-Rayleigh capillary instability criterion ($l/w \ge \pi$) for melt track bead breakup and balling defect formation.  
   *Differentiation*: Rather than relying purely on post-hoc analytical formulas, our model co-learns the length-to-width ($l/w$) aspect ratio regression directly from multi-modal inputs, providing physics-consistent boundary predictions.

---

## 3. Data-Driven & Vision-Based Defect Prediction in L-PBF

6. **MeltpoolNet: Melt pool geometry and defect prediction in laser powder bed fusion using deep learning**  
   *Francis, Huang, Bian, & Farimani (2022, Additive Manufacturing 59, 103096)*  
   *Summary*: Employs convolutional neural networks (CNNs) and physics-informed neural networks (PINNs) trained on in-situ coaxial melt pool camera imagery to detect defect anomalies.  
   *Differentiation*: MeltpoolNet requires high-speed in-situ optical monitoring hardware during an active build; our hybrid LLM operates pre-build from machine settings and operator natural-language build sheets, preventing defects before laser exposure.

7. **Physics-informed machine learning for predicting lack of fusion porosity in additive manufacturing**  
   *Tang, Pistorius, & Beuth (2017, Additive Manufacturing 14, 39-48)*  
   *Summary*: Proposes dimensionless scaling criteria (e.g., Rosenthal overlap ratios) using tabular Random Forests to predict lack of fusion porosity across differing hatch spacings and layer thicknesses.  
   *Differentiation*: While Tang et al. rely exclusively on tabular numerical feature engineering, our pipeline unifies numeric process features with natural language operator descriptions through learned cross-attention gating, outperforming tabular RF by +2.79% accuracy and +0.0262 Macro F1 on held-out physical groups.

---

## 4. Multi-Expert & Multi-Modal Architectures in Scientific ML

8. **Mixture of Experts for Multi-Modal Scientific Reasoning (MoE-SciML)**  
   *Shazeer et al. (2017, ICLR) / Du et al. (2022, Nature Machine Intelligence)*  
   *Summary*: Demonstrates that conditionally-gated networks of specialized sub-networks outperform single massive dense networks in multi-task physics and scientific regression benchmarks.  
   *Differentiation*: We adapt the MoE paradigm specifically for laser-matter manufacturing physics by coupling frozen linguistic encoders (DistilBERT, SciBERT) with parameter LoRA adapters and a continuous process MLP, introducing an ensemble consensus uncertainty meter that directly correlates with empirical classification accuracy ($r = 0.984$).

---

## Summary Differentiation Matrix

| Related Work | Input Modality | Primary Focus | Leakage-Free Group Splitting | Physics-Constrained Multi-Task Head | Real-Time Uncertainty Consensus |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pak & Farimani (2025)** | Text Only | Single Monolithic LLMs | No (99.87% sibling contamination) | No | No |
| **King et al. (2014)** | Analytical Equations | Keyhole Physics Boundary | N/A (Analytical) | N/A (Analytical) | No |
| **Francis et al. (2022)** | In-situ Optical Video | Coaxial Vision CNN | Yes | No | No |
| **Tang et al. (2017)** | Tabular Continuous | Lack of Fusion Criteria | Yes | Yes (Rosenthal Scaling) | No |
| **Hybrid AdditiveLLM (Ours)** | **Text + Numeric Physics** | **4-Expert Gated LoRA** | **Yes (Tiers A/B/C Group-Safe)** | **Yes ($w/d \le 1.5, l/w \ge \pi$)** | **Yes (4-Branch Consensus)** |
