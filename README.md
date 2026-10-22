# AI-Based DDoS Detection: Hybrid Detection and Cross-Dataset Generalization

**A research project investigating known-attack detection, zero-day detection, data leakage, and cross-dataset generalization in machine-learning-based DDoS intrusion detection systems.**

## Overview

Machine-learning-based Distributed Denial-of-Service (DDoS) detection systems frequently report near-perfect performance when evaluated on a held-out split of the same dataset used for training. However, such results do not necessarily indicate that a model will generalize to traffic collected from a different network or capture environment.

This research investigates that gap through a three-part experimental study. It develops and evaluates supervised classifiers, an unsupervised anomaly detector, and a two-stage hybrid detection framework, followed by cross-dataset evaluation and feature-level diagnostics.

The experiments use three publicly available benchmark datasets:

- CIC-DDoS2019
- CSE-CIC-IDS2018
- CIC-IDS2017

The central focus is not simply maximizing classification accuracy. It is understanding how detection performance changes under unseen attack types and dataset shift, how train/test overlap affects reported results, and which feature-level behaviors are associated with cross-dataset performance degradation.

## Research Objectives

- Develop a compact feature representation for network-flow-based DDoS detection.
- Compare supervised classification, anomaly detection, and hybrid detection approaches.
- Evaluate detection performance on known attacks and attack types excluded from training.
- Quantify the impact of exact train/test feature-vector overlap on evaluation metrics.
- Test cross-dataset generalization without retraining or adapting models to the target dataset.
- Investigate feature-distribution shift using statistical tests and feature-level diagnostics.
- Analyze model behavior using SHAP feature attribution.
- Assess the stability of results through repeated experiments across multiple random seeds.

## Experimental Framework

The research is organized into three complementary parts.

### Part I: Within-Dataset Detection and Zero-Day Evaluation

CIC-DDoS2019 is used for model development and within-dataset evaluation.

The experimental pipeline includes:

1. Data cleaning, label normalization, and integrity checks.
2. Mutual-information-based feature selection, reducing 77 candidate features to 20.
3. Multiclass and binary classification using Decision Tree, Random Forest, XGBoost, and LightGBM.
4. Isolation Forest anomaly detection trained exclusively on benign traffic.
5. A two-stage hybrid detector combining anomaly detection with a classifier and an explicit `Unknown Attack` rejection category.
6. Evaluation on known-attack, zero-day-style, and combined open-set test conditions.
7. Feature-count ablation, train/test overlap auditing, and SHAP-based interpretability analysis.

Five attack types, SNMP, NTP, DNS, WebDDoS, and TFTP, are held out from supervised training to construct a zero-day-style evaluation.

### Part II: Cross-Dataset Generalization

The binary classifiers trained on CIC-DDoS2019 are frozen and evaluated on CIC-IDS2017.

No retraining or target-domain threshold recalibration is used in the primary external evaluation. The purpose is to measure how the original models behave when applied to a different dataset.

The analysis includes:

- Internal versus external performance comparison.
- Kolmogorov–Smirnov tests for feature-distribution differences.
- Sentinel-value auditing for potentially problematic feature representations.
- Leave-one-feature-out sensitivity analysis.
- Decision-threshold transfer experiments using source-domain validation data only.

### Part III: Independent External Validation and Robustness

The cross-dataset evaluation is extended to CSE-CIC-IDS2018 to test whether the observed failure pattern is specific to a single external dataset.

This stage includes:

- External evaluation of frozen classifiers on a second dataset.
- Five-seed repeated experiments with uncertainty reporting.
- Cross-dataset SHAP attribution analysis.
- Comparison of model-specific responses to changes in feature distributions.

The analysis distinguishes observed results, statistical associations, model-sensitivity findings, feature attributions, and untested hypotheses. It does not treat feature sensitivity or SHAP attribution as proof of causality.

## Models and Techniques

| Component | Methods |
|---|---|
| Supervised classification | Decision Tree, Random Forest, XGBoost, LightGBM |
| Anomaly detection | Isolation Forest |
| Hybrid detection | Isolation Forest followed by a classifier with unknown-attack rejection |
| Feature selection | Mutual information |
| Distribution-shift analysis | Two-sample Kolmogorov–Smirnov tests |
| Feature diagnostics | Sentinel-value auditing and leave-one-feature-out analysis |
| Explainability | SHAP TreeExplainer |
| Robustness analysis | Five-seed evaluation, bootstrap intervals, and confidence intervals |

## Key Findings

The results illustrate why strong benchmark performance should not be interpreted in isolation.

### 1. High within-dataset performance does not guarantee generalization

The binary classifiers achieved validation F1-scores of at least 0.9995 on CIC-DDoS2019. When evaluated without adaptation on CIC-IDS2017, their performance deteriorated sharply.

Across five random seeds, Random Forest achieved a mean external recall of approximately 0.070, with a 95% confidence interval of 0.036–0.104. XGBoost recorded zero external recall across all five seeds under the evaluated protocol.

The same qualitative pattern appeared on CSE-CIC-IDS2018, although the magnitude of degradation differed by model and dataset.

### 2. Train/test overlap materially affects evaluation

An integrity audit identified exact duplication of selected feature vectors between the training and known-attack test sets.

Approximately 13.93% of the original known-attack test set overlapped with training data in the selected 20-feature representation. After excluding overlapping samples, multiclass macro-F1 decreased from approximately 0.550 to 0.509, and several binary classification results also declined.

This demonstrates the importance of explicitly auditing train/test overlap when evaluating intrusion detection systems.

### 3. Hybrid detection reduces false alarms in open-set evaluation

Within CIC-DDoS2019, the two-stage hybrid detector achieved a zero-day recall of approximately 98.82% and a benign false-positive rate of approximately 0.019% at the reported operating point.

Compared with Isolation Forest alone, the hybrid reduced the observed open-set false-positive rate from approximately 7.06% to 0.019%, a reduction of roughly 370-fold, while sacrificing a small amount of aggregate zero-day recall.

These results are specific to the evaluated within-dataset protocol and do not establish that the hybrid generalizes to external datasets.

### 4. Feature-level diagnostics reveal potential sources of failure

Statistical tests identified substantial distributional differences across the 20 shared features between the source and external datasets.

A leave-one-feature-out experiment found that removing `Init Fwd Win Bytes`, a feature affected by sentinel-value behavior, increased external F1 from approximately 0.135 to 0.526 in the evaluated configuration.

This is a model-sensitivity finding, not proof that the feature alone causes cross-dataset failure. Further controlled feature harmonization is needed to establish the underlying mechanism.

### 5. Threshold recalibration alone is insufficient

Experiments using thresholds selected exclusively from source-domain validation data did not reliably improve external performance. In the evaluated configurations, recalibration could make external recall worse.

This highlights the limitations of relying on decision-threshold adjustment alone when the underlying feature distributions differ between datasets.

*All reported values correspond to specific experimental configurations described in the accompanying research documentation. Results should be interpreted alongside the relevant evaluation protocol, overlap-excluded analyses, and seed-repeated estimates.*

## Evaluation Metrics

The study uses metrics appropriate for binary classification, multiclass classification, and anomaly detection:

- Accuracy
- Precision
- Recall
- F1-score and macro-F1
- ROC-AUC
- Confusion matrices
- Zero-day detection rate
- Benign false-positive rate

For repeated experiments, mean performance, standard deviation, and 95% confidence intervals are used where applicable. Per-attack detection rates are also examined to avoid allowing aggregate results to conceal individual attack-type failures.

## Repository Structure

```text
DDOS_Detection_ML/
│
├── code.ipynb
├── experimentation_documentation_v3.md
├── requirements.txt
├── README.md
│
├── Datasets/          # Local datasets, excluded from Git
└── research_results/  # Generated results, excluded from Git
```

The notebook contains the experimental workflow, while the accompanying Markdown document provides supplementary experimentation documentation.

## Getting Started

### Prerequisites

- Python
- pip
- Jupyter Notebook
- The dependencies listed in `requirements.txt`
- Local copies of the required benchmark datasets

### 1. Clone the repository

```bash
git clone https://github.com/Atharvagaur/DDOS_Detection_ML.git
cd DDOS_Detection_ML
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the datasets

The datasets are not included in this repository. Obtain the required data from the original Canadian Institute for Cybersecurity sources, subject to their applicable access and licensing terms.

Place the required files in the appropriate local directories and update the dataset paths in the notebook as needed.

The CIC-DDoS2019, CIC-IDS2017, and CSE-CIC-IDS2018 inputs must be processed according to the feature mapping, cleaning, and label-normalization rules used in the experiments.

### 5. Run the experiments

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `code.ipynb` and execute the cells in the required order. Review `experimentation_documentation_v3.md` for additional details about the experimental procedures.

## Limitations

The findings are bounded by the experimental design:

- All three datasets belong to the CIC/CSE-CIC benchmark family and use CICFlowMeter-derived flow features.
- The cross-dataset experiments do not establish generalization to arbitrary operational networks.
- The study is based on offline datasets rather than live network traffic.
- The hybrid detector has not been externally validated in the same manner as the frozen binary classifiers.
- The feature-removal and SHAP analyses identify model sensitivities and attribution patterns, not causal mechanisms.
- The study does not establish production deployment readiness or robustness to temporal drift and evolving attack behavior.

## Future Work

Potential extensions include:

- Controlled feature harmonization and sentinel-value intervention experiments.
- Evaluation on datasets collected outside the CIC/CSE-CIC family.
- Domain adaptation and target-domain calibration under explicitly defined access constraints.
- Improved handling of difficult low-volume attack types such as WebDDoS.
- Calibrated abstention and selective classification for unknown attacks.
- Live or replayed-traffic evaluation under realistic latency and throughput requirements.

## Research Context

This repository supports an ongoing journal research study titled:

*AI-Based DDoS Detection System: A Hybrid Classifier-Anomaly Framework for Known-Attack, Zero-Day, and Cross-Dataset Generalization Evaluation.*

The project emphasizes reproducible experimentation, leakage-aware evaluation, and careful interpretation of model behavior rather than benchmark accuracy alone.

## Author

**Atharva Gaur**

GitHub: [@Atharvagaur](https://github.com/Atharvagaur)

## Data and Code Availability

The benchmark datasets are obtained separately from their original sources and are not distributed through this repository.

The repository is intended to serve as the public research-code companion to the study. Before the journal submission is finalized, the code, preprocessing and overlap-exclusion routines, feature-mapping configuration, model parameters, random seeds, evaluation scripts, environment specification, and result-generation artifacts should be checked for completeness and consistency with the manuscript.

## Contributors

- [Atharva Gaur](https://github.com/Atharvagaur)
- [Krish N Doshi](https://github.com/kndoshi)
