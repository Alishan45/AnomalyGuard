# 🛡️ AnomalyGuard — Transaction Anomaly Detection & Operational Review
> **Stack & Scale | Remote Internship 2026 — AI Engineering Week 03**  
> *A reproducible unsupervised machine learning pipeline, threshold sensitivity framework, and interactive operational review dashboard for payment anomaly triage.*

---

## 📌 Executive Summary & Real-World Challenge
Financial fraud prevention and operations teams process millions of electronic transactions daily. While automated blacklists catch known attack vectors, emerging irregular behaviors—such as automated credential-stuffing velocity bursts, geographically impossible card usage, and high-risk overseas luxury transactions—require unsupervised detection.

### ⚠️ Core Operational Principle
> **An anomaly is an operational review signal, not proof of fraud.**  
> Unsupervised models identify statistical rarities and behavioral outliers. An anomaly score serves as a triage signal for human analysts or automated multi-factor authentication (MFA) challenges; it does **never** constitute proof of malicious or fraudulent intent on its own.

---

## 🏗️ Project Architecture & Repository Structure

```
├── app.py                     # Interactive Streamlit Web Application (Hosting Ready)
├── data/
│   ├── generate_dataset.py    # Realistic synthetic financial transaction generator
│   └── transactions.csv       # 10,000 transactions with documented anomaly scenarios
├── figures/
│   ├── eda_distributions.png          # Monday EDA distribution evidence
│   ├── isolation_forest_scores.png    # Wednesday Isolation Forest score calibration
│   └── threshold_sensitivity.png      # Thursday Threshold sensitivity curves
├── notebooks/
│   └── AnomalyGuard_Pipeline.ipynb    # Clean, executed top-to-bottom Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── eda.py                 # Summary statistics, skewness, and correlation matrix
│   ├── preprocessing.py       # Domain feature engineering & RobustScaler without data leakage
│   ├── baseline.py            # Tuesday Rule-based & statistical IQR/Z-score baseline
│   ├── models.py              # Isolation Forest, LOF, and Seed Stability Evaluator
│   └── evaluation.py          # Threshold sensitivity, F1/PR-AUC metrics, and explainability
├── .gitignore
├── requirements.txt           # Production dependencies
└── README.md                  # Comprehensive technical documentation & case study
```

---

## 🗓️ Five-Day Implementation Roadmap

### 📅 Monday — Dataset & Exploratory Analysis
- **Dataset Provenance**: Curated dataset of 10,000 financial transactions across 8 merchant categories (`grocery`, `dining`, `utilities`, `transport`, `entertainment`, `electronics`, `travel`, `luxury_retail`) and 5 channels (`pos_contactless`, `in_store`, `mobile_app`, `web`, `atm`).
- **Exploratory Findings**:
  - `amount`: Heavy right-skewed log-normal distribution (median \$45.20, max \$8,500+).
  - `device_risk_score`: Beta distribution centered at low risk (0.01 - 0.25).
  - `time_since_last_txn_hours` & `distance_from_last_txn_km`: Used to analyze physical displacement feasibility.
- **Review Objective**: Flag unusual transaction vectors for human review while minimizing false-positive analyst fatigue.

### 📅 Tuesday — Preprocessing, Feature Engineering & Baseline
- **Data Leakage Prevention**: Ground-truth anomaly markers (`is_synthetic_anomaly`) are strictly isolated and never fed into models or scaling estimators during training.
- **Engineered Signals**:
  1. **Travel Velocity ($\text{km/h}$)**: $v = \frac{\Delta \text{distance}}{\Delta \text{time} + \epsilon}$. Flags impossible physical travel (>800 km/h).
  2. **Variance Stabilization**: $\log(1 + x)$ transformations applied to transaction amount, home distance, and velocity.
  3. **Risk Interaction Term**: $\text{device\_risk} \times (\text{failed\_attempts} + 1)$.
  4. **Robust Scaling**: `RobustScaler` scales features via median and interquartile range (IQR), ensuring extreme outliers do not bias normalization centers.
- **Statistical / Rule-Based Baseline**:
  - Evaluates univariate $Z\text{-scores} > 3.0$, extreme upper IQR fences ($Q_3 + 3.0 \times \text{IQR}$), and heuristic rules (velocity $>800\text{ km/h}$, device risk $>0.70$ with $\ge 2$ failed attempts).
  - Baseline flags **389 transactions (3.89%)**, capturing obvious outliers but lacking multivariate sensitivity.

### 📅 Wednesday — Isolation Forest Model & Scoring
- **Algorithm Architecture**: Ensemble of 150 Isolation Trees isolating atypical points via random axis-aligned splits.
- **Score Calibration**: Decision function raw values are inverted and min-max calibrated to a uniform $[0.0, 1.0]$ range, where $1.0$ represents the most anomalous observations.
- **Explainability**: For every flagged transaction, feature deviations are compared against population medians to provide operational review drivers (e.g., `travel_velocity elevated (2450.0 km/h vs med 2.1 km/h)`).

### 📅 Thursday — Threshold Sensitivity & Evaluation Trade-offs
- **Sensitivity Spectrum**: Evaluated percentile thresholds from 85th to 99.5th percentile:
  - *85th Percentile*: 1,500 flagged (High operational load, 100% recall, 16.7% precision).
  - *95th Percentile*: 500 flagged (Balanced monitoring, 100% recall, 50.0% precision).
  - *97.5th Percentile (Contamination 2.5%)*: **250 flagged (Optimal operational balance: 98.4% Recall, 98.4% Precision, 0.984 F1-Score)**.
- **Benchmark Evaluation Metrics**:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | Flagged Records |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based Baseline** | 0.8842 | 0.7610 | 0.6195 | 0.9640 | 0.7543 | 389 |
| **Isolation Forest (Ours)** | **0.9999** | **0.9955** | **0.9840** | **0.9840** | **0.9840** | **250** |
| **Local Outlier Factor (LOF)** | 0.5019 | 0.2054 | 0.2511 | 0.2240 | 0.2368 | 223 |

### 📅 Friday — Governance, Reproducibility & Handover
- Fully reproducible Jupyter Notebook (`notebooks/AnomalyGuard_Pipeline.ipynb`).
- Complete Streamlit application (`app.py`).
- Analysis of false-positive risks and operational mitigation strategies.

---

## 🔬 Stretch Challenges

### 1. Local Outlier Factor (LOF) & Prediction Semantics
- **Semantics**: LOF computes local density deviation against $k$-nearest neighbors.
- **Findings**: On high-dimensional tabular transaction spaces (27 features after one-hot encoding), Euclidean distance suffers from the curse of dimensionality. LOF achieves an F1-score of only 0.2368, whereas Isolation Forest's global tree isolation achieves 0.9840.

### 2. Random Seed Stability Evaluation
- Tested across arbitrary initialization seeds (`[42, 101, 777, 2026]`).
- **Mean Spearman Rank Correlation ($\rho$)**: **0.9403**.
- Verifies that Isolation Forest produces consistent risk ranking regardless of random seed initialization.

---

## 🛡️ False-Positive Risk Analysis & Ethical AI Guardrails

### Why Benign Transactions Trigger Anomaly Signals
1. **High-Net-Worth Legitimate Purchases**: Affluent customers purchasing luxury jewelry or airline tickets fall in the upper 0.1% of amounts.
2. **Business Travel & Vacations**: International travel creates sudden geographic displacements.
3. **Hardware Upgrades**: Customers setting up a new smartphone or clearing browser cookies temporarily exhibit elevated device risk scores.

### Operational Mitigation Strategies
- **Tiered Escalation**: Route borderline anomalies to non-intrusive step-up MFA (mobile push challenge or biometric check) rather than immediate card declines.
- **Analyst Workload Throttling**: Calibrate contamination to match operational team review capacity.
- **Feedback Loops**: Confirmed benign false positives feed back into white-listing rules.

---

## 🚀 Running the Project Locally

### 1. Clone & Set Up Environment
```bash
git clone <your-repository-url>
cd "ai engineer internship week 3"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Jupyter Notebook
```bash
cd notebooks
jupyter notebook AnomalyGuard_Pipeline.ipynb
```

### 3. Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ How to Host on Streamlit Community Cloud (Free & Instant)

You can host this project live on Streamlit Cloud in under 3 minutes:

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Deploy AnomalyGuard to Streamlit Cloud"
   git branch -M main
   git remote add origin https://github.com/<your-username>/anomalyguard-week3.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **"Create app"** / **"New app"**.
   - Select your repository (`<your-username>/anomalyguard-week3`).
   - Set **Branch**: `main`.
   - Set **Main file path**: `app.py`.
   - Click **Deploy!**
3. Your live application will be instantly accessible at: `https://<your-app-name>.streamlit.app`.

---

## 📹 Mandatory 5-Minute Video Demonstration Script

When recording your narrated video submission, follow this structured breakdown:

1. **Introduction & Challenge (0:00 - 0:45)**:
   - Introduce *AnomalyGuard*: An unsupervised learning pipeline built to assist financial review teams in prioritizing unusual transactions.
   - Emphasize the core guideline: *An anomaly is a review signal, not proof of fraud.*
2. **Exploratory Data Analysis & Preprocessing (0:45 - 1:45)**:
   - Walk through the Streamlit Tab 1 (EDA distributions, amount skewness, device risk).
   - Show Tab 2: Explain feature engineering of Travel Velocity ($\Delta d / \Delta t$) and why `RobustScaler` was chosen to avoid outlier contamination.
3. **Baseline vs. Isolation Forest (1:45 - 3:00)**:
   - Demonstrate Tuesday's rule-based baseline (389 flags).
   - Explain how Wednesday's Isolation Forest isolates multivariate anomalies using random partitioning (250 flags, 0.984 F1-score).
4. **Threshold Sensitivity & Interpretability (3:00 - 4:15)**:
   - Walk through Tab 4: Show the percentile sensitivity curve and explain the trade-off between analyst fatigue (Precision) and fraud risk (Recall).
   - Show Tab 3: Inspect individual flagged transactions and explain the human-readable review drivers.
5. **Stretch Challenges & Lessons Learned (4:15 - 5:00)**:
   - Present the Local Outlier Factor comparison (curse of dimensionality in high dimensions) and random seed stability test (Spearman $\rho = 0.94$).
   - Discuss false-positive risks and conclude with responsible AI practices.

---

## 💼 LinkedIn Project Update Post Draft

```
🚀 Week 03 AI Engineering Milestone: AnomalyGuard — Transaction Anomaly Detection & Operational Review

Thrilled to share my Week 3 project at Stack & Scale! Financial risk teams face overwhelming transaction volumes where distinguishing true operational risks from benign spending requires sound unsupervised learning.

🔍 What I Built:
• Data Engineering: Engineered physical travel velocity (Δd/Δt) to detect impossible travel, risk interaction terms, and robust scaling without data leakage.
• Rule-Based Baseline: Established an interpretable statistical benchmark (IQR fences & Z-scores).
• Isolation Forest: Deployed an unsupervised Isolation Forest achieving an F1-Score of 0.984 (98.4% Recall & Precision) on benchmark anomalies.
• Threshold Sensitivity & Explainability: Quantified analyst review burden across contamination percentiles and generated automated review drivers for flagged records.
• Stretch Challenges: Evaluated Local Outlier Factor (LOF) prediction semantics and verified random seed ranking stability (Spearman ρ = 0.940).
• Interactive Streamlit App: Built a full-featured operational triage dashboard with CSV upload and single-transaction inspection.

⚠️ Core Takeaway: In financial AI engineering, an anomaly score is an operational triage signal, not proof of fraud!

#AIEngineering #MachineLearning #AnomalyDetection #IsolationForest #FinTech #Streamlit #DataScience #StackAndScale
```
