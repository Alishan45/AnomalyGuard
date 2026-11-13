"""
Generates the complete reproducible Jupyter Notebook for AnomalyGuard.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 🛡️ AnomalyGuard — Transaction Anomaly Detection & Analysis
**Stack & Scale | Remote Internship 2026 — AI Engineering Week 03**
**Author:** AI Engineering Intern
**Domain:** Financial Technology & Transaction Monitoring

---

### Real-World Challenge
Financial and operational risk teams face immense volumes of daily payment transactions. To prevent alert fatigue and maintain audit integrity, organizations require a reproducible unsupervised learning pipeline to rank and highlight unusual records.

> **CRITICAL OPERATIONAL PRINCIPLE**:
> An anomaly is an analytical **review signal**, not proof of fraud. Unsupervised anomaly detection surfaces statistical rarities for operational triage or stepped-up authentication; it never establishes criminal or malicious intent on its own.
"""))

# Monday - Day 1
cells.append(nbf.v4.new_markdown_cell("""## 📅 Monday: Dataset Exploration and Problem Definition
**Objectives:**
- Load transaction dataset with clear provenance and schema.
- Inspect missing values, data types, and statistical distributions.
- Quantify right-skewed transaction amounts and define review objective.
"""))

cells.append(nbf.v4.new_code_cell("""import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src modules can be imported
sys.path.append(os.path.abspath('..'))

from src.eda import compute_dataset_summary, get_numeric_distribution_stats, compute_correlation_matrix
from src.preprocessing import TransactionPreprocessor
from src.baseline import RuleBasedStatisticalBaseline
from src.models import IsolationForestDetector, LocalOutlierFactorDetector, evaluate_seed_stability
from src.evaluation import evaluate_threshold_sensitivity, compute_comprehensive_metrics, explain_top_anomalies

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

print("Environment loaded successfully.")"""))

cells.append(nbf.v4.new_code_cell("""# Load transaction dataset
data_path = '../data/transactions.csv'
df = pd.read_csv(data_path)
print(f"Dataset Shape: {df.shape[0]} transactions, {df.shape[1]} features.")
df.head()"""))

cells.append(nbf.v4.new_code_cell("""# Dataset Summary and Missing Value Inspection
summary = compute_dataset_summary(df)
print(f"Total Rows: {summary['n_rows']}")
print(f"Total Missing Values: {summary['total_missing_values']}")
print(f"Numerical Columns: {summary['numeric_columns']}")
print(f"Categorical Columns: {summary['categorical_columns']}")

# Distribution Statistics
dist_stats = get_numeric_distribution_stats(df)
dist_stats"""))

cells.append(nbf.v4.new_code_cell("""# Visualize Distribution of Key Signals
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

sns.histplot(df['amount'], bins=50, kde=True, ax=axes[0], color='#2563EB')
axes[0].set_title('Transaction Amount ($) - Skewed')
axes[0].set_xlabel('Amount ($)')

sns.histplot(df['distance_from_home_km'], bins=50, kde=True, ax=axes[1], color='#10B981')
axes[1].set_title('Distance from Home (km)')
axes[1].set_xlabel('Distance (km)')

sns.histplot(df['device_risk_score'], bins=50, kde=True, ax=axes[2], color='#F59E0B')
axes[2].set_title('Device Risk Score [0.0 - 1.0]')
axes[2].set_xlabel('Device Risk')

plt.tight_layout()
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/eda_distributions.png', dpi=200)
plt.show()"""))

# Tuesday - Day 2
cells.append(nbf.v4.new_markdown_cell(r"""## 📅 Tuesday: Preprocessing, Feature Engineering & Baseline Benchmark
**Objectives:**
- Handle missing values and categorical features without evaluation leakage.
- Synthesize domain behavioral signals:
  - **Travel Velocity ($\Delta d / \Delta t$)**: Highlights physically impossible geographic travel (>800 km/h).
  - **Log-amount transform**: Mitigates heavy right skewness.
  - **Risk interaction term**: Multiplies device risk by authentication failure count.
- Scale numeric features using `RobustScaler` (median & IQR based to avoid outlier distortion).
- Construct a transparent **Statistical and Rule-Based Baseline** as an operational benchmark.
"""))

cells.append(nbf.v4.new_code_cell("""# Initialize and fit Preprocessor
preprocessor = TransactionPreprocessor(use_log_transforms=True, scale_features=True)
X_scaled, feat_df = preprocessor.fit_transform(df)

print(f"Original Feature Count: {df.shape[1]}")
print(f"Processed Feature Matrix Shape: {X_scaled.shape}")
print("Engineered Feature Names:", preprocessor.feature_names_[:8], "...")"""))

cells.append(nbf.v4.new_code_cell("""# Define and Execute Tuesday's Rule-Based Baseline Benchmark
baseline = RuleBasedStatisticalBaseline(z_score_threshold=3.0, velocity_cutoff_kmh=800.0)
baseline_scores, baseline_flags, rule_triggers = baseline.predict_scores_and_flags(df)

print(f"Baseline Rules Flagged: {baseline_flags.sum()} records ({(baseline_flags.sum()/len(df))*100:.2f}%)")
print("\\nRule Trigger Counts:")
print(rule_triggers.sum())"""))

# Wednesday - Day 3
cells.append(nbf.v4.new_markdown_cell(r"""## 📅 Wednesday: Isolation Forest Anomaly Detection
**Objectives:**
- Train an unsupervised `IsolationForest` with documented hyper-parameters and fixed random seed.
- Generate decision function and normalize to calibrated anomaly scores $\in [0, 1]$.
- Rank most anomalous transactions and inspect feature deviations.
"""))

cells.append(nbf.v4.new_code_cell("""# Configure and Train Isolation Forest
CONTAMINATION_RATE = 0.025  # Expected anomaly frequency (2.5%)
RANDOM_SEED = 42

iso_forest = IsolationForestDetector(
    n_estimators=150,
    contamination=CONTAMINATION_RATE,
    max_samples='auto',
    random_state=RANDOM_SEED
)

# Fit strictly on feature matrix (NO ground-truth label input!)
iso_forest.fit(X_scaled)

iso_scores = iso_forest.predict_anomaly_scores(X_scaled)
iso_flags = iso_forest.predict_flags(X_scaled)

print(f"Isolation Forest Flagged: {iso_flags.sum()} transactions for review.")"""))

cells.append(nbf.v4.new_code_cell("""# Score Distribution Plot
plt.figure(figsize=(9, 4.5))
sns.histplot(iso_scores, bins=60, kde=True, color='#4338CA')
threshold_val = np.percentile(iso_scores, (1 - CONTAMINATION_RATE) * 100)
plt.axvline(threshold_val, color='red', linestyle='--', label=f'Contamination Cutoff ({threshold_val:.3f})')
plt.title('Calibrated Isolation Forest Anomaly Score Distribution')
plt.xlabel('Anomaly Score [0.0 = Typical, 1.0 = Highly Anomalous]')
plt.ylabel('Transaction Count')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/isolation_forest_scores.png', dpi=200)
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Inspect Top 10 Anomalies with Human-Interpretable Review Drivers
top_anomalies = explain_top_anomalies(df, iso_scores, iso_flags, top_k=10)
top_anomalies[['Rank', 'Anomaly_Score', 'Transaction_ID', 'Amount', 'Category', 'Key_Review_Drivers']]"""))

# Thursday - Day 4
cells.append(nbf.v4.new_markdown_cell("""## 📅 Thursday: Threshold Sensitivity & Evaluation
**Objectives:**
- Evaluate threshold sensitivity: analyze flag counts across contamination percentiles.
- Quantify operational trade-offs: Precision vs Recall, review capacity vs detection coverage.
- If trustworthy evaluation labels exist, calculate F1, PR-AUC, and ROC-AUC.
"""))

cells.append(nbf.v4.new_code_cell("""# Threshold Sensitivity Analysis
y_eval = df['is_synthetic_anomaly'].values if 'is_synthetic_anomaly' in df.columns else None

sensitivity_table = evaluate_threshold_sensitivity(
    anomaly_scores=iso_scores,
    ground_truth=y_eval,
    percentile_thresholds=[85.0, 90.0, 95.0, 97.0, 97.5, 98.0, 99.0, 99.5]
)
sensitivity_table"""))

cells.append(nbf.v4.new_code_cell("""# Comparative Model Metrics
iso_metrics = compute_comprehensive_metrics(iso_scores, iso_flags, y_eval, "Isolation Forest")
baseline_metrics = compute_comprehensive_metrics(baseline_scores, baseline_flags, y_eval, "Rule-Based Baseline")

benchmark_df = pd.DataFrame([baseline_metrics, iso_metrics])
benchmark_df"""))

cells.append(nbf.v4.new_code_cell("""# Review Queue vs Threshold Visualization
plt.figure(figsize=(9, 4.5))
plt.plot(sensitivity_table['Percentile_Cutoff'], sensitivity_table['Flagged_Count'], marker='o', color='#DC2626', linewidth=2)
plt.title('Review Queue Volume vs Percentile Cutoff Strictness')
plt.xlabel('Percentile Threshold')
plt.ylabel('Number of Flagged Transactions')
plt.grid(True)
plt.tight_layout()
plt.savefig('../figures/threshold_sensitivity.png', dpi=200)
plt.show()"""))

# Stretch Challenges
cells.append(nbf.v4.new_markdown_cell("""## 🔬 Optional Stretch Challenges
1. **Local Outlier Factor (LOF) Comparison & Prediction Semantics**:
   - Compares local neighbor density vs global recursive partition.
2. **Random Seed Stability**:
   - Assesses rank stability across arbitrary initialization seeds using Spearman rank correlation.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Local Outlier Factor (LOF)
lof = LocalOutlierFactorDetector(n_neighbors=25, contamination=CONTAMINATION_RATE)
lof.fit(X_scaled)
lof_scores = lof.predict_anomaly_scores(X_scaled)
lof_flags = lof.predict_flags(X_scaled)

lof_metrics = compute_comprehensive_metrics(lof_scores, lof_flags, y_eval, "Local Outlier Factor (LOF)")
stretch_benchmark = pd.DataFrame([baseline_metrics, iso_metrics, lof_metrics])
stretch_benchmark"""))

cells.append(nbf.v4.new_code_cell("""# 2. Random Seed Stability Analysis
stability_result = evaluate_seed_stability(X_scaled[:2000], seeds=[42, 101, 777, 2026], n_estimators=100)
print(f"Mean Spearman Rank Correlation across seeds: {stability_result['mean_spearman_rho']:.4f}")
print(f"Highly Stable (>0.90): {stability_result['is_highly_stable']}")
stability_result['pairwise_correlations']"""))

# Friday - Day 5
cells.append(nbf.v4.new_markdown_cell("""## 📅 Friday: Practical Findings, False-Positive Risk & Limitations
### Why False Positives Occur in Unsupervised Anomaly Detection
1. **Legitimate High-Value Purchases**: High-net-worth customers purchasing luxury goods, electronics, or booking travel fall in the extreme right tail of amounts.
2. **Vacation & Commuting**: Fast legitimate transit or connecting flights cause geographic displacement spikes.
3. **New Device Onboarding**: Switching phones or using private browsing elevates device risk scores without fraud.

### Operational Recommendations
- **Tiered Triage**: Route top 1% to automated 2FA challenges (SMS/App prompt) rather than card declines.
- **Explainability**: Always surface key feature deviation notes to review analysts to prevent audit fatigue.
- **Model Audits**: Continuously re-calibrate contamination rates based on staffing capacity.
"""))

nb['cells'] = cells

with open('notebooks/AnomalyGuard_Pipeline.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created notebooks/AnomalyGuard_Pipeline.ipynb successfully.")
