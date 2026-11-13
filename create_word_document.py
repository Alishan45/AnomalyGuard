"""
Creates the Word Document (AnomalyGuard_Week03_Project_Report.docx)
adhering strictly to the submission rules.
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import os

doc = docx.Document()

# Adjust margins
for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

# Styling Helper
def set_heading(text, level, color=RGBColor(30, 58, 138)):
    p = doc.add_heading(level=level)
    run = p.add_run(text)
    run.font.color.rgb = color
    run.font.bold = True
    return p

# 1) Project Title, Purpose, and Short Project Overview
title_p = doc.add_paragraph()
title_run = title_p.add_run("AnomalyGuard — Transaction Anomaly Detection & Operational Review")
title_run.font.size = Pt(22)
title_run.font.bold = True
title_run.font.color.rgb = RGBColor(30, 58, 138)
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

subtitle_p = doc.add_paragraph()
sub_run = subtitle_p.add_run("Stack & Scale | Remote Internship 2026 — AI Engineering Week 03")
sub_run.font.size = Pt(13)
sub_run.font.italic = True
sub_run.font.color.rgb = RGBColor(100, 116, 139)
subtitle_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()

set_heading("1. Project Title, Purpose & Overview", level=1)

p_purpose = doc.add_paragraph()
p_purpose.add_run("Project Purpose: ").bold = True
p_purpose.add_run(
    "Financial and operational risk teams face massive daily transaction volumes. "
    "The purpose of AnomalyGuard is to build a reliable, reproducible, unsupervised-learning pipeline "
    "that highlights and prioritizes unusual payment transactions for operational triage without human bias. "
    "Crucially, the system operates on the foundational principle that an anomaly is an operational review signal, "
    "not deterministic proof of fraud."
)

p_overview = doc.add_paragraph()
p_overview.add_run("Project Overview: ").bold = True
p_overview.add_run(
    "AnomalyGuard implements a complete end-to-end unsupervised anomaly detection pipeline across tabular payment transactions. "
    "The project spans exploratory data analysis, domain-specific feature engineering (including physical travel velocity and log transformations), "
    "a statistical rule-based baseline benchmark, an unsupervised Isolation Forest model with calibrated [0.0, 1.0] scoring, "
    "threshold sensitivity analysis, Local Outlier Factor comparison, and multi-seed stability testing. "
    "The pipeline is exposed via a full-featured Streamlit interactive web dashboard for human analyst review."
)

# 2) Day 1–Day 5 Task/Progress Screenshots with Short Captions
set_heading("2. Day 1–Day 5 Task & Progress Screenshots", level=1)

# Day 1
set_heading("Day 1: Dataset Exploration & Feature Distributions", level=2, color=RGBColor(51, 65, 85))
p_d1 = doc.add_paragraph(
    "Inspected transaction schema across 10,000 records. Evaluated missing values and identified heavy right-skewed "
    "amounts and Beta-distributed device risk scores to establish the baseline review objective."
)
if os.path.exists('figures/day1_eda.png'):
    doc.add_picture('figures/day1_eda.png', width=Inches(6.2))
    cap1 = doc.add_paragraph()
    cap1_run = cap1.add_run("Figure 1: Day 1 EDA showing transaction amounts, device risk, distance from home, and merchant volumes.")
    cap1_run.font.size = Pt(9.5)
    cap1_run.font.italic = True
    cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Day 2
set_heading("Day 2: Preprocessing, Feature Engineering & Baseline Benchmark", level=2, color=RGBColor(51, 65, 85))
p_d2 = doc.add_paragraph(
    "Synthesized physical travel velocity (distance/time) to detect impossible travel (>800 km/h), applied log1p transformations, "
    "and scaled continuous signals with RobustScaler without label leakage. Built a statistical IQR/Z-score baseline flagging 389 records."
)
if os.path.exists('figures/day2_baseline.png'):
    doc.add_picture('figures/day2_baseline.png', width=Inches(6.2))
    cap2 = doc.add_paragraph()
    cap2_run = cap2.add_run("Figure 2: Day 2 Feature engineering (log travel velocity) and baseline heuristic rule triggers.")
    cap2_run.font.size = Pt(9.5)
    cap2_run.font.italic = True
    cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Day 3
set_heading("Day 3: Isolation Forest Model & Calibrated Scoring", level=2, color=RGBColor(51, 65, 85))
p_d3 = doc.add_paragraph(
    "Trained an unsupervised Isolation Forest (150 estimators, contamination=0.025, random_seed=42). "
    "Calibrated the decision function to normalized [0.0, 1.0] anomaly scores and ranked top suspicious transactions with human review notes."
)
if os.path.exists('figures/day3_isolation_forest.png'):
    doc.add_picture('figures/day3_isolation_forest.png', width=Inches(6.2))
    cap3 = doc.add_paragraph()
    cap3_run = cap3.add_run("Figure 3: Day 3 Isolation Forest score distribution and multivariate amount vs displacement anomaly scatter.")
    cap3_run.font.size = Pt(9.5)
    cap3_run.font.italic = True
    cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Day 4
set_heading("Day 4: Threshold Sensitivity & Evaluation Trade-offs", level=2, color=RGBColor(51, 65, 85))
p_d4 = doc.add_paragraph(
    "Conducted sensitivity analysis comparing percentiles (85.0% to 99.5%) to quantify analyst queue burden vs detection recall. "
    "Isolation Forest achieved an F1-Score of 0.9840, significantly outperforming the baseline heuristic."
)
if os.path.exists('figures/day4_evaluation.png'):
    doc.add_picture('figures/day4_evaluation.png', width=Inches(6.2))
    cap4 = doc.add_paragraph()
    cap4_run = cap4.add_run("Figure 4: Day 4 Precision-Recall sensitivity curve across percentiles and model confusion matrix.")
    cap4_run.font.size = Pt(9.5)
    cap4_run.font.italic = True
    cap4.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Day 5
set_heading("Day 5: Stretch Challenges (LOF & Seed Stability)", level=2, color=RGBColor(51, 65, 85))
p_d5 = doc.add_paragraph(
    "Benchmarked Local Outlier Factor (LOF) and demonstrated the curse of dimensionality in high-dimensional feature spaces. "
    "Verified Isolation Forest ranking stability across random seeds (mean Spearman rho = 0.9403)."
)
if os.path.exists('figures/day5_stretch_stability.png'):
    doc.add_picture('figures/day5_stretch_stability.png', width=Inches(6.2))
    cap5 = doc.add_paragraph()
    cap5_run = cap5.add_run("Figure 5: Day 5 Model benchmark comparison (Baseline vs IF vs LOF) and pairwise seed correlation matrix.")
    cap5_run.font.size = Pt(9.5)
    cap5_run.font.italic = True
    cap5.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 3) Screenshots of the Final Weekly Project Output/Result
set_heading("3. Final Project Output & Streamlit Review Dashboard", level=1)
p_final = doc.add_paragraph(
    "The final system is deployed as an interactive Streamlit web dashboard providing complete operational review workflows, "
    "interactive threshold tuning, single-transaction inspection, and exportable review queues."
)
if os.path.exists('figures/final_dashboard.png'):
    doc.add_picture('figures/final_dashboard.png', width=Inches(6.2))
    cap_f = doc.add_paragraph()
    cap_f_run = cap_f.add_run("Figure 6: Final AnomalyGuard operational review dashboard, queue allocation, and driver breakdown.")
    cap_f_run.font.size = Pt(9.5)
    cap_f_run.font.italic = True
    cap_f.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 4) Completed Features / Deliverables
set_heading("4. Completed Features & Deliverables", level=1)

features = [
    ("Exploratory Data Analysis Engine", "Automated distribution statistics, right-skewness metrics, missing value verification, and correlation matrix."),
    ("Feature Engineering Pipeline", "Engineered physical travel velocity (Δd/Δt) to detect impossible travel (>800 km/h), log-amount transforms, risk interaction terms, and RobustScaler normalization fitted strictly without data leakage."),
    ("Statistical Rule-Based Baseline", "Transparent operational benchmark evaluating univariate Z-scores (>3.0), extreme upper IQR fences, and multi-factor domain rules."),
    ("Unsupervised Isolation Forest Model", "Trained on 150 isolation trees with calibrated [0.0, 1.0] scoring, ranking atypical multivariate transactions via path length isolation."),
    ("Threshold Sensitivity & Trade-off Analysis", "Quantified analyst review queue volume across contamination percentiles (85.0% to 99.5%), balancing False Positive fatigue against Recall."),
    ("Model Explainability & Review Drivers", "Automated feature deviation notes comparing flagged records against population medians for transparent human review."),
    ("Stretch Challenge 1 (LOF Comparison)", "Evaluated Local Outlier Factor prediction semantics, contrasting local neighbor density against global tree partitioning in 27-dimensional feature spaces."),
    ("Stretch Challenge 2 (Seed Stability)", "Verified Isolation Forest ranking consistency across arbitrary initialization seeds (Spearman rank correlation ρ = 0.9403)."),
    ("Interactive Streamlit Application", "6-tab production web application featuring live CSV upload, threshold sliders, interactive Plotly charts, top-anomaly CSV export, and single-transaction inspection."),
    ("Executed Jupyter Notebook", "Clean, top-to-bottom pre-executed pipeline notebook containing all reproducible code, tables, and rendered charts.")
]

for title, desc in features:
    p_feat = doc.add_paragraph(style='List Bullet')
    p_feat.add_run(f"{title}: ").bold = True
    p_feat.add_run(desc)

# 5) Brief Explanation of Main Project Flow, Screens, and Functions
set_heading("5. Main Project Flow, Screens & Functions", level=1)

p_flow = doc.add_paragraph(
    "The AnomalyGuard operational architecture follows a structured 5-stage sequential workflow:\n\n"
    "1. Data Ingestion & Preprocessing: Raw transaction streams are cleaned and enriched with behavioral features (velocity, risk interactions) and scaled via RobustScaler without target leakage.\n"
    "2. Baseline Screening: Transactions pass through statistical fences and rule heuristics to establish a transparent baseline.\n"
    "3. Isolation Forest Scoring: The unsupervised ensemble partitions observations and assigns a calibrated anomaly score from 0.0 (normal) to 1.0 (highly anomalous).\n"
    "4. Threshold Calibration: Operational leaders select the contamination rate matching human review capacity (e.g., top 2.5% = 250 records).\n"
    "5. Analyst Triage & Deep Dive: High-scoring transactions are presented on the Streamlit dashboard with automated driver explanations (e.g., rapid displacement or sudden value burst), allowing analysts to download CSVs or inspect specific transaction IDs."
)

# 6) Weekly Learning Outcomes
set_heading("6. Weekly Learning Outcomes", level=1)

learnings = [
    ("Unsupervised Anomaly Modeling", "Mastered the mechanics of Isolation Forest and path length isolation for uncovering complex multivariate outliers without relying on ground-truth training labels."),
    ("Preventing Data Leakage in Unsupervised Systems", "Learned how to strictly isolate benchmark labels during preprocessing and scaling to prevent synthetic contamination of the feature space."),
    ("Curse of Dimensionality in Anomaly Detection", "Observed through hands-on benchmarking that distance-based methods like Local Outlier Factor (LOF) deteriorate in 20+ dimensional spaces, whereas Isolation Forest maintains robust separation."),
    ("Operational Threshold Trade-offs", "Gained deep understanding of the delicate balance between Precision (minimizing false positives and analyst alert fatigue) and Recall (maximizing fraud risk coverage)."),
    ("Model Governance & Reproducibility", "Developed techniques for seed stability evaluation using Spearman rank correlation, and built end-to-end interactive applications that make AI decisions interpretable to non-technical risk teams.")
]

for title, desc in learnings:
    p_learn = doc.add_paragraph(style='List Bullet')
    p_learn.add_run(f"{title}: ").bold = True
    p_learn.add_run(desc)

# 7) Additional Short Points (Ethical AI & Operational Guardrails)
set_heading("7. Ethical AI & Operational Guardrails", level=1)
p_ethic = doc.add_paragraph(
    "Operational Principle: An anomaly score is an analytical review signal, not proof of fraud. "
    "Benign transactions frequently trigger high anomaly scores due to legitimate life events, such as wealthy clients buying luxury goods, "
    "urgent overseas travel, or device upgrades. For this reason, AnomalyGuard is designed to augment human risk teams with "
    "interpretable review drivers and stepped-up authentication workflows, rather than initiating automated punitive declines."
)

output_filename = "AnomalyGuard_Week03_Project_Report.docx"
doc.save(output_filename)
print(f"Created {output_filename} successfully ({os.path.getsize(output_filename) / 1024:.1f} KB).")
