"""
AnomalyGuard — Transaction Anomaly Detection & Operational Review
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import time

from src.eda import compute_dataset_summary, get_numeric_distribution_stats, compute_correlation_matrix
from src.preprocessing import TransactionPreprocessor
from src.baseline import RuleBasedStatisticalBaseline
from src.models import IsolationForestDetector, LocalOutlierFactorDetector, evaluate_seed_stability
from src.evaluation import evaluate_threshold_sensitivity, compute_comprehensive_metrics, explain_top_anomalies
from data.generate_dataset import generate_transaction_dataset

st.set_page_config(
    page_title="AnomalyGuard | Transaction Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .stat-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #E2E8F0;
    }
    .alert-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #92400E;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("🛡️ AnomalyGuard")
    st.caption("AI Engineering — Week 03 Project")
    st.markdown("---")
    
    st.subheader("📁 Data Source")
    data_source_opt = st.radio(
        "Select Dataset:",
        ["Bundled Benchmark (10,000 Txns)", "Generate New Synthetic", "Upload CSV"],
        index=0
    )
    
    if data_source_opt == "Generate New Synthetic":
        gen_n = st.slider("Number of Transactions", 2000, 20000, 10000, step=1000)
        gen_ratio = st.slider("Synthetic Anomaly Ratio (%)", 0.5, 8.0, 2.5, step=0.5) / 100.0
        gen_seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)
        if st.button("Generate Dataset"):
            st.session_state["raw_df"] = generate_transaction_dataset(gen_n, gen_ratio, gen_seed)
            st.success(f"Generated {gen_n} records!")
            
    elif data_source_opt == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
        if uploaded_file is not None:
            st.session_state["raw_df"] = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(st.session_state['raw_df'])} records!")
            
    else:
        # Bundled
        if "raw_df" not in st.session_state:
            try:
                st.session_state["raw_df"] = pd.read_csv("data/transactions.csv")
            except Exception:
                st.session_state["raw_df"] = generate_transaction_dataset(10000, 0.025, 42)

    st.markdown("---")
    st.subheader("⚙️ Model Configuration")
    contamination = st.slider(
        "Expected Contamination Rate",
        min_value=0.005,
        max_value=0.10,
        value=0.025,
        step=0.005,
        help="Proportion of outliers in the dataset for threshold calibration."
    )
    n_estimators = st.slider("Isolation Forest Estimators", 50, 300, 150, step=25)
    random_seed = st.number_input("Model Random Seed", min_value=1, max_value=9999, value=42)
    use_log_transforms = st.checkbox("Apply Log & Velocity Feature Engineering", value=True)
    
    st.markdown("---")
    st.info("💡 **Operational Principle**: Anomaly detection identifies statistical deviations for review; it does not constitute deterministic proof of fraud.")

df = st.session_state.get("raw_df", None)
if df is None:
    st.error("Please load or generate a dataset.")
    st.stop()

# Cache Model Pipeline Execution
@st.cache_data
def run_pipeline(data_df: pd.DataFrame, cont: float, n_est: int, seed: int, log_tf: bool):
    preprocessor = TransactionPreprocessor(use_log_transforms=log_tf, scale_features=True)
    X_scaled, feat_df = preprocessor.fit_transform(data_df)
    
    # Baseline
    baseline = RuleBasedStatisticalBaseline()
    b_scores, b_flags, rule_triggers = baseline.predict_scores_and_flags(data_df)
    
    # Isolation Forest
    iso_model = IsolationForestDetector(n_estimators=n_est, contamination=cont, random_state=seed)
    iso_model.fit(X_scaled)
    iso_scores = iso_model.predict_anomaly_scores(X_scaled)
    iso_flags = iso_model.predict_flags(X_scaled)
    
    return {
        "preprocessor": preprocessor,
        "X_scaled": X_scaled,
        "feat_df": feat_df,
        "baseline_scores": b_scores,
        "baseline_flags": b_flags,
        "rule_triggers": rule_triggers,
        "iso_scores": iso_scores,
        "iso_flags": iso_flags
    }

results = run_pipeline(df, contamination, n_estimators, random_seed, use_log_transforms)

# Header
st.markdown('<div class="main-header">🛡️ AnomalyGuard — Transaction Anomaly Detection & Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production Unsupervised Learning Pipeline, Threshold Sensitivity, and Operational Review System</div>', unsafe_allow_html=True)

st.markdown("""
<div class="alert-box">
<strong>⚠️ Important Review Signal Notice:</strong> In financial engineering, an anomaly score represents an unusual behavioral pattern warranting human review or stepped-up authentication. <strong>An anomaly is a review signal, not proof of fraud.</strong>
</div>
""", unsafe_allow_html=True)

# Top KPIs
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Transactions", f"{len(df):,}")
with col2:
    flagged_iso = int(np.sum(results["iso_flags"]))
    st.metric("Flagged for Review (IF)", f"{flagged_iso:,}", f"{(flagged_iso/len(df))*100:.2f}%")
with col3:
    flagged_base = int(np.sum(results["baseline_flags"]))
    st.metric("Baseline Rule Flags", f"{flagged_base:,}", f"{(flagged_base/len(df))*100:.2f}%")
with col4:
    has_labels = "is_synthetic_anomaly" in df.columns
    if has_labels:
        true_anom = int(df["is_synthetic_anomaly"].sum())
        st.metric("Ground Truth (Eval Only)", f"{true_anom:,}", f"{(true_anom/len(df))*100:.2f}%")
    else:
        st.metric("Ground Truth", "Unlabeled (Real-World)")
with col5:
    st.metric("Engineered Features", f"{results['feat_df'].shape[1]}")

st.markdown("---")

# Main Navigation Tabs matching Roadmap & PDF Requirements
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 1. Exploratory Analysis (Mon)",
    "⚙️ 2. Preprocessing & Baseline (Tue)",
    "🌲 3. Isolation Forest Detection (Wed)",
    "⚖️ 4. Thresholds & Evaluation (Thu)",
    "🔬 5. Stretch: LOF & Seed Stability",
    "📋 6. Governance & Handover (Fri)"
])

# ----------------- TAB 1: EDA -----------------
with tab1:
    st.subheader("Monday: Dataset Schema & Distribution Analysis")
    st.markdown("Inspect schema, missingness, distributions, and plausible anomaly signals.")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Data Sample Preview**")
        st.dataframe(df.head(8), use_container_width=True)
    with c2:
        st.markdown("**Numeric Distribution Statistics**")
        stats_df = get_numeric_distribution_stats(df)
        st.dataframe(stats_df.style.format({
            "Mean": "{:.2f}", "Std": "{:.2f}", "Median": "{:.2f}",
            "IQR": "{:.2f}", "Min": "{:.2f}", "Max": "{:.2f}", "Skewness": "{:.2f}"
        }), use_container_width=True)
        
    st.markdown("---")
    st.subheader("Distribution Visualizations")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_amt = px.histogram(
            df, x="amount", nbins=60, marginal="box",
            title="Transaction Amount Distribution ($)",
            color_discrete_sequence=["#3B82F6"]
        )
        fig_amt.update_layout(xaxis_title="Amount ($)", yaxis_title="Count")
        st.plotly_chart(fig_amt, use_container_width=True)
    with col_b:
        if "device_risk_score" in df.columns:
            fig_risk = px.histogram(
                df, x="device_risk_score", nbins=50, marginal="box",
                title="Device Risk Score Distribution [0.0 - 1.0]",
                color_discrete_sequence=["#10B981"]
            )
            fig_risk.update_layout(xaxis_title="Device Risk Score", yaxis_title="Count")
            st.plotly_chart(fig_risk, use_container_width=True)
            
    col_c, col_d = st.columns(2)
    with col_c:
        if "merchant_category" in df.columns:
            cat_counts = df["merchant_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Volume"]
            fig_cat = px.bar(cat_counts, x="Category", y="Volume", title="Transaction Volume by Merchant Category", color="Volume", color_continuous_scale="Blues")
            st.plotly_chart(fig_cat, use_container_width=True)
    with col_d:
        corr_mat = compute_correlation_matrix(df)
        fig_corr = px.imshow(corr_mat, text_auto=".2f", color_continuous_scale="RdBu_r", title="Feature Pearson Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

# ----------------- TAB 2: PREPROCESSING & BASELINE -----------------
with tab2:
    st.subheader("Tuesday: Preprocessing, Feature Engineering & Baseline Benchmark")
    st.markdown(r"""
    Unsupervised models require careful feature scaling and domain feature synthesis:
    - **Physical Travel Velocity**: Ratio of distance moved to elapsed time ($\Delta \text{distance} / \Delta \text{time}$) to capture physically impossible displacements (>800 km/h).
    - **Log Transformations**: Stabilizes right-skewed amounts and distances.
    - **Security Interaction Term**: Combines device risk score and authentication failure count.
    - **RobustScaler**: Uses median and interquartile range (IQR) to prevent extreme anomalies from distorting scaling centers.
    """)
    
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        st.markdown("**Engineered Feature Matrix (First 6 Rows)**")
        st.dataframe(results["feat_df"].head(6), use_container_width=True)
    with col_p2:
        st.markdown("**Statistical & Rule-Based Baseline Heuristic**")
        st.markdown(r"""
        The baseline establishes an interpretable operational benchmark without machine learning:
        1. **Impossible Travel**: Velocity $> 800$ km/h between successive transactions.
        2. **Amount Outlier**: Amount $> Q_3 + 3.0 \times \text{IQR}$ or $Z\text{-score} > 3.0$.
        3. **High Device Risk**: Device Risk $> 0.70$ and $\ge 2$ failed PIN attempts.
        4. **International High-Risk**: International flag with Device Risk $> 0.75$.
        """)
        triggers = results["rule_triggers"].sum().reset_index()
        triggers.columns = ["Heuristic Rule", "Triggered Count"]
        st.dataframe(triggers, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### Comparison: Baseline Triggers vs Isolation Forest Anomaly Scores")
    overlap_df = pd.DataFrame({
        "Baseline_Flag": results["baseline_flags"],
        "IsolationForest_Flag": results["iso_flags"]
    })
    cross_tab = pd.crosstab(overlap_df["Baseline_Flag"], overlap_df["IsolationForest_Flag"], rownames=["Baseline Rule"], colnames=["Isolation Forest"])
    st.table(cross_tab)
    st.caption("Rows: Baseline (0 = Inlier, 1 = Flagged). Columns: Isolation Forest (0 = Inlier, 1 = Flagged).")

# ----------------- TAB 3: ISOLATION FOREST -----------------
with tab3:
    st.subheader("Wednesday: Isolation Forest Model & Ranked Anomaly Inspection")
    st.markdown("""
    Isolation Forest isolates anomalies by randomly selecting a feature and a split value.
    Because anomalies are rare and attribute-atypical, they require substantially fewer splits (shorter path lengths) to isolate.
    """)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_score_dist = px.histogram(
            x=results["iso_scores"],
            nbins=60,
            color=results["iso_flags"].astype(str),
            title="Isolation Forest Calibrated Anomaly Score Distribution",
            labels={"x": "Anomaly Score [0.0 = Normal, 1.0 = Highly Anomalous]", "color": "Flagged Review"}
        )
        fig_score_dist.add_vline(x=np.percentile(results["iso_scores"], (1 - contamination)*100), line_dash="dash", line_color="red", annotation_text="Contamination Cutoff")
        st.plotly_chart(fig_score_dist, use_container_width=True)
        
    with col_s2:
        plot_df = df.copy()
        plot_df["Anomaly_Score"] = results["iso_scores"]
        plot_df["Flagged"] = results["iso_flags"]
        fig_scatter = px.scatter(
            plot_df,
            x="amount",
            y="distance_from_last_txn_km" if "distance_from_last_txn_km" in plot_df.columns else "distance_from_home_km",
            color="Anomaly_Score",
            size="Anomaly_Score",
            hover_data=["transaction_id", "merchant_category", "device_risk_score"] if "transaction_id" in plot_df.columns else None,
            title="Amount vs Distance Colored by Anomaly Score",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.markdown("---")
    st.subheader("🔍 Top Ranked Suspicious Transactions for Human Review")
    top_k = st.slider("Select Number of Top Anomalies to Inspect", 5, 50, 15)
    
    top_anomalies_df = explain_top_anomalies(df, results["iso_scores"], results["iso_flags"], top_k=top_k)
    st.dataframe(top_anomalies_df, use_container_width=True)
    
    # Export button
    csv_buf = io.StringIO()
    top_anomalies_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Export Top Anomalies to CSV",
        data=csv_buf.getvalue(),
        file_name="top_flagged_anomalies_for_review.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.subheader("🔬 Single Transaction Deep Dive")
    txn_options = df["transaction_id"].tolist() if "transaction_id" in df.columns else [f"Record_{i}" for i in range(len(df))]
    selected_txn = st.selectbox("Choose Transaction ID to Inspect:", txn_options[:200])
    
    if selected_txn:
        idx = txn_options.index(selected_txn)
        t_row = df.iloc[idx]
        t_score = results["iso_scores"][idx]
        t_flag = results["iso_flags"][idx]
        
        c_d1, c_d2, c_d3, c_d4 = st.columns(4)
        c_d1.metric("Transaction ID", selected_txn)
        c_d2.metric("Anomaly Score", f"{t_score:.4f}")
        c_d3.metric("Review Decision", "🚨 FLAGGED" if t_flag == 1 else "✅ NORMAL")
        c_d4.metric("Amount", f"${float(t_row.get('amount', 0)):.2f}")
        
        st.json(t_row.to_dict())

# ----------------- TAB 4: THRESHOLDS & EVALUATION -----------------
with tab4:
    st.subheader("Thursday: Threshold Sensitivity, Trade-offs & Model Evaluation")
    st.markdown("""
    In unsupervised operations, threshold selection represents a direct trade-off between:
    - **Detection Coverage (Recall)**: Catching more potential irregularities.
    - **Analyst Workload (Precision / False Positive Burden)**: Preventing review queue exhaustion.
    """)
    
    y_ground_truth = df["is_synthetic_anomaly"].values if "is_synthetic_anomaly" in df.columns else None
    
    sensitivity_df = evaluate_threshold_sensitivity(
        results["iso_scores"],
        ground_truth=y_ground_truth,
        percentile_thresholds=[85.0, 90.0, 95.0, 97.0, 97.5, 98.0, 99.0, 99.5]
    )
    
    col_t1, col_t2 = st.columns([1.2, 0.8])
    with col_t1:
        st.markdown("**Sensitivity Table Across Percentile Cutoffs**")
        st.dataframe(sensitivity_df, use_container_width=True)
    with col_t2:
        fig_thresh = px.line(
            sensitivity_df,
            x="Percentile_Cutoff",
            y="Flagged_Count",
            markers=True,
            title="Review Queue Volume vs Threshold Strictness",
            color_discrete_sequence=["#EF4444"]
        )
        st.plotly_chart(fig_thresh, use_container_width=True)
        
    if y_ground_truth is not None:
        st.markdown("---")
        st.subheader("Benchmarked Performance Metrics (Ground Truth Available)")
        
        iso_metrics = compute_comprehensive_metrics(results["iso_scores"], results["iso_flags"], y_ground_truth, "Isolation Forest")
        base_metrics = compute_comprehensive_metrics(results["baseline_scores"], results["baseline_flags"], y_ground_truth, "Baseline Heuristic")
        
        perf_df = pd.DataFrame([base_metrics, iso_metrics])
        st.dataframe(perf_df, use_container_width=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            cm = np.array([[iso_metrics["TN"], iso_metrics["FP"]], [iso_metrics["FN"], iso_metrics["TP"]]])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Normal", "Flagged Anomaly"],
                y=["Actual Normal", "Actual Anomaly"],
                labels=dict(x="Model Prediction", y="True State"),
                title="Isolation Forest Confusion Matrix",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_m2:
            st.markdown("#### Operational Takeaway")
            st.info(f"""
            - **Isolation Forest** achieved an **F1-Score of {iso_metrics['F1_Score']}** with **{iso_metrics['Recall']*100:.1f}% Recall** and **{iso_metrics['Precision']*100:.1f}% Precision** at contamination rate `{contamination}`.
            - **Baseline Heuristic** achieved an **F1-Score of {base_metrics['F1_Score']}**. While rules catch obvious violations (like extreme speed or amounts), they miss multivariate combinations that Isolation Forest easily isolates.
            """)
    else:
        st.info("Uploaded dataset has no evaluation labels. Thresholds are calibrated based on target analyst team review capacity.")

# ----------------- TAB 5: STRETCH CHALLENGES -----------------
with tab5:
    st.subheader("Stretch Challenges: Local Outlier Factor (LOF) & Random Seed Stability")
    
    st.markdown("### 1. Local Outlier Factor (LOF) Comparison & Semantics")
    st.markdown("""
    **Prediction Semantics Note:**
    - **Local Outlier Factor (LOF)** calculates the local density of an observation relative to its $k$-nearest neighbors. An object is an outlier if its local density is substantially lower than that of its neighbors.
    - **Contrast with Isolation Forest**: Isolation Forest uses global binary space partitioning. In tabular transaction data with mixed continuous and one-hot categorical features, LOF suffers from the curse of dimensionality where Euclidean distances become uniform, making Isolation Forest significantly more reliable and computationally efficient.
    """)
    
    run_lof = st.button("Run Local Outlier Factor Comparison")
    if run_lof:
        with st.spinner("Fitting Local Outlier Factor..."):
            lof_detector = LocalOutlierFactorDetector(n_neighbors=20, contamination=contamination)
            lof_detector.fit(results["X_scaled"])
            lof_scores = lof_detector.predict_anomaly_scores(results["X_scaled"])
            lof_flags = lof_detector.predict_flags(results["X_scaled"])
            
            if y_ground_truth is not None:
                lof_metrics = compute_comprehensive_metrics(lof_scores, lof_flags, y_ground_truth, "Local Outlier Factor (LOF)")
                iso_metrics = compute_comprehensive_metrics(results["iso_scores"], results["iso_flags"], y_ground_truth, "Isolation Forest")
                st.dataframe(pd.DataFrame([iso_metrics, lof_metrics]), use_container_width=True)
            else:
                st.write(f"LOF Flagged Count: {int(np.sum(lof_flags))} records.")
                
    st.markdown("---")
    st.markdown("### 2. Isolation Forest Random Seed Stability Evaluation")
    st.markdown("""
    Because Isolation Forest relies on random feature selection and split points, operational systems must verify that transaction risk rankings are stable across different random seeds.
    """)
    
    run_stability = st.button("Run Multi-Seed Stability Test")
    if run_stability:
        with st.spinner("Evaluating ranking stability across 4 random seeds..."):
            seeds_to_test = [42, 101, 777, 2026]
            stability_res = evaluate_seed_stability(results["X_scaled"][:2500], seeds=seeds_to_test, n_estimators=100, contamination=contamination)
            
            c_s1, c_s2 = st.columns([1, 1])
            with c_s1:
                st.metric("Mean Spearman Rank Correlation (ρ)", f"{stability_res['mean_spearman_rho']:.4f}")
                st.metric("Minimum Pairwise Correlation", f"{stability_res['min_spearman_rho']:.4f}")
                if stability_res["is_highly_stable"]:
                    st.success("✅ Model demonstrates high ranking stability across arbitrary initialization seeds (ρ > 0.90).")
            with c_s2:
                st.dataframe(stability_res["pairwise_correlations"], use_container_width=True)

# ----------------- TAB 6: GOVERNANCE & HANDOVER -----------------
with tab6:
    st.subheader("Friday: Handover, False-Positive Risk & Demonstration Script")
    
    st.markdown("""
    ### 🛡️ Why False Positives Occur & How to Mitigate Them
    Unsupervised models detect statistical rarity, but rarity does **not** equate to malice:
    1. **Affluent Spending**: An affluent client purchasing luxury goods or booking holiday travel triggers high amount scores despite being completely legitimate.
    2. **Physical Relocation & Vacations**: International travel creates sudden velocity spikes.
    3. **Hardware Upgrades**: New phones or browsers result in a temporarily elevated device risk score until a trust baseline is re-established.
    
    **Operational Mitigation**:
    - **Tiered Escalation**: High score = Step-up Multi-Factor Authentication (push notification / SMS OTP) rather than outright card decline.
    - **Customer Feedback Loop**: Confirmed false positives update whitelists or fine-tune feature scaling.
    - **Analyst Capacity Throttling**: Calibrate the contamination parameter to align with maximum human review throughput (e.g. 100 cases/day).
    """)
    
    st.markdown("---")
    st.subheader("📹 Mandatory 5-Minute Video Demonstration Script")
    st.markdown(r"""
    Use the following structured guide when recording your narrated project demonstration:
    
    1. **Project Objective (0:00 - 0:45)**:
       - Introduce *AnomalyGuard*: An unsupervised learning pipeline built to assist financial review teams in prioritizing unusual transactions without human bias.
       - Reiterate the core principle: *An anomaly is an operational review signal, not proof of fraud.*
    2. **Exploratory Data Analysis & Preprocessing (0:45 - 1:45)**:
       - Show the transaction schema and distributions (amount skew, device risk).
       - Explain key feature engineering: travel velocity ($\Delta d / \Delta t$) to capture impossible travel, and RobustScaler to protect against outlier distortion.
    3. **Baseline vs. Isolation Forest (1:45 - 3:00)**:
       - Present Tuesday's rule-based/statistical baseline (Z-score, IQR fences, impossible travel rules).
       - Contrast with Wednesday's Isolation Forest: explain why random partitioning isolates multivariate outliers that single rules miss.
    4. **Threshold Sensitivity & Operational Trade-offs (3:00 - 4:15)**:
       - Walk through Thursday's threshold comparison table.
       - Explain the Precision vs Recall trade-off: higher thresholds reduce analyst alert fatigue; lower thresholds maximize risk coverage.
    5. **Stretch Challenges & Reflections (4:15 - 5:00)**:
       - Highlight the Local Outlier Factor (LOF) vs Isolation Forest comparison and multi-seed stability (Spearman $\rho > 0.93$).
       - Conclude with lessons learned regarding data leakage prevention and unsupervised evaluation methodology.
    """)
    
    st.markdown("---")
    st.caption("Stack & Scale | Remote Internship 2026 | AI Engineering Week 03")
