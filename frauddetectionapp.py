import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import os
from datetime import datetime
from io import BytesIO

st.set_page_config(layout="wide")

hide_anchor_css = """
<style>
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
}
</style>
"""
st.markdown(hide_anchor_css, unsafe_allow_html=True)

@st.cache_resource
def load_model_any(path_candidates=None):
    if path_candidates is None:
        path_candidates = [
            "fraud_model.pkl", "xgb_fraud_model.pkl",
            "fraud_detection_xgb.pkl", "model.pkl"
        ]
    for p in path_candidates:
        if os.path.exists(p):
            try:
                m = joblib.load(p)
                return m, p
            except Exception:
                pass
    for ext in ("*.pkl", "*.joblib"):
        for f in glob.glob(ext):
            try:
                m = joblib.load(f)
                return m, f
            except Exception:
                continue
    return None, None


def assemble_expected_input(time_seconds, v_values, amount):
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    row = [time_seconds] + list(v_values) + [amount]
    return pd.DataFrame([row], columns=cols)


def compute_rule_score(amount, hour_of_day, txns_last_minute=0,
                       night_hours=(0, 6)):
    if amount <= 50000:
        amount_score = 0.0    # Normal
    elif amount <= 100000:
        amount_score = 0.5    # Suspicious
    else:
        amount_score = 1.0    # High-Risk

    start_n, end_n = night_hours
    night_score = 1.0 if start_n <= hour_of_day < end_n else 0.0

    if txns_last_minute <= 1:
        vel_score = 0.0
    elif txns_last_minute <= 3:
        vel_score = 0.5
    else:
        vel_score = 1.0

    w_amount, w_night, w_vel = 0.6, 0.25, 0.15
    raw = w_amount * amount_score + w_night * night_score + w_vel * vel_score
    return max(0.0, min(1.0, raw))


def blend_scores(model_prob, rule_score, alpha):
    return alpha * model_prob + (1 - alpha) * rule_score


def get_shap_feature_contributions(model, input_df, top_n=8):
    try:
        import shap
        explainer = shap.Explainer(model, input_df, silent=True)
        shap_vals = explainer(input_df)
        arr = shap_vals.values[0] if hasattr(shap_vals, "values") else np.array(shap_vals)
        feat_names = input_df.columns.tolist()
        df = pd.DataFrame({"feature": feat_names, "shap_value": arr})
        df["abs_shap"] = df["shap_value"].abs()
        df = df.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap")
        return df
    except Exception:
        return None


def df_to_bytes_csv(df: pd.DataFrame):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.read()

st.title("Fraud Detection Via XGBOOST & Rule Engine")
st.markdown("""
This application is a **Hybrid Fraud Detection System** that combines the power of an XGBoost machine learning model with a rule-based engine.  
The system evaluates transactions using both model predictions and configurable rules such as transaction amount thresholds and time-of-day checks.  
You can adjust the **model weight** to control how much influence the ML model has compared to the rules, and set a **detection threshold** to decide when a transaction should be flagged as fraud.  
The app supports three modes of operation: checking a single transaction, analyzing bulk data from a CSV file, and running synthetic simulations for testing scenarios.  
""")


model, model_path = load_model_any()
if model is None:
    st.error("No model found in working directory. Place a joblib/pkl model file here.")
    st.stop()

expected_features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
st.markdown("### Configuration:")
col1, col2 = st.columns(2)
with col1:
    alpha = st.slider("Model weight (alpha):", 0.0, 1.0, 0.75, 0.01,
                      help="Higher alpha → more weight to ML model, lower → more weight to rule engine")
with col2:
    threshold = st.slider("Detection threshold:", 0.0, 1.0, 0.5, 0.01)

# Tabs
tab_single, tab_bulk, tab_sim = st.tabs(["Single Transaction", "Upload CSV (Batch)", "Simulation"])

with tab_single:
    st.subheader("Single Transaction Check:")
    c1, c2 = st.columns([1, 1])

    with c1:
        # Default None, taake reset issue na ho
        txn_date = st.date_input("Transaction date")
        txn_time = st.time_input("Transaction time")
        amount = st.number_input("Transaction amount", 0.0, 1e9, 100.0, 1.0)
        txns_last_minute = st.number_input("Transactions in last 1 minute", 0, 100, 0)

    with c2:
        avg_spend = st.number_input("User average spend (optional)", 0.0, 1e9, 200.0, 1.0)
        usual_start = st.slider("User usual start hour", 0, 23, 8)
        usual_end = st.slider("User usual end hour", 0, 23, 20)

    if st.button("Run Hybrid Check"):
        if txn_date and txn_time:  # Ensure both are selected
            txn_dt = pd.to_datetime(f"{txn_date} {txn_time}")
            reference = pd.to_datetime("2025-01-01 00:00:00")
            time_seconds = (txn_dt - reference).total_seconds()

            base_v = np.random.normal(0, 1, 28)
            if amount > avg_spend * 5:
                base_v = base_v + np.random.normal(1.0, 0.5, 28)

            input_df = assemble_expected_input(time_seconds, base_v, amount)

            try:
                model_prob = float(model.predict_proba(input_df)[0][1])
            except Exception:
                model_prob = float(model.predict(input_df)[0])

            rule_score = compute_rule_score(amount, txn_dt.hour, txns_last_minute, (0, 6))
            combined = blend_scores(model_prob, rule_score, alpha)

            st.write(pd.DataFrame({
                "metric": ["Model probability", "Rule score", "Combined score"],
                "value": [round(model_prob, 4), round(rule_score, 4), round(combined, 4)]
            }))

            classification = "FRAUD" if combined >= threshold else "LEGIT"
            if classification == "FRAUD":
                st.error(f"Classification: FRAUD (score {combined:.3f})")
            else:
                st.success(f"Classification: LEGIT (score {combined:.3f})")

            shap_df = get_shap_feature_contributions(model, input_df, top_n=8)
            if shap_df is not None:
                st.table(shap_df.rename(columns={"feature": "Feature", "shap_value": "SHAP value"}))
            else:
                st.info("SHAP not available. Showing input features instead.")
                st.table(input_df.T.rename(columns={0: "value"}))
        else:
            st.warning("⚠ Please select both Transaction Date and Time before running the check.")

with tab_bulk:
    st.subheader("Bulk Transaction Analysis (CSV Upload):")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="bulk")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"✅ File loaded successfully! Total rows: {len(df)}")

        mode = st.radio("Select Processing Mode:", ["Quick Mode (1000 rows)", "Full Mode (All rows)"])

        if mode == "Quick Mode (1000 rows)":
            df = df.head(1000)
            st.info("Processing only first 1000 rows for faster results.")
        else:
            st.warning("⚠ Full dataset selected. This may take time...")

        progress = st.progress(0)
        status_text = st.empty()

        batch_size = 500
        predictions = []
        total_batches = len(df) // batch_size + 1

        for i in range(total_batches):
            start, end = i * batch_size, min((i + 1) * batch_size, len(df))
            if start < end:
                preds = model.predict(df.drop(columns=["Class"], errors="ignore").iloc[start:end])
                predictions.extend(preds)

            progress.progress(int(((i + 1) / total_batches) * 100))
            status_text.text(f"Processing batch {i+1}/{total_batches}...")

        df["Prediction"] = predictions
        fraud_count = (df["Prediction"] == 1).sum()
        legit_count = (df["Prediction"] == 0).sum()
        st.success(f"✅ Complete! Legit: {legit_count}, Fraud: {fraud_count}")
        st.dataframe(df.head(20))

with tab_sim:
    st.subheader("Simulation (synthetic transactions):")
    sim_cols = st.columns(3)
    n_sim = sim_cols[0].number_input("Number of transactions", 10, 20000, 1000, 10)
    n_users = sim_cols[1].number_input("Number of distinct users", 1, 1000, 50, 1)
    anomaly_rate = sim_cols[2].slider("Anomaly rate", 0.0, 0.5, 0.05, 0.01)

    if st.button("Generate and Run Simulation"):
        rng = np.random.default_rng(42)
        users = [f"user_{i}" for i in range(int(n_users))]
        rows = []
        for i in range(int(n_sim)):
            user = rng.choice(users)
            user_avg = rng.normal(200, 50)
            t_seconds = rng.integers(0, 86400 * 30)
            if rng.random() < anomaly_rate:
                amount = float(rng.uniform(100001, 500000))  # High-risk
            else:
                amount = float(abs(rng.normal(user_avg, user_avg * 0.5)))
            v = rng.normal(0, 1, 28)
            if amount > 100000:
                v = v + rng.normal(1.5, 0.8, 28)
            rows.append({
                "user": user,
                "Time": float(t_seconds),
                **{f"V{i}": float(v[i - 1]) for i in range(1, 29)},
                "Amount": amount
            })
        sim_df = pd.DataFrame(rows)
        X = sim_df[expected_features].copy()
        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception:
            probs = model.predict(X)

        rule_scores = X.apply(
            lambda r: compute_rule_score(
                r["Amount"],
                int((pd.to_datetime("2025-01-01") + pd.to_timedelta(r["Time"], unit="s")).hour),
                txns_last_minute=0,
                night_hours=(0, 6)
            ),
            axis=1
        )
        combined_scores = [blend_scores(float(p), float(r), alpha) for p, r in zip(probs, rule_scores)]
        sim_df["model_prob"] = probs
        sim_df["rule_score"] = rule_scores
        sim_df["combined_score"] = combined_scores
        sim_df["prediction"] = (sim_df["combined_score"] >= threshold).astype(int)

        total = len(sim_df)
        frauds = int(sim_df["prediction"].sum())
        fraud_pct = (frauds / total) * 100
        st.success(f"Simulation done. Total: {total} | Fraud detected: {frauds} | Fraud %: {fraud_pct:.4f}%")

        st.subheader("Top suspicious transactions")
        st.dataframe(sim_df.sort_values("combined_score", ascending=False).head(20))

        csv_bytes = df_to_bytes_csv(sim_df)
        st.download_button("Download simulation results", csv_bytes,
                           "simulation_predictions.csv", "text/csv")
st.markdown("<hr><center><small>© 2025 Fraud Detection System | Developed by Eman Fatima</small></center>", unsafe_allow_html=True)
