import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer

# =========================================================
# App Configuration
# =========================================================
st.set_page_config(
    page_title="IEQ Satisfaction Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 IEQ Satisfaction Prediction System")

st.markdown(
    """
- Upload a CSV / Excel file **OR** enter values manually  
- **IEQSatisfaction is predicted**, not required as input  
"""
)

# =========================================================
# Load Trained Artifacts
# =========================================================
@st.cache_resource
def load_artifacts():
    artifact = joblib.load("ieq_models.joblib")
    return (
        artifact["models"],
        artifact["scaler"],
        artifact["feature_means"],
        artifact["metrics"]
    )

models, scaler, feature_means, metrics_df = load_artifacts()
FEATURE_COLUMNS = list(feature_means.index)

# =========================================================
# Sidebar — Model Selection
# =========================================================
st.sidebar.header("⚙️ Model Settings")

model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

selected_model = models[model_name]

# =========================================================
# Tabs
# =========================================================
tab_input, tab_metrics = st.tabs(
    ["🔢 Input & Prediction", "📈 Model Performance"]
)

# =========================================================
# TAB 1 — INPUT & PREDICTION
# =========================================================
with tab_input:

    # -----------------------------------------------------
    # Downloads
    # -----------------------------------------------------
    st.subheader("📥 Download Input Files")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "⬇️ CSV Template",
            open("ieq_full_feature_template.csv", "rb"),
            file_name="ieq_full_feature_template.csv"
        )

    with c2:
        st.download_button(
            "⬇️ Test Data",
            open("ieq_test_samples.csv", "rb"),
            file_name="ieq_test_samples.csv"
        )

    with c3:
        st.download_button(
            "⬇️ Satisfied Samples",
            open("ieq_satisfied_test_samples.csv", "rb"),
            file_name="ieq_satisfied_test_samples.csv"
        )

    st.markdown("---")

    # -----------------------------------------------------
    # Sub-tabs
    # -----------------------------------------------------
    csv_tab, manual_tab = st.tabs(["📂 CSV Upload", "✍️ Manual Input"])

    input_df = None
    ground_truth = None

    # ================= CSV UPLOAD =================
    with csv_tab:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx"]
        )

        if uploaded_file is not None:
            input_df = (
                pd.read_csv(uploaded_file)
                if uploaded_file.name.endswith(".csv")
                else pd.read_excel(uploaded_file)
            )
            st.success("File uploaded successfully")
            st.dataframe(input_df.head())

    # ================= MANUAL INPUT =================
    with manual_tab:
        with st.form("manual_form"):
            manual_data = {
                col: st.number_input(
                    col,
                    value=float(feature_means[col])
                )
                for col in FEATURE_COLUMNS
            }
            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([manual_data])

    # =================================================
    # Prediction Logic (FIXED)
    # =================================================
    if input_df is not None:

        if "IEQSatisfaction" in input_df.columns:
            ground_truth = input_df["IEQSatisfaction"]
            input_df = input_df.drop(columns=["IEQSatisfaction"])

        input_df = input_df.reindex(columns=FEATURE_COLUMNS)

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame([feature_means]))

        X = scaler.transform(imputer.transform(input_df))

        probs = selected_model.predict_proba(X)[:, 1]
        labels = np.where(probs >= 0.5, "Satisfied", "Not Satisfied")

        st.markdown("---")
        st.subheader("🔮 Prediction Result")

        # ========== SINGLE ROW ==========
        if len(input_df) == 1:
            st.metric(
                "Predicted IEQSatisfaction",
                labels[0]
            )
            st.metric(
                "Prediction Probability",
                f"{probs[0] * 100:.2f}%"
            )

        # ========== MULTIPLE ROWS ==========
        else:
            results_df = input_df.copy()
            results_df["Predicted Label"] = labels
            results_df["Probability (%)"] = (probs * 100).round(2)

            st.dataframe(results_df)

            st.markdown("### 📊 Summary")
            st.write({
                "Average Probability (%)": round(probs.mean() * 100, 2),
                "Minimum Probability (%)": round(probs.min() * 100, 2),
                "Maximum Probability (%)": round(probs.max() * 100, 2),
            })

        if ground_truth is not None:
            st.markdown("### 🧪 Ground Truth Comparison")
            st.write(pd.DataFrame({
                "Actual IEQSatisfaction": ground_truth,
                "Predicted Label": labels,
                "Probability (%)": (probs * 100).round(2)
            }))

# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================
with tab_metrics:
    st.dataframe(metrics_df)

    st.bar_chart(metrics_df.set_index("Model")["Accuracy"])
    st.bar_chart(metrics_df.set_index("Model")["AUC"])
    st.bar_chart(metrics_df.set_index("Model")[["Precision", "Recall", "F1"]])
    st.bar_chart(metrics_df.set_index("Model")["MCC"])
