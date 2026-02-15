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
- You may **upload a CSV** or **enter values manually**
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

st.sidebar.markdown(
    """
**Input Notes**
- Upload a CSV with all features  
- Or enter values manually  
- IEQSatisfaction is optional and used only for comparison  
"""
)

# =========================================================
# Tabs
# =========================================================
tab_input, tab_metrics = st.tabs(
    ["🔢 Input & Prediction", "📈 Model Performance"]
)

# =========================================================
# TAB 1 — Input & Prediction
# =========================================================
with tab_input:

    # -----------------------------------------------------
    # CSV Downloads
    # -----------------------------------------------------
    st.subheader("📥 Download CSV Files")

    d1, d2, d3 = st.columns(3)

    with d1:
        with open("ieq_full_feature_template.csv", "rb") as f:
            st.download_button(
                "⬇️ CSV Template (Example Values)",
                f,
                "ieq_full_feature_template.csv",
                mime="text/csv"
            )

    with d2:
        with open("ieq_test_samples.csv", "rb") as f:
            st.download_button(
                "⬇️ Test Data Samples",
                f,
                "ieq_test_samples.csv",
                mime="text/csv"
            )

    with d3:
        with open("ieq_satisfied_test_samples.csv", "rb") as f:
            st.download_button(
                "⬇️ Satisfied Samples",
                f,
                "ieq_satisfied_test_samples.csv",
                mime="text/csv"
            )

    st.markdown("---")

    # -----------------------------------------------------
    # Input Section
    # -----------------------------------------------------
    st.subheader("Input Data")

    upload_col, manual_col = st.columns(2)
    input_df = None
    ground_truth = None

    # ---------------- CSV Upload ----------------
    with upload_col:
        st.markdown("### 📂 Upload CSV")

        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"]
        )

        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.success("CSV uploaded successfully")
            st.dataframe(input_df.head())

    # ---------------- Manual Input ----------------
    with manual_col:
        st.markdown("### ✍️ Manual Input")

        with st.form("manual_input_form"):
            manual_data = {}

            for col in FEATURE_COLUMNS:
                manual_data[col] = st.number_input(
                    col,
                    value=float(feature_means[col])
                )

            submit_manual = st.form_submit_button("Predict")

        if submit_manual:
            input_df = pd.DataFrame([manual_data])

    # -----------------------------------------------------
    # Prediction Logic
    # -----------------------------------------------------
    if input_df is not None:

        st.subheader("🔮 Prediction Result")

        # Extract ground truth if present
        if "IEQSatisfaction" in input_df.columns:
            ground_truth = input_df["IEQSatisfaction"]
            input_df = input_df.drop(columns=["IEQSatisfaction"])

        # Align features
        input_df = input_df.reindex(columns=FEATURE_COLUMNS)

        # Imputation (same as notebook)
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame([feature_means]))

        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Prediction
        probs = selected_model.predict_proba(input_scaled)[:, 1]
        avg_prob = probs.mean()
        label = "Satisfied" if avg_prob >= 0.5 else "Not Satisfied"

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Predicted IEQSatisfaction",
                label
            )

        with c2:
            st.metric(
                "Prediction Probability",
                f"{avg_prob * 100:.2f}%"
            )

        if ground_truth is not None:
            st.markdown("### 🧪 Ground Truth Comparison")

            compare_df = pd.DataFrame({
                "Actual IEQSatisfaction": ground_truth,
                "Predicted Label": label,
                "Predicted Probability (%)": (probs * 100).round(2)
            })

            st.dataframe(compare_df)

# =========================================================
# TAB 2 — Model Performance
# =========================================================
with tab_metrics:

    st.subheader("📈 Model Performance (Test Dataset)")
    st.dataframe(metrics_df)

    st.markdown("### Accuracy")
    st.bar_chart(metrics_df.set_index("Model")["Accuracy"])

    st.markdown("### AUC Score")
    st.bar_chart(metrics_df.set_index("Model")["AUC"])

    st.markdown("### Precision / Recall / F1")
    st.bar_chart(
        metrics_df.set_index("Model")[["Precision", "Recall", "F1"]]
    )

    st.markdown("### MCC Score")
    st.bar_chart(metrics_df.set_index("Model")["MCC"])

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption(
    "IEQ Satisfaction Prediction • ML Assignment • Streamlit Application"
)
