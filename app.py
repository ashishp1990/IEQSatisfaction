import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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
This application predicts **Indoor Environmental Quality (IEQ) Satisfaction**
using multiple machine learning models trained on classroom environmental data.

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
**Input Note**
- Upload a CSV with all features  
- Or enter values manually  
- IEQSatisfaction is optional and used only for comparison
"""
)

# =========================================================
# Tabs
# =========================================================
tab_input, tab_metrics = st.tabs(["🔢 Input & Prediction", "📈 Model Performance"])

# =========================================================
# TAB 1 — Input & Prediction
# =========================================================
with tab_input:
    st.subheader("Input Data")

    upload_col, manual_col = st.columns(2)

    input_df = None

    # -------------------------------
    # CSV Upload
    # -------------------------------
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

    # -------------------------------
    # Manual Input
    # -------------------------------
    with manual_col:
        st.markdown("### ✍️ Manual Input")

        with st.form("manual_input_form"):
            manual_data = {}

            for col in FEATURE_COLUMNS:
                default_val = float(feature_means[col])
                manual_data[col] = st.number_input(
                    col,
                    value=default_val
                )

            submit_manual = st.form_submit_button("Predict")

        if submit_manual:
            input_df = pd.DataFrame([manual_data])

    # -------------------------------
    # Prediction Logic
    # -------------------------------
    if input_df is not None:
        st.subheader("Prediction Result")

        # Drop target column if user included it
        if "IEQSatisfaction" in input_df.columns:
            ground_truth = input_df["IEQSatisfaction"]
            input_df = input_df.drop(columns=["IEQSatisfaction"])
        else:
            ground_truth = None

        # Ensure correct feature order
        input_df = input_df.reindex(columns=FEATURE_COLUMNS)

        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame([feature_means]))

        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Prediction
        prob = selected_model.predict_proba(input_scaled)[0][1]
        label = "Satisfied" if prob >= 0.5 else "Not Satisfied"

        # Display
        result_col, prob_col = st.columns(2)

        with result_col:
            st.metric(
                "Predicted IEQSatisfaction",
                label
            )

        with prob_col:
            st.metric(
                "Prediction Probability",
                f"{prob * 100:.2f}%"
            )

        if ground_truth is not None:
            st.markdown("### Ground Truth Comparison")
            comparison_df = pd.DataFrame({
                "Actual IEQSatisfaction": ground_truth,
                "Predicted Label": [label] * len(input_df),
                "Predicted Probability (%)": [(prob * 100).round(2)]
            })
            st.dataframe(comparison_df)

# =========================================================
# TAB 2 — Model Performance
# =========================================================
with tab_metrics:
    st.subheader("📈 Model Performance (Test Dataset)")

    st.dataframe(metrics_df)

    st.markdown("### Accuracy Comparison")
    st.bar_chart(
        metrics_df.set_index("Model")["Accuracy"]
    )

    st.markdown("### AUC Comparison")
    st.bar_chart(
        metrics_df.set_index("Model")["AUC"]
    )

    st.markdown("### Precision / Recall / F1")
    st.bar_chart(
        metrics_df.set_index("Model")[["Precision", "Recall", "F1"]]
    )

    st.markdown("### MCC Comparison")
    st.bar_chart(
        metrics_df.set_index("Model")["MCC"]
    )

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption(
    "IEQ Satisfaction Prediction • ML Assignment • Streamlit Application"
)
