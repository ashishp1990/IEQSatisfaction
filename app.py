import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="IEQ Satisfaction Prediction",
    page_icon="🏫",
    layout="wide"
)

# --------------------------------------------------
# Load Artifacts
# --------------------------------------------------
artifact = joblib.load("ieq_models.joblib")

models = artifact["models"]
scaler = artifact["scaler"]
metrics_df = artifact["metrics"]

FEATURE_COLUMNS = list(scaler.feature_names_in_)
TARGET_COLUMN = "IEQSatisfaction"

# --------------------------------------------------
# Load Template & Test Samples
# --------------------------------------------------
TEMPLATE_PATH = Path("ieq_full_feature_template.csv")
TEST_SAMPLE_PATH = Path("ieq_test_samples.csv")

template_df = pd.read_csv(TEMPLATE_PATH)
template_row = template_df.iloc[0].to_dict()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🏫 IEQ Satisfaction Prediction</h1>
    <p style="text-align:center; font-size:16px;">
    Full-feature prediction using trained machine learning models
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys()),
    index=list(models.keys()).index("XGBoost")
)

model = models[model_name]

input_mode = st.sidebar.radio(
    "Input Mode",
    ["Manual Input (Form)", "CSV Upload"]
)

# --------------------------------------------------
# MANUAL INPUT MODE
# --------------------------------------------------
if input_mode == "Manual Input (Form)":
    st.header("✍️ Manual Feature Input")

    st.info(
        "All fields are pre-filled with realistic example values. "
        "You may modify any value before prediction."
    )

    input_data = {}

    with st.form("manual_input_form"):
        cols = st.columns(3)

        for i, col_name in enumerate(FEATURE_COLUMNS):
            with cols[i % 3]:
                default_value = float(template_row.get(col_name, 0.0))
                input_data[col_name] = st.number_input(
                    col_name,
                    value=default_value,
                    format="%.4f"
                )

        submitted = st.form_submit_button("🔮 Predict IEQ Satisfaction")

    if submitted:
        X = pd.DataFrame([input_data])
        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[0][1]
        label = "Satisfied" if prob >= 0.5 else "Not Satisfied"

        st.success(
            f"""
            **Predicted IEQSatisfaction:** {label}  
            **Probability:** {prob*100:.2f}%
            """
        )

# --------------------------------------------------
# CSV UPLOAD MODE
# --------------------------------------------------
else:
    st.header("📂 Upload CSV (Full Feature Set)")

    st.markdown(
        """
        **CSV Rules**
        - Must contain **all feature columns**
        - Must NOT contain `IEQSatisfaction`
        - Column order does not matter
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "⬇️ Download CSV Template (With Example Values)",
            template_df.to_csv(index=False),
            file_name="ieq_full_feature_template.csv"
        )

    with col2:
        if TEST_SAMPLE_PATH.exists():
            st.download_button(
                "⬇️ Download Test Samples (From Dataset)",
                open(TEST_SAMPLE_PATH, "rb"),
                file_name="ieq_test_samples.csv"
            )

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # Validation
        missing = set(FEATURE_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(FEATURE_COLUMNS)

        if missing:
            st.error(f"❌ Missing columns: {sorted(missing)}")
        elif extra:
            st.error(f"❌ Unexpected columns: {sorted(extra)}")
        else:
            X = df[FEATURE_COLUMNS]
            X_scaled = scaler.transform(X)

            probs = model.predict_proba(X_scaled)[:, 1]
            preds = np.where(probs >= 0.5, "Satisfied", "Not Satisfied")

            result_df = df.copy()
            result_df["Predicted IEQSatisfaction"] = preds
            result_df["Probability (%)"] = (probs * 100).round(2)

            st.subheader("📊 Prediction Results")
            st.dataframe(result_df, use_container_width=True)

# --------------------------------------------------
# Model Performance
# --------------------------------------------------
st.markdown("---")
st.header("📈 Model Performance (Test Dataset)")
st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("ML Assignment 2 | IEQ Satisfaction Prediction")
