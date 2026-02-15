import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="IEQ Satisfaction Prediction",
    page_icon="🏫",
    layout="wide"
)

# --------------------------------------------------
# Load Trained Artifacts
# --------------------------------------------------
artifact = joblib.load("ieq_models.joblib")

models = artifact["models"]
scaler = artifact["scaler"]
metrics_df = artifact["metrics"]

# Exact feature list used during training
FEATURE_COLUMNS = list(scaler.feature_names_in_)

TARGET_COLUMN = "IEQSatisfaction"

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🏫 IEQ Satisfaction Prediction</h1>
    <p style="text-align:center; font-size:16px;">
    Predict Indoor Environmental Quality satisfaction using full-feature input
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
    ["Manual Input (All Features)", "CSV Upload"]
)

# --------------------------------------------------
# Manual Input Mode
# --------------------------------------------------
if input_mode == "Manual Input (All Features)":
    st.header("✍️ Manual Feature Input")

    st.warning(
        "You must provide values for ALL features used during training. "
        "IEQSatisfaction is predicted and must NOT be entered."
    )

    input_data = {}

    with st.form("manual_input_form"):
        cols = st.columns(3)
        for i, col_name in enumerate(FEATURE_COLUMNS):
            with cols[i % 3]:
                input_data[col_name] = st.number_input(
                    col_name,
                    value=0.0,
                    format="%.3f"
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
# CSV Upload Mode
# --------------------------------------------------
else:
    st.header("📂 Upload CSV (All Features)")

    st.markdown(
        """
        **CSV Requirements**
        - Must contain **all feature columns**
        - Must NOT contain `IEQSatisfaction`
        - Column order does not matter
        """
    )

    # CSV template (FULL FEATURE SET)
    template_df = pd.DataFrame(columns=FEATURE_COLUMNS)

    st.download_button(
        "⬇️ Download CSV Template",
        template_df.to_csv(index=False),
        file_name="ieq_full_feature_template.csv"
    )

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # Validation
        missing = set(FEATURE_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(FEATURE_COLUMNS)

        if missing:
            st.error(f"Missing columns: {sorted(missing)}")
        elif extra:
            st.error(f"Unexpected columns: {sorted(extra)}")
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
