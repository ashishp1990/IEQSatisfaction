import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="IEQ Satisfaction Prediction",
    layout="wide"
)

# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------
model_artifact = joblib.load("ieq_models.joblib")

models = model_artifact["models"]
scaler = model_artifact["scaler"]
feature_means = model_artifact["feature_means"]
results_df = model_artifact["metrics"]

# 🔑 EXACT training feature order (CRITICAL)
TRAINING_FEATURES = list(scaler.feature_names_in_)

# --------------------------------------------------
# Load conditional / global feature means
# --------------------------------------------------
conditional_artifact = joblib.load("conditional_feature_means.joblib")

global_means = conditional_artifact["global_means"]

TARGET_COLUMN = "IEQSatisfaction"

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🏫 Indoor Environmental Quality (IEQ) Satisfaction Prediction")

st.markdown("""
This application predicts **IEQ Satisfaction** using trained machine learning models.

Users provide **high-level classroom inputs**.  
The application reconstructs the **full sensor-level feature space** using
**statistical averages derived from the training dataset**, ensuring consistency
between training and deployment.
""")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Application Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys()),
    index=list(models.keys()).index("XGBoost")
)

model = models[model_name]

# --------------------------------------------------
# Feature reconstruction logic (CORE FIX)
# --------------------------------------------------
def build_features_from_inputs(
    students,
    temperature,
    season,
    windows,
    noise,
    lighting
):
    # Start from global numeric means
    X = pd.DataFrame(
        global_means.values.reshape(1, -1),
        columns=global_means.index
    )

    # Student count
    if "Student" in X.columns:
        X["Student"] = students

    # Temperature
    for col in ["Temp_Back", "Temp_Middle", "Temp_Front", "Trm"]:
        if col in X.columns:
            X[col] = temperature

    # Relative humidity
    rh_map = {
        "Summer": 60,
        "Winter": 40,
        "Rainy": 70,
        "Autumn": 55
    }
    for col in ["RH_Back", "RH_Middle", "RH_Front"]:
        if col in X.columns:
            X[col] = rh_map.get(season, 55)

    # CO₂ (occupancy-driven)
    co2 = np.clip(students * 8.0, 400, 1500)
    for col in ["CO2_Back", "CO2_Middle", "CO2_Front"]:
        if col in X.columns:
            X[col] = co2

    # Noise
    noise_map = {"Low": 40, "Medium": 55, "High": 70}
    for col in ["SoundLevel_Back", "SoundLevel_Middle", "SoundLevel_Front"]:
        if col in X.columns:
            X[col] = noise_map[noise]

    # Lighting
    lux_map = {"Poor": 150, "Average": 300, "Good": 600}
    for col in ["LightLux_Back", "LightLux_Middle", "LightLux_Front"]:
        if col in X.columns:
            X[col] = lux_map[lighting]

    # 🔥 MOST IMPORTANT LINE – align with training schema
    X = X.reindex(columns=TRAINING_FEATURES)

    return X

# --------------------------------------------------
# CSV Upload Mode (Primary)
# --------------------------------------------------
st.header("📂 CSV Upload – Primary Prediction Mode")

template_df = pd.DataFrame({
    "Students": [45],
    "Temperature": [26],
    "Season": ["Summer"],
    "Windows": [2],
    "Noise": ["Medium"],
    "Lighting": ["Good"],
    "IEQSatisfaction": [4]  # optional
})

st.download_button(
    "⬇️ Download CSV Template",
    template_df.to_csv(index=False),
    file_name="ieq_input_template.csv"
)

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    user_df = pd.read_csv(uploaded)
    predictions = []

    for _, row in user_df.iterrows():
        X = build_features_from_inputs(
            row["Students"],
            row["Temperature"],
            row["Season"],
            row["Windows"],
            row["Noise"],
            row["Lighting"]
        )

        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]
        pred_target = 1 if prob >= 0.5 else 0

        result = {
            **row.to_dict(),
            "Model Used": model_name,
            "Predicted Target": pred_target,
            "Predicted Label": "Satisfied" if pred_target == 1 else "Not Satisfied",
            "Probability of Satisfaction (%)": round(prob * 100, 2)
        }

        # Show actual target if present
        if TARGET_COLUMN in row:
            actual_target = 1 if row[TARGET_COLUMN] >= 4 else 0
            result["Actual Target"] = actual_target
            result["Correct Prediction"] = (
                "Yes" if actual_target == pred_target else "No"
            )

        predictions.append(result)

    st.subheader("📊 Prediction Results")
    st.dataframe(pd.DataFrame(predictions))

# --------------------------------------------------
# Model Performance Metrics
# --------------------------------------------------
st.header("📈 Model Performance (Test Dataset)")
st.dataframe(results_df.style.format("{:.3f}"))

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("ML Assignment 2 | IEQ Satisfaction Prediction | Streamlit Application")
