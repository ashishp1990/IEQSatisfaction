import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="IEQ Satisfaction Prediction",
    layout="wide"
)

# --------------------------------------------------
# Load trained artifacts
# --------------------------------------------------
artifact = joblib.load("ieq_models.joblib")

models = artifact["models"]
scaler = artifact["scaler"]
feature_names = artifact["feature_names"]
results_df = artifact["metrics"]
feature_means = artifact["feature_means"]  # <<< CRITICAL FIX

# --------------------------------------------------
# App Header
# --------------------------------------------------
st.title("🏫 Indoor Environmental Quality (IEQ) Satisfaction Prediction")

st.markdown("""
This application predicts **Indoor Environmental Quality (IEQ) Satisfaction**
based on classroom conditions.

✔ Human-friendly inputs  
✔ Internally mapped to full ML feature space  
✔ Trained ensemble models
""")

# --------------------------------------------------
# Sidebar – Model Selection
# --------------------------------------------------
st.sidebar.header("⚙️ Model Selection")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys()),
    index=list(models.keys()).index("Random Forest")
)

model = models[model_name]

# --------------------------------------------------
# User Inputs (CLEAN & LIMITED)
# --------------------------------------------------
st.header("📝 Classroom Information")

c1, c2, c3 = st.columns(3)

with c1:
    students = st.number_input("Number of Students", 10, 200, 40)
    temperature = st.slider("Average Room Temperature (°C)", 15, 40, 26)

with c2:
    season = st.selectbox("Season", ["Summer", "Winter", "Rainy", "Autumn"])
    windows = st.slider("Number of Windows Open", 0, 10, 2)

with c3:
    noise = st.selectbox("Noise Level", ["Low", "Medium", "High"])
    lighting = st.selectbox("Lighting Quality", ["Poor", "Average", "Good"])

# --------------------------------------------------
# Feature Builder (FIXED – MEAN INITIALIZATION)
# --------------------------------------------------
def build_feature_vector():
    # Start from TRAINING MEANS (CRITICAL)
    X = pd.DataFrame(
        feature_means.values.reshape(1, -1),
        columns=feature_means.index
    )

    # ---- Students ----
    if "Student" in X.columns:
        X["Student"] = students

    # ---- Temperature ----
    for col in ["Temp_Back", "Temp_Middle", "Temp_Front", "Trm"]:
        if col in X.columns:
            X[col] = temperature

    # ---- Season → Humidity ----
    rh_map = {
        "Summer": 60,
        "Winter": 40,
        "Rainy": 70,
        "Autumn": 55
    }
    for col in ["RH_Back", "RH_Middle", "RH_Front"]:
        if col in X.columns:
            X[col] = rh_map[season]

    # ---- CO2 estimation (based on students + ventilation) ----
    co2_base = 420 + (students * 8) - (windows * 40)
    co2_base = np.clip(co2_base, 400, 1200)

    for col in ["CO2_Back", "CO2_Middle", "CO2_Front"]:
        if col in X.columns:
            X[col] = co2_base

    # ---- Noise ----
    noise_map = {"Low": 40, "Medium": 55, "High": 70}
    for col in ["SoundLevel_Back", "SoundLevel_Middle", "SoundLevel_Front"]:
        if col in X.columns:
            X[col] = noise_map[noise]

    # ---- Lighting ----
    lux_map = {"Poor": 150, "Average": 300, "Good": 600}
    for col in ["LightLux_Back", "LightLux_Middle", "LightLux_Front"]:
        if col in X.columns:
            X[col] = lux_map[lighting]

    return X

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.header("🔍 Prediction Result")

if st.button("Predict IEQ Satisfaction"):
    input_df = build_feature_vector()
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.success(f"✅ **Satisfied** (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ **Not Satisfied** (Confidence: {1 - prob:.2f})")

# --------------------------------------------------
# Model Performance Table
# --------------------------------------------------
st.header("📊 Model Performance Metrics")
st.dataframe(results_df.style.format("{:.3f}"))

# --------------------------------------------------
# Bar Chart Visualization
# --------------------------------------------------
st.header("📈 Model Comparison")

metric_selected = st.selectbox(
    "Select Metric",
    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
)

fig, ax = plt.subplots(figsize=(8, 4))
results_df[metric_selected].plot(kind="bar", ax=ax)
ax.set_title(f"Model Comparison based on {metric_selected}")
ax.set_ylabel(metric_selected)
ax.set_xlabel("Models")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
st.pyplot(fig)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("ML Assignment-2 | IEQ Satisfaction Prediction | Streamlit Application")
