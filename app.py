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
# Load trained artifacts
# --------------------------------------------------
artifact = joblib.load("ieq_models.joblib")

models = artifact["models"]
scaler = artifact["scaler"]
feature_means = artifact["feature_means"]
results_df = artifact["metrics"]

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🏫 Indoor Environmental Quality (IEQ) Satisfaction Prediction")

st.markdown(
    """
This application predicts **IEQ Satisfaction**.

- The model outputs **Probability of Satisfaction**
- Prediction rule:
  - **Satisfied** if probability ≥ 50%
  - **Not Satisfied** otherwise
"""
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Application Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys()),
    index=list(models.keys()).index("Random Forest")
)
model = models[model_name]

input_mode = st.sidebar.radio(
    "Input Method",
    ["CSV Upload (Recommended)", "Manual Input"]
)

# --------------------------------------------------
# Feature builder (mean-based, stable)
# --------------------------------------------------
def build_features(students, temperature, season, windows, noise, lighting):
    X = pd.DataFrame(
        feature_means.values.reshape(1, -1),
        columns=feature_means.index
    )

    if "Student" in X.columns:
        X["Student"] = students

    for col in ["Temp_Back", "Temp_Middle", "Temp_Front", "Trm"]:
        if col in X.columns:
            X[col] = temperature

    rh_map = {"Summer": 60, "Winter": 40, "Rainy": 70, "Autumn": 55}
    for col in ["RH_Back", "RH_Middle", "RH_Front"]:
        if col in X.columns:
            X[col] = rh_map[season]

    co2 = np.clip(420 + students * 8 - windows * 40, 400, 1200)
    for col in ["CO2_Back", "CO2_Middle", "CO2_Front"]:
        if col in X.columns:
            X[col] = co2

    noise_map = {"Low": 40, "Medium": 55, "High": 70}
    for col in ["SoundLevel_Back", "SoundLevel_Middle", "SoundLevel_Front"]:
        if col in X.columns:
            X[col] = noise_map[noise]

    lux_map = {"Poor": 150, "Average": 300, "Good": 600}
    for col in ["LightLux_Back", "LightLux_Middle", "LightLux_Front"]:
        if col in X.columns:
            X[col] = lux_map[lighting]

    return X

# --------------------------------------------------
# CSV Upload Mode
# --------------------------------------------------
if input_mode == "CSV Upload (Recommended)":
    st.header("📂 CSV Upload Prediction")

    template_df = pd.DataFrame({
        "Students": [45],
        "Temperature": [26],
        "Season": ["Summer"],
        "Windows": [2],
        "Noise": ["Medium"],
        "Lighting": ["Good"]
    })

    st.download_button(
        "⬇️ Download CSV Template",
        template_df.to_csv(index=False),
        file_name="ieq_input_template.csv"
    )

    uploaded = st.file_uploader("Upload filled CSV file", type=["csv"])

    if uploaded:
        user_df = pd.read_csv(uploaded)
        output = []

        for _, row in user_df.iterrows():
            X = build_features(
                row["Students"],
                row["Temperature"],
                row["Season"],
                row["Windows"],
                row["Noise"],
                row["Lighting"]
            )

            X_scaled = scaler.transform(X)
            prob = model.predict_proba(X_scaled)[0][1]

            output.append({
                **row.to_dict(),
                "Prediction": "Satisfied" if prob >= 0.5 else "Not Satisfied",
                "Probability_of_Satisfaction (%)": round(prob * 100, 2)
            })

        st.subheader("📊 Prediction Results")
        st.dataframe(pd.DataFrame(output))

# --------------------------------------------------
# Manual Input Mode
# --------------------------------------------------
else:
    st.header("🔢 Manual Input Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        students = st.number_input("Number of Students", 10, 200, 40)
        temperature = st.slider("Average Temperature (°C)", 15, 60, 26)

    with c2:
        season = st.selectbox("Season", ["Summer", "Winter", "Rainy", "Autumn"])
        windows = st.slider("Windows Open", 0, 10, 2)

    with c3:
        noise = st.selectbox("Noise Level", ["Low", "Medium", "High"])
        lighting = st.selectbox("Lighting Quality", ["Poor", "Average", "Good"])

    if st.button("Predict IEQ Satisfaction"):
        X = build_features(
            students, temperature, season, windows, noise, lighting
        )

        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]

        if prob >= 0.5:
            st.success(
                f"✅ Satisfied\n\nProbability of Satisfaction: **{prob*100:.2f}%**"
            )
        else:
            st.error(
                f"❌ Not Satisfied\n\nProbability of Satisfaction: **{prob*100:.2f}%**"
            )

# --------------------------------------------------
# Metrics
# --------------------------------------------------
st.header("📊 Model Performance Metrics")
st.dataframe(results_df.style.format("{:.3f}"))

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("ML Assignment 2 | IEQ Satisfaction Prediction | Streamlit App")
