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
# Load Model Artifacts
# --------------------------------------------------
artifact = joblib.load("ieq_models.joblib")

models = artifact["models"]
scaler = artifact["scaler"]
feature_means = artifact["feature_means"]
results_df = artifact["metrics"]

TRAINING_FEATURES = list(scaler.feature_names_in_)

# --------------------------------------------------
# Load Global Feature Means
# --------------------------------------------------
conditional_artifact = joblib.load("conditional_feature_means.joblib")
global_means = conditional_artifact["global_means"]

TARGET_COLUMN = "IEQSatisfaction"

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🏫 IEQ Satisfaction Prediction System</h1>
    <p style="text-align:center; font-size:16px;">
    Predict Indoor Environmental Quality satisfaction using machine learning
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Model Settings")

model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys()),
    index=list(models.keys()).index("XGBoost")
)

model = models[model_name]

input_mode = st.sidebar.radio(
    "Input Method",
    ["Manual Input", "CSV Upload"]
)

# --------------------------------------------------
# Feature Reconstruction
# --------------------------------------------------
def build_features_from_inputs(
    students, temperature, season, windows, noise, lighting
):
    X = pd.DataFrame(
        global_means.values.reshape(1, -1),
        columns=global_means.index
    )

    if "Student" in X.columns:
        X["Student"] = students

    for col in ["Temp_Back", "Temp_Middle", "Temp_Front", "Trm"]:
        if col in X.columns:
            X[col] = temperature

    rh_map = {"Summer": 60, "Winter": 40, "Rainy": 70, "Autumn": 55}
    for col in ["RH_Back", "RH_Middle", "RH_Front"]:
        if col in X.columns:
            X[col] = rh_map.get(season, 55)

    co2 = np.clip(students * 8.0, 400, 1500)
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

    X = X.reindex(columns=TRAINING_FEATURES)
    X = X.fillna(feature_means)

    return X

# --------------------------------------------------
# MANUAL INPUT MODE
# --------------------------------------------------
if input_mode == "Manual Input":
    st.header("✍️ Manual Classroom Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        students = st.number_input("Number of Students", 10, 80, 40)
        temperature = st.slider("Average Temperature (°C)", 18, 35, 26)

    with col2:
        season = st.selectbox("Season", ["Summer", "Winter", "Rainy", "Autumn"])
        windows = st.slider("Windows Open", 0, 8, 2)

    with col3:
        noise = st.selectbox("Noise Level", ["Low", "Medium", "High"])
        lighting = st.selectbox("Lighting Quality", ["Poor", "Average", "Good"])

    if st.button("🔮 Predict IEQ Satisfaction"):
        X = build_features_from_inputs(
            students, temperature, season, windows, noise, lighting
        )

        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]

        st.success(
            f"""
            **Predicted IEQSatisfaction:**  
            **{'Satisfied' if prob >= 0.5 else 'Not Satisfied'}**  

            **Probability:** {prob*100:.2f}%
            """
        )

# --------------------------------------------------
# CSV UPLOAD MODE
# --------------------------------------------------
else:
    st.header("📂 Upload Classroom Data (CSV)")

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

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        user_df = pd.read_csv(uploaded)
        results = []

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
            pred = "Satisfied" if prob >= 0.5 else "Not Satisfied"

            results.append({
                **row.to_dict(),
                "Predicted IEQSatisfaction": pred,
                "Probability (%)": round(prob * 100, 2)
            })

        st.subheader("📊 Prediction Results")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

# --------------------------------------------------
# Model Performance
# --------------------------------------------------
st.markdown("---")
st.header("📈 Model Performance (Test Dataset)")
st.dataframe(results_df.style.format("{:.3f}"), use_container_width=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("ML Assignment 2 | IEQ Satisfaction Prediction | Streamlit App")
