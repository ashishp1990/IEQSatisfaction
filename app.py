import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="IEQ Satisfaction Prediction",
    layout="wide"
)

# --------------------------------------------------
# Light custom CSS (SAFE for Streamlit Cloud)
# --------------------------------------------------
st.markdown(
    """
    <style>
        .main {
            max-width: 1200px;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .block-container {
            padding-top: 1.5rem;
        }
        .card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        .section-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .muted {
            color: #6c757d;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load artifacts
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
    "<p class='muted'>An interactive machine learning application for predicting classroom IEQ satisfaction.</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Application Settings")

model_name = st.sidebar.selectbox(
    "Prediction Model",
    list(models.keys()),
    index=list(models.keys()).index("Random Forest")
)
model = models[model_name]

input_mode = st.sidebar.radio(
    "Input Method",
    ["📂 CSV Upload (Recommended)", "🔢 Manual Input (Demo)"]
)

# --------------------------------------------------
# Feature builder
# --------------------------------------------------
def build_features_from_inputs(students, temperature, season, windows, noise, lighting):
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
# INPUT SECTION
# --------------------------------------------------
if input_mode == "📂 CSV Upload (Recommended)":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📂 CSV Upload Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='muted'>Download the template, fill classroom data, and upload for batch prediction.</p>",
        unsafe_allow_html=True
    )

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
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]

            results.append({
                **row.to_dict(),
                "Prediction": "Satisfied" if pred == 1 else "Not Satisfied",
                "Confidence": round(prob if pred == 1 else 1 - prob, 3)
            })

        st.markdown("<div class='section-title'>📊 Prediction Results</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(results))

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# MANUAL INPUT
# --------------------------------------------------
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔢 Manual Input (Demonstration)</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='muted'>This mode is for interactive demonstration. "
        "Internally, remaining features are auto-filled using training statistics.</p>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        students = st.number_input("Number of Students", 10, 200, 40)
        temperature = st.slider("Average Temperature (°C)", 15, 40, 26)

    with c2:
        season = st.selectbox("Season", ["Summer", "Winter", "Rainy", "Autumn"])
        windows = st.slider("Windows Open", 0, 10, 2)

    with c3:
        noise = st.selectbox("Noise Level", ["Low", "Medium", "High"])
        lighting = st.selectbox("Lighting Quality", ["Poor", "Average", "Good"])

    if st.button("🔍 Predict IEQ Satisfaction"):
        X = build_features_from_inputs(
            students, temperature, season, windows, noise, lighting
        )
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        if pred == 1:
            st.success(f"✅ **Satisfied** (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ **Not Satisfied** (Confidence: {1 - prob:.2f})")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📊 Model Performance Metrics</div>", unsafe_allow_html=True)
st.dataframe(results_df.style.format("{:.3f}"))
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("ML Assignment 2 | IEQ Satisfaction Prediction | Streamlit Application")
