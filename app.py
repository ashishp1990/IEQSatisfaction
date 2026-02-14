import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --------------------------------------------------
# Page config
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

# --------------------------------------------------
# App title & description
# --------------------------------------------------
st.title("Indoor Environmental Quality (IEQ) Satisfaction Prediction")

st.markdown("""
This application predicts **Indoor Environmental Quality (IEQ) Satisfaction**
based on classroom environmental and contextual features.

**Target:**
- `1` → Satisfied  
- `0` → Not Satisfied
""")

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a model",
    list(models.keys()),
    index=list(models.keys()).index("Random Forest")
)

model = models[model_name]

# --------------------------------------------------
# Section 1: User Input Prediction
# --------------------------------------------------
st.header("🔢 Manual Input Prediction")

st.markdown("Enter feature values to predict IEQ Satisfaction.")

user_input = {}
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        user_input[feature] = st.number_input(
            feature,
            value=0.0
        )

input_df = pd.DataFrame([user_input])

if st.button("Predict Satisfaction"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ Satisfied (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Not Satisfied (Probability: {1 - probability:.2f})")

# --------------------------------------------------
# Section 2: CSV Upload Prediction
# --------------------------------------------------
st.header("📂 CSV Upload Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file (test data only)",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview:")
    st.dataframe(test_df.head())

    # Ensure correct feature order
    test_df = test_df[feature_names]

    test_scaled = scaler.transform(test_df)
    preds = model.predict(test_scaled)
    probs = model.predict_proba(test_scaled)[:, 1]

    output_df = test_df.copy()
    output_df["Prediction"] = preds
    output_df["Probability"] = probs

    st.subheader("Prediction Results")
    st.dataframe(output_df)

# --------------------------------------------------
# Section 3: Model Metrics Table
# --------------------------------------------------
st.header("📊 Model Performance Metrics")

st.dataframe(results_df.style.format("{:.3f}"))

# --------------------------------------------------
# Section 4: Bar Chart Visualization
# --------------------------------------------------
st.header("📈 Model Comparison (Bar Charts)")

metric_selected = st.selectbox(
    "Select metric",
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
# Section 5: Confusion Matrix
# --------------------------------------------------
st.header("🧩 Confusion Matrix")

# Use test set metrics from training (best practice)
# Assumes Random Forest as reference for confusion matrix
reference_model = models["Random Forest"]

y_true = artifact.get("y_test", None)
X_test_ref = artifact.get("X_test", None)

if y_true is not None and X_test_ref is not None:
    y_pred_ref = reference_model.predict(X_test_ref)

    cm = confusion_matrix(y_true, y_pred_ref)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)
else:
    st.info(
        "Confusion matrix uses training test data. "
        "To enable it, store X_test and y_test in the artifact."
    )
