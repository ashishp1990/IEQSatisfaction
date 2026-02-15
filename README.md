# IEQ Satisfaction Prediction System

This project is a **Machine Learning–based web application** developed to predict **Indoor Environmental Quality (IEQ) Satisfaction** using classroom environmental, acoustic, visual, thermal, and demographic features.

The application is built using **Streamlit** and multiple supervised ML models trained and evaluated in a Jupyter Notebook. The deployment strictly follows the **same preprocessing and inference pipeline** used during training to ensure consistent results.

---

## Project Overview

- **Target Variable**: `IEQSatisfaction`
- **Problem Type**: Binary Classification  
  - `Satisfied`
  - `Not Satisfied`
- **Models Used**:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Random Forest
  - XGBoost

---

## Application Features

### Input & Prediction
- Upload **CSV or Excel** files containing feature values  
- Or provide **manual input** via an interactive form  
- `IEQSatisfaction` is **not required as input** (predicted by the model)  
- Handles **missing values** using mean imputation  
- Displays:
  - Predicted IEQ Satisfaction
  - Prediction probability
- Supports **single-row and multi-row CSV predictions**

### Downloadable Files
The app provides ready-to-use input files:
- **CSV Template** (`ieq_full_feature_template.csv`)
- **Test Data Samples** (`ieq_test_samples.csv`)
- **Satisfied Samples** (`ieq_satisfied_test_samples.csv`)

These ensure correct formatting and reproducible predictions.

### Model Performance
- Displays evaluation metrics on the test dataset:
  - Accuracy
  - AUC
  - Precision
  - Recall
  - F1 Score
  - Matthews Correlation Coefficient (MCC)
- Visual comparison using bar charts for all models

---

## Machine Learning Pipeline

The application uses the **same pipeline as the notebook**:

