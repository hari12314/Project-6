import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("preprocessor_inference.joblib")
model = joblib.load("best_model_top15.joblib")

preprocessor = pipeline["preprocessor"]
scaler = pipeline["scaler"]
all_features = pipeline["all_features"]
num_cols = pipeline["num_cols"]
cat_cols = pipeline["cat_cols"]
top15_original_features = pipeline["top15_original_features"]
top15_ohe_cols = pipeline["top15_ohe_cols"]

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Prediction"])

# Page 1: Introduction
if page == "Introduction":
    st.title("Insurance Claim Prediction Project")
    st.markdown("""
    ## Overview
    This project predicts whether a customer will make an insurance claim or not.  
    It uses a **machine learning model (LightGBM)** trained on insurance policyholder data.  

    ## Dataset
    - Includes **numeric** (age, premium, vehicle age, etc.) and **categorical** (fuel type, region, etc.) features.  
    - Target variable: `is_claim` (Yes/No).  

    ## Approach
    1. Preprocessing with **OneHotEncoding** + **Scaling**.  
    2. Feature selection using **SelectKBest (Top 15 features)**.  
    3. Trained **LightGBM with class balancing**.  

    ## App Functionality
    - Page 1: Project introduction.  
    - Page 2: Input user details → Predict claim probability.  
    """)

# Page 2: Prediction
elif page == "Prediction":
    st.title("Insurance Claim Prediction")

    # User Input
    user_input = {}
    for feature in top15_original_features:
        if feature in num_cols:
            user_input[feature] = st.number_input(f"{feature}", value=0.0)
        elif feature in cat_cols:
            ohe = preprocessor.named_transformers_["cat"]
            idx = cat_cols.index(feature)
            categories = ohe.categories_[idx].tolist()
            user_input[feature] = st.selectbox(f"{feature}", options=categories)

    # Build full row for preprocessing
    row = {}
    for col in num_cols:
        row[col] = 0
    for i, col in enumerate(cat_cols):
        row[col] = preprocessor.named_transformers_["cat"].categories_[i][0]
    row.update(user_input)
    user_df = pd.DataFrame([row])

    # Prediction
    if st.button("Predict Claim"):
        user_encoded = preprocessor.transform(user_df)
        user_scaled = scaler.transform(user_encoded)
        user_top15 = pd.DataFrame(user_scaled, columns=all_features)[top15_ohe_cols]

        pred = model.predict(user_top15)[0]
        pred_prob = model.predict_proba(user_top15)[0][1]

        result_text = "Yes (Claim)" if pred == 1 else "No (No Claim)"
        st.subheader("Prediction Result")
        st.write(f"Claim: **{result_text}**")
        st.write(f"Prediction Probability: {pred_prob:.4f}")
