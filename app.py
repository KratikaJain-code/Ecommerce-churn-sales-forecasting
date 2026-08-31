import streamlit as st
import pandas as pd
import joblib


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)


# ============================================
# LOAD MODEL AND SCALER
# ============================================

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")


# ============================================
# APP TITLE
# ============================================

st.title("📊 Customer Churn Prediction App")

st.write(
    "Predict customer churn risk using RFM-based customer behavior."
)

st.divider()


# ============================================
# INPUT SECTION
# ============================================

st.subheader("Enter Customer Information")

recency = st.number_input(
    "Recency (days)",
    min_value=0,
    value=30
)

frequency = st.number_input(
    "Frequency (number of orders)",
    min_value=1,
    value=2
)

monetary = st.number_input(
    "Monetary (total spending)",
    min_value=0.0,
    value=500.0
)

average_order_value = st.number_input(
    "Average Order Value",
    min_value=0.0,
    value=250.0
)


# ============================================
# PREDICTION
# ============================================

if st.button("🔮 Predict Churn Risk"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary],
        "AverageOrderValue": [average_order_value]
    })

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Churn probability
    probability = model.predict_proba(input_scaled)[0][1]

    # ========================================
    # RISK CATEGORY
    # ========================================

    if probability < 0.30:
        risk_category = "Low Risk"
    elif probability < 0.60:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"


    # ========================================
    # DISPLAY RESULTS
    # ========================================

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    if risk_category == "High Risk":
        st.error(f"🔴 {risk_category}")

    elif risk_category == "Medium Risk":
        st.warning(f"🟡 {risk_category}")

    else:
        st.success(f"🟢 {risk_category}")


    # ========================================
    # INTERPRETATION
    # ========================================

    st.write("### Customer Analysis")

    if risk_category == "High Risk":

        st.write(
            "This customer has a high predicted probability of churn. "
            "Consider targeted retention campaigns, personalized offers, "
            "or re-engagement strategies."
        )

    elif risk_category == "Medium Risk":

        st.write(
            "This customer shows moderate churn risk. "
            "Consider proactive engagement and personalized communication."
        )

    else:

        st.write(
            "This customer currently has a relatively low churn risk. "
            "Continue maintaining engagement and customer satisfaction."
        )
