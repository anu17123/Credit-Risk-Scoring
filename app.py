import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load preprocessor and model
with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit Page Config
st.set_page_config(page_title="Credit Risk Scoring Dashboard", layout="wide")

# Title
st.title("📊 Credit Risk Scoring Dashboard")
st.markdown("Adjust loan details and click Predict to see updated results.")

# --- Loan Applicant Inputs ---
st.markdown("### 📝 Loan Applicant Inputs")
st.markdown("Enter the details below:")

loan_amnt = st.number_input("💰 Loan Amount", min_value=1000, value=10000, step=500)
annual_inc = st.number_input("📈 Annual Income", min_value=1000, value=50000, step=1000)
dti = st.number_input("⚖️ DTI (Debt-to-Income Ratio)", min_value=0.0, value=15.0, step=0.1)

if st.button("Predict"):
    # Prepare input
    input_data = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "annual_inc": annual_inc,
        "dti": dti
    }])

    # Preprocess
    input_data_processed = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(input_data_processed)[0]
    prediction_prob = model.predict_proba(input_data_processed)[0]
    prob_fully_paid = round(prediction_prob[1] * 100, 2)
    prob_default = round(prediction_prob[0] * 100, 2)

    # --- Prediction Results ---
    st.markdown("### 📊 Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Probability of Fully Paid\n\n**{prob_fully_paid}%**")
    with col2:
        st.error(f"⚠️ Probability of Default\n\n**{prob_default}%**")

    # --- Visual Analytics ---
    st.markdown("### 📈 Visual Analytics")
    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        st.subheader("📈 Feature Impact Visualization")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data_processed)

        # Bar chart for feature contributions
        feature_importance = dict(zip(["Loan Amount", "Annual Income", "DTI"], shap_values[0]))
        plt.figure(figsize=(5, 3))
        plt.barh(list(feature_importance.keys()), list(feature_importance.values()), color="skyblue", edgecolor="black")
        plt.xlabel("SHAP Value (Impact)")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

    with graph_col2:
        st.subheader("📊 Prediction Probability Distribution")
        plt.figure(figsize=(5, 3))
        plt.bar(["Fully Paid", "Default"], [prob_fully_paid, prob_default],
                color=["green", "red"], edgecolor="black")
        plt.ylabel("Probability (%)")
        plt.ylim(0, 100)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

# Footer
st.markdown("---")
st.markdown("⚡ Interactive Credit Risk Dashboard by **Anupriya**")
