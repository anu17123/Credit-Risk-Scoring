import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Load Preprocessor & Model
# -----------------------
with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------
# App Config
# -----------------------
st.set_page_config(page_title="📊 Credit Risk Scoring Dashboard", layout="wide")

st.title("📊 Credit Risk Scoring Dashboard")
st.markdown("### Adjust loan details and click **Predict** to see updated results.")

# Two main sections: Inputs (left) | Outputs (right)
col1, col2 = st.columns([1, 2], gap="large")

# -----------------------
# Left Column (Inputs)
# -----------------------
with col1:
    st.subheader("📝 Loan Applicant Inputs")
    st.markdown("Enter the details below:")

    loan_amnt = st.number_input("💰 Loan Amount", min_value=1000, value=10000, step=500)
    annual_inc = st.number_input("📈 Annual Income", min_value=1000, value=50000, step=1000)
    dti = st.number_input("⚖️ DTI (Debt-to-Income Ratio)", min_value=0.0, value=15.0, step=0.1)

    predict_btn = st.button("🔮 Predict", use_container_width=True)

# -----------------------
# Right Column (Results + Graphs)
# -----------------------
with col2:
    if predict_btn:
        # Prepare input
        X_new = pd.DataFrame([{ "loan_amnt": loan_amnt, "annual_inc": annual_inc, "dti": dti }])
        X_new_scaled = preprocessor.transform(X_new)

        prediction = model.predict(X_new_scaled)[0]
        prediction_prob = model.predict_proba(X_new_scaled)[0][1]

        prob_fully_paid = 100 * (1 - prediction_prob)
        prob_default = 100 * prediction_prob

        # --- Results Section ---
        st.subheader("📊 Prediction Results")

        # Metrics in two columns
        m1, m2 = st.columns(2)
        m1.metric(label="✅ Probability of Fully Paid", value=f"{prob_fully_paid:.2f}%")
        m2.metric(label="⚠️ Probability of Default", value=f"{prob_default:.2f}%")

        st.divider()

        # --- Feature Impact Visualization ---
        st.subheader("📈 Feature Impact Visualization")
        feature_df = pd.DataFrame({
            "Feature": ["Loan Amount", "Annual Income", "DTI"],
            "Value": [loan_amnt, annual_inc, dti]
        })

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=feature_df, x="Feature", y="Value", palette="viridis", ax=ax)
        ax.set_title("Input Feature Values", fontsize=10)
        st.pyplot(fig, use_container_width=True)

        st.divider()

        # --- Probability Distribution ---
        st.subheader("📊 Prediction Probability Distribution")
        prob_df = pd.DataFrame({
            "Outcome": ["Fully Paid", "Default"],
            "Probability": [prob_fully_paid, prob_default]
        })

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.barplot(data=prob_df, x="Outcome", y="Probability", palette=["green", "red"], ax=ax2)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Probability (%)")
        ax2.set_title("Risk Distribution", fontsize=10)
        st.pyplot(fig2, use_container_width=True)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>⚡ Interactive Credit Risk Dashboard by <b>Anupriya</b></p>", unsafe_allow_html=True)
