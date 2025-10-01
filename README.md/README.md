# 📊 Credit Risk Scoring Dashboard

An **Interactive Credit Risk Scoring Dashboard** built with Python, Streamlit, and XGBoost.  
This dashboard predicts the likelihood of a loan being fully paid or defaulted based on user-provided inputs, and provides visual analytics to interpret predictions.

---

## 🛠 Project Overview

This project uses:
- **Machine Learning** — XGBoost classifier for prediction
- **Preprocessing** — Scikit-learn for scaling and transformation
- **Dashboard** — Streamlit for interactive UI
- **Visualizations** — Dynamic charts to display feature impacts and prediction probabilities

**Key Inputs:**
- Loan Amount
- Annual Income
- Debt-to-Income Ratio (DTI)

---

## 📈 Features

- Interactive sliders for input values  
- Real-time prediction results  
- Probability distribution charts  
- Feature impact visualizations  
- Professional, responsive dashboard layout

---

## 🚀 How to Run

1. Clone the repository  
2. Create and activate the Python environment:

   conda create -n credit_env python=3.10
   conda activate credit_env
   
3. Install requirements:

pip install -r requirements.txt


4. Run the dashboard:

streamlit run app.py

🖼 Dashboard Screenshots
<table> <tr> <td><img src="dashboard_1.png" width="300"></td> <td><img src="dashboard_2.png" width="300"></td> <td><img src="dashboard_3.png" width="300"></td> </tr> </table>

📊 Prediction Output Example

Loan Applicant Inputs

Loan Amount: 10000

Annual Income: 50000

DTI: 15.00

Prediction Results

Probability of Fully Paid: 44.52%

Probability of Default: 55.48%

📂 Project Structure
Credit-Risk-Scoring/
│
├── app.py               # Main dashboard app
├── README.md           # Project documentation
├── data/               # Dataset and preprocessing files
├── models/             # Saved model and preprocessing objects
├── notebooks/          # Jupyter notebooks for EDA & model building
├── requirements.txt    # Python dependencies
└── images/             # Screenshots

📝 Author

Anupriya K
Data Science Enthusiast | Interactive Dashboard Developer
⚡ Interactive Credit Risk Dashboard by Anupriya

📌 License

This project is licensed under the MIT License — see the LICENSE
 file for details.