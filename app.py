import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")


# PAGE CONFIG

st.set_page_config(
    page_title="XplainCredit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #00FF00;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #EBEBEB;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-low      { background: #d4edda; border: 2px solid #28a745; }
    .risk-medium   { background: #fff3cd; border: 2px solid #ffc107; }
    .risk-high     { background: #ffe5d0; border: 2px solid #fd7e14; }
    .risk-veryhigh { background: #f8d7da; border: 2px solid #dc3545; }
    .risk-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .risk-score {
        font-size: 1.1rem;
        margin-top: 0.3rem;
    }
    .metric-box {
        background: #15273C;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #CCCCCC;
        border-left: 4px solid #4361ee;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    .tip-box {
        background: black;
        border-left: 4px solid #4361ee;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    div[data-testid="stAlert"] {
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)


# LOAD MODEL & ENCODERS

@st.cache_resource
def load_artifacts():
    model    = joblib.load("model/xplaincredit_model.pkl")
    le_emp   = joblib.load("model/le_employment.pkl")
    le_edu   = joblib.load("model/le_education.pkl")
    le_purp  = joblib.load("model/le_purpose.pkl")
    features = joblib.load("model/feature_names.pkl")
    return model, le_emp, le_edu, le_purp, features

@st.cache_resource
def load_training_data():
    df = pd.read_csv("data/loan_data.csv")
    le_emp   = joblib.load("model/le_employment.pkl")
    le_edu   = joblib.load("model/le_education.pkl")
    le_purp  = joblib.load("model/le_purpose.pkl")
    features = joblib.load("model/feature_names.pkl")
    df["employment_type_enc"] = le_emp.transform(df["employment_type"])
    df["education_enc"]       = le_edu.transform(df["education"])
    df["loan_purpose_enc"]    = le_purp.transform(df["loan_purpose"])
    return df[features]

try:
    model, le_emp, le_edu, le_purp, FEATURES = load_artifacts()
    X_train_bg = load_training_data()
except FileNotFoundError:
    st.error("Model files not found. Please run `python train_model.py` first.")
    st.stop()


# HELPER FUNCTIONS

RISK_TIERS = {
    (0.00, 0.25): ("Low Risk",       "🟢", "risk-low",      "#28a745"),
    (0.25, 0.50): ("Medium Risk",    "🟡", "risk-medium",   "#ffc107"),
    (0.50, 0.75): ("High Risk",      "🟠", "risk-high",     "#fd7e14"),
    (0.75, 1.00): ("Very High Risk", "🔴", "risk-veryhigh", "#dc3545"),
}

def get_risk_tier(prob):
    for (lo, hi), info in RISK_TIERS.items():
        if lo <= prob < hi or (prob == 1.0 and hi == 1.00):
            return info
    return ("Very High Risk", "🔴", "risk-veryhigh", "#dc3545")

FEATURE_LABELS = {
    "age":                "Age",
    "income":             "Monthly Income (₹)",
    "loan_amount":        "Loan Amount (₹)",
    "loan_tenure":        "Loan Tenure (months)",
    "cibil_score":        "CIBIL Score",
    "existing_loans":     "Existing Loans",
    "monthly_emi":        "Monthly EMI (₹)",
    "emi_to_income":      "EMI-to-Income Ratio",
    "employment_type_enc":"Employment Type",
    "education_enc":      "Education Level",
    "loan_purpose_enc":   "Loan Purpose",
}

def build_input_df(age, income, loan_amount, loan_tenure,
                   emp_type, cibil, existing, education, purpose):
    monthly_emi   = (loan_amount / loan_tenure) * (1 + 0.12 / 12)
    emi_to_income = monthly_emi / income if income > 0 else 0

    emp_enc  = le_emp.transform([emp_type])[0]
    edu_enc  = le_edu.transform([education])[0]
    purp_enc = le_purp.transform([purpose])[0]

    row = {
        "age":                age,
        "income":             income,
        "loan_amount":        loan_amount,
        "loan_tenure":        loan_tenure,
        "cibil_score":        cibil,
        "existing_loans":     existing,
        "monthly_emi":        round(monthly_emi, 2),
        "emi_to_income":      round(emi_to_income, 4),
        "employment_type_enc":emp_enc,
        "education_enc":      edu_enc,
        "loan_purpose_enc":   purp_enc,
    }
    return pd.DataFrame([row])[FEATURES], monthly_emi, emi_to_income

def get_improvement_tips(prob, cibil, emi_to_income, existing_loans, income):
    tips = []
    if cibil < 700:
        tips.append(f"Improve your CIBIL score from {cibil} to above 700 — this alone can drop risk significantly.")
    if emi_to_income > 0.4:
        tips.append(f"Your EMI-to-income ratio is {emi_to_income:.1%}. Try reducing the loan amount or extending the tenure.")
    if existing_loans > 2:
        tips.append(f"You have {existing_loans} existing loans. Closing 1–2 of them before applying will help.")
    if income < 30000:
        tips.append("A higher reported income or a co-applicant can improve your eligibility.")
    if not tips:
        tips.append("Your profile is strong. Maintaining your CIBIL score and low EMI burden will keep risk low.")
    return tips


# SIDEBAR — INPUT FORM

with st.sidebar:
    st.markdown("## 📋 Applicant Details")
    st.markdown("Fill in the loan applicant's information below.")
    st.markdown("---")

    age         = st.slider("Age", 21, 65, 32)
    income      = st.number_input("Monthly Income (₹)", 10000, 500000, 45000, step=1000)
    loan_amount = st.number_input("Loan Amount (₹)", 50000, 5000000, 500000, step=10000)
    loan_tenure = st.selectbox("Loan Tenure (months)", [12, 24, 36, 48, 60], index=2)
    cibil       = st.slider("CIBIL Score", 300, 900, 720)
    existing    = st.slider("Existing Active Loans", 0, 5, 1)
    emp_type    = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business"])
    education   = st.selectbox("Education", ["Graduate", "Post-Graduate", "Under-Graduate"])
    purpose     = st.selectbox("Loan Purpose", ["Home", "Personal", "Education", "Business", "Vehicle"])

    st.markdown("---")
    predict_btn = st.button("🔍 Analyze Risk", use_container_width=True, type="primary")


# MAIN AREA

st.markdown('<p class="main-title">💳 XplainCredit</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Explainable AI for Loan Default Risk Assessment</p>', unsafe_allow_html=True)

if not predict_btn:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h2>🎯</h2>
            <b>Risk Tier Prediction</b><br>
            <small>Low / Medium / High / Very High</small>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h2>🔎</h2>
            <b>SHAP Explainability</b><br>
            <small>Why was this decision made?</small>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h2>💡</h2>
            <b>What-If Simulator</b><br>
            <small>How can the applicant improve?</small>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Fill in the applicant details in the sidebar and click **Analyze Risk** to get started.")

else:
    # Build input + predict 
    input_df, monthly_emi, emi_to_income = build_input_df(
        age, income, loan_amount, loan_tenure,
        emp_type, cibil, existing, education, purpose
    )
    prob       = model.predict_proba(input_df)[0][1]
    risk_label, risk_icon, risk_class, risk_color = get_risk_tier(prob)

    # ROW 1: Risk card + key metrics 
    col_risk, col_metrics = st.columns([1, 2])

    with col_risk:
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <p style="font-size:3rem;margin:0">{risk_icon}</p>
            <p class="risk-label" style="color:{risk_color}">{risk_label}</p>
            <p class="risk-score">Default Probability: <b>{prob:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        st.markdown('<p class="section-header">Key Financial Metrics</p>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Monthly EMI", f"₹{monthly_emi:,.0f}")
        m2.metric("EMI / Income", f"{emi_to_income:.1%}",
                  delta="High" if emi_to_income > 0.4 else "OK",
                  delta_color="inverse")
        m3.metric("CIBIL Score", cibil,
                  delta="Good" if cibil >= 700 else "Low",
                  delta_color="normal" if cibil >= 700 else "inverse")
        m4.metric("Existing Loans", existing,
                  delta="Risky" if existing > 2 else "Fine",
                  delta_color="inverse" if existing > 2 else "normal")

    st.markdown("---")

    # ROW 2: SHAP + Tips 
    col_shap, col_tips = st.columns([3, 2])

    with col_shap:
        st.markdown('<p class="section-header">🔎 Why this decision? (SHAP Explanation)</p>',
                    unsafe_allow_html=True)

        with st.spinner("Calculating SHAP values..."):
            try:
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)

                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                else:
                    sv = shap_values[0]

                fig, ax = plt.subplots(figsize=(7, 4))
                feature_display = [FEATURE_LABELS.get(f, f) for f in FEATURES]
                colors = ["#dc3545" if v > 0 else "#28a745" for v in sv]
                sorted_idx = np.argsort(np.abs(sv))[::-1][:8]
                vals   = sv[sorted_idx]
                labels = [feature_display[i] for i in sorted_idx]
                bar_colors = ["#dc3545" if v > 0 else "#28a745" for v in vals]

                bars = ax.barh(range(len(vals)), vals[::-1], color=bar_colors[::-1], height=0.6)
                ax.set_yticks(range(len(vals)))
                ax.set_yticklabels(labels[::-1], fontsize=9)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP Value (impact on default probability)", fontsize=8)
                ax.set_title("Feature Impact — Red = increases risk, Green = reduces risk", fontsize=9)
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.caption("The longer the bar, the more that feature influenced the prediction.")
            except Exception as e:
                st.error(f"SHAP calculation error: {e}")

    with col_tips:
        st.markdown('<p class="section-header">💡 How to Improve Approval Chances</p>',
                    unsafe_allow_html=True)
        tips = get_improvement_tips(prob, cibil, emi_to_income, existing, income)
        for tip in tips:
            st.markdown(f'<div class="tip-box">• {tip}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ROW 3: WHAT-IF SIMULATOR
    st.markdown('<p class="section-header">🧪 What-If Simulator — See how changes affect risk</p>',
                unsafe_allow_html=True)
    st.caption("Adjust the sliders below to simulate changes and see the impact on risk instantly.")

    s1, s2, s3 = st.columns(3)
    with s1:
        sim_cibil = st.slider("Simulate CIBIL Score", 300, 900, cibil, key="sim_cibil")
    with s2:
        sim_income = st.slider("Simulate Monthly Income (₹)", 10000, 500000, income,
                                step=1000, key="sim_income")
    with s3:
        sim_loan = st.slider("Simulate Loan Amount (₹)", 50000, 5000000, loan_amount,
                              step=10000, key="sim_loan")

    sim_df, sim_emi, sim_ratio = build_input_df(
        age, sim_income, sim_loan, loan_tenure,
        emp_type, sim_cibil, existing, education, purpose
    )
    sim_prob = model.predict_proba(sim_df)[0][1]
    sim_label, sim_icon, sim_class, sim_color = get_risk_tier(sim_prob)

    sa, sb, sc = st.columns(3)
    delta_prob = sim_prob - prob
    sa.metric("Original Risk",   f"{prob:.1%}",     label_visibility="visible")
    sb.metric("Simulated Risk",  f"{sim_prob:.1%}",
              delta=f"{delta_prob:+.1%}",
              delta_color="inverse")
    sc.markdown(f"""
    <div class="risk-card {sim_class}" style="padding:0.8rem">
        <p style="margin:0;font-size:1.4rem">{sim_icon} <b style="color:{sim_color}">{sim_label}</b></p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ROW 4: FULL REPORT 
    with st.expander("📄 View Full Applicant Report"):
        st.markdown(f"""
| Field | Value |
|---|---|
| Age | {age} years |
| Monthly Income | ₹{income:,} |
| Loan Amount | ₹{loan_amount:,} |
| Loan Tenure | {loan_tenure} months |
| Monthly EMI | ₹{monthly_emi:,.0f} |
| EMI-to-Income Ratio | {emi_to_income:.1%} |
| CIBIL Score | {cibil} |
| Existing Loans | {existing} |
| Employment Type | {emp_type} |
| Education | {education} |
| Loan Purpose | {purpose} |
| **Default Probability** | **{prob:.1%}** |
| **Risk Tier** | **{risk_icon} {risk_label}** |
        """)

st.markdown('<p class="footer">XplainCredit • Built with Python, XGBoost & SHAP • Final Year Project</p>',
            unsafe_allow_html=True)
