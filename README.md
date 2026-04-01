<<<<<<< HEAD
# XplainCredit
=======
# 💳 XplainCredit
### Explainable AI for Loan Default Risk Assessment

> A final year B.Tech CSE (Data Science) project that predicts loan default risk and **explains every decision** using SHAP — mimicking what real Indian fintech companies and NBFCs need for transparent lending.

---

## 🚀 What Makes This Different

| Feature | Typical Project | XplainCredit |
|---|---|---|
| Output | Binary yes/no | 4-tier risk level |
| Explainability | None | SHAP waterfall per applicant |
| Interactivity | Static notebook | Live web app |
| Business angle | Accuracy only | EMI-to-income, CIBIL logic |
| What-if analysis | ❌ | ✅ Simulate changes live |

---

## 🛠 Tech Stack

- **Python 3.10+**
- **XGBoost** — best performing model (vs Logistic Regression, Random Forest)
- **SHAP** — individual prediction explainability
- **Streamlit** — interactive web application
- **Pandas / NumPy / Scikit-learn** — data processing & evaluation
- **Matplotlib** — SHAP visualizations

---

## 📁 Project Structure

```
XplainCredit/
├── app.py              ← Streamlit web app (main file)
├── train_model.py      ← Data generation + model training
├── requirements.txt    ← All dependencies
├── data/
│   └── loan_data.csv   ← Generated after running train_model.py
├── model/
│   ├── xplaincredit_model.pkl
│   ├── le_employment.pkl
│   ├── le_education.pkl
│   ├── le_purpose.pkl
│   └── feature_names.pkl
└── README.md
```

---

## ⚙️ How to Run Locally

### Step 1 — Clone / download the project
```bash
cd XplainCredit
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the model (run once)
```bash
python train_model.py
```

### Step 5 — Launch the app
```bash
streamlit run app.py
```

Open your browser at → **http://localhost:8501**

---

## 🎯 Features

- **Risk Tier Classification** — Low / Medium / High / Very High with default probability
- **SHAP Explainability** — Bar chart showing which features increased or decreased risk
- **Key Metrics Dashboard** — EMI, EMI-to-income ratio, CIBIL score at a glance
- **Improvement Tips** — Actionable advice for the applicant
- **What-If Simulator** — Change CIBIL, income, or loan amount and see risk update live
- **Full Applicant Report** — Downloadable summary of all inputs and outputs

---

## 📊 Model Performance

Three models were trained and compared:

| Model | AUC-ROC | F1 Score |
|---|---|---|
| Logistic Regression | ~0.78 | ~0.62 |
| Random Forest | ~0.85 | ~0.70 |
| **XGBoost** | **~0.89** | **~0.76** |

XGBoost was selected as the final model. Class imbalance was handled using `scale_pos_weight`.

---

## 💼 Resume One-Liner

> *XplainCredit — An explainable ML system that predicts loan default risk and justifies each decision using SHAP, with a what-if simulator deployed as an interactive web app.*

---

## 👤 Author

Built as Final Year Project | B.Tech CSE (Data Science)
>>>>>>> dda4d930 (Initial commit)
