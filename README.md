# ⚽ Football Match Winner Predictor

A Flask-based machine learning web app that predicts whether the **home team will win** using historical football data and engineered features.

### 🧠 Model

* **Algorithm:** Logistic Regression
* **Features:** Recent Form, H2H Win Rate, Form Difference, Goal Difference
* **5-Fold CV Accuracy:** **56.48%**
* **Test Accuracy:** **56.59%**

### 🛠️ Tech Stack

Python · Pandas · Scikit-learn · Flask · HTML/CSS · Pickle

### 🚀 Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

### ☁️ Deployment

Deployed using **Render** with Gunicorn.

> Predictions are machine-learning estimates and are not guaranteed outcomes.
