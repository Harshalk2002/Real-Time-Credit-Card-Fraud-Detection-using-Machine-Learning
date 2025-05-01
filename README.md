# 🛡️ Credit Card Fraud Detection

A machine learning pipeline designed to detect fraudulent credit card transactions in real-time, using the highly imbalanced but popular Kaggle dataset. Built as part of the MSA 8150 - Machine Learning for Analytics course.

## 📂 Project Structure

- `Fraud_detection.ipynb` – Full notebook with data exploration, preprocessing, model training, tuning, and evaluation.
- `Fraud Detection in Credit Card Transactions(Report).pdf` – Final project report.
- `machine learning final presentation.pdf` – Presentation slides outlining problem, model choices, results, and business impact.

## 🧠 Problem Statement

Credit card fraud causes over **$28B in global losses annually**, requiring advanced, automated solutions for real-time detection. This project aims to:

- Maximize **recall** on fraudulent transactions while maintaining **acceptable precision**.
- Build a scalable and interpretable solution deployable in real-time.

## 📊 Dataset Overview

- Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions | 492 frauds (0.17%)
- Features: 28 anonymized PCA components (V1–V28), Time, and Amount
- Target: `Class` (1 = Fraud, 0 = Legitimate)

## 🔧 Methods & Tools

- **Language**: Python 3.10
- **Libraries**: Pandas, scikit-learn, XGBoost, Matplotlib, Seaborn
- **Models Used**:
  - Logistic Regression (baseline)
  - Random Forest (with hyperparameter tuning)
  - XGBoost (gradient boosting)
- **Techniques**:
  - Log transformation & standard scaling
  - Threshold tuning using Precision-Recall curve
  - 3-fold Stratified Cross-Validation

## 🚀 Results

| Model              | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.06      | 0.90   | 0.11     | 0.927   |
| Random Forest       | 0.87      | 0.77   | 0.82     | 0.9966  |
| Tuned RF (0.70 th)  | **0.90**  | **0.80** | **0.85** | **0.997** |

> 📌 Final Model: Tuned Random Forest with optimal threshold = 0.70

## 📈 Business Implications

- Real-time scoring of transactions.
- Automatic blocking (probability > 0.85), human review (0.70–0.85).
- Reduces manual reviews while maintaining high fraud catch rate.
- Highlights top predictors (e.g., V14, V17, log-scaled Amount) for risk explanations.

## ⚠️ Limitations & Future Work

- PCA anonymization reduces interpretability.
- Dataset spans only 48 hours — lacks seasonality insights.
- Future: integrate temporal features, SMOTE/ADASYN, ensemble methods like autoencoders.



## 📄 License

MIT License (for academic use)

---

