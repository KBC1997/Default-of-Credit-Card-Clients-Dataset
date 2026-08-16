# Asset Management: Credit Card Default Risk Classifier

## 1. Problem Statement
In financial asset management and banking domain operations, predicting credit default risk is critical to minimizing non-performing assets (NPAs), optimizing credit line allocations, and safeguarding financial capital[cite: 1]. 

This project implements an end-to-end machine learning solution to classify whether a credit card client will default on their payment in the next month based on 23 financial, demographic, and historical payment features[cite: 1, 2]. The goal is to compare 5 distinct classification algorithms across 6 standard evaluation metrics and deploy an interactive decision-support application on Streamlit Community Cloud[cite: 4].

---

## 2. Dataset Description
- **Source:** UCI Machine Learning Repository — *Default of Credit Card Clients Dataset*[cite: 1].
- **Instances:** 30,000 credit card clients.
- **Features:** 23 predictive input features + 1 binary target variable (`default`)[cite: 2].
- **Target Variable:** `default` (0 = Paid on time, 1 = Defaulted next month)[cite: 1, 2].

### Key Input Features:
- **Demographics:** `LIMIT_BAL` (Given credit limit), `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`[cite: 2].
- **Repayment Status (April–September):** `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` (History of past payment delays)[cite: 2].
- **Bill Statement Amounts:** `BILL_AMT1` to `BILL_AMT6` (Amount of bill statement from April to September)[cite: 2].
- **Previous Payment Amounts:** `PAY_AMT1` to `PAY_AMT6` (Amount of previous payment made from April to September)[cite: 2].

---

## 3. GitHub Repository Link
- **GitHub Repository:** https://github.com/KBC1997/Default-of-Credit-Card-Clients-Dataset/tree/main [cite: 4]
- **Live Streamlit App:** https://kbc1997-default-of-credit-card-clients-dat-streamlit-app-lc3thy.streamlit.app/ [cite: 4]

---

## 4. Models Used & Evaluation Metrics Comparison

All 5 classification models were trained using an **80/20 stratified train-test split** with scikit-learn preprocessing `Pipeline` objects (incorporating `StandardScaler` for distance-based and linear models)[cite: 2, 4].

### Evaluation Metrics Comparison Table:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| **Decision Tree** | 0.8172 | 0.7418 | 0.6615 | 0.3549 | 0.4620 | 0.3893 |
| **kNN** | 0.7928 | 0.7015 | 0.5487 | 0.3564 | 0.4322 | 0.3233 |
| **Naive Bayes** | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| **Random Forest (Ensemble)** | **0.8168** | **0.7731** | **0.6629** | **0.3497** | **0.4578** | **0.3868** |

---

## 5. Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves strong baseline accuracy (80.77%), but suffers from low recall (23.96%) on minority default cases due to linear boundary constraints on non-linear payment features[cite: 2]. |
| **Decision Tree** | Captures non-linear decision thresholds effectively with improved recall (35.49%) and high overall accuracy (81.72%), though susceptible to variance across split thresholds[cite: 2]. |
| **kNN** | Distance-based classification performs reasonably well (79.28% Accuracy, 0.7015 AUC) after feature scaling, but computation scales heavily with data size[cite: 2]. |
| **Naive Bayes** | Demonstrates the highest recall (55.39%) among all models by predicting default probability aggressively, though feature independence assumptions lead to lower overall precision (45.15%)[cite: 2]. |
| **Random Forest (Ensemble)** | Delivers superior performance across the board with the highest **AUC Score (0.7731)** and strong Matthews Correlation Coefficient (**MCC = 0.3868**), effectively balancing precision and recall via ensemble decision aggregation[cite: 2]. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — Chosen as the overall winner due to its dominant **AUC Score (0.7731)** and robust generalization across imbalanced default risk classes[cite: 2]. |

---

## 6. Project Directory Structure

```text
Asset-Management-Credit-Default/
├── app.py                          <-- Streamlit Interactive Frontend Application
├── ML Assignment 2.ipynb            <-- Main Notebook for Training, Pipeline Pickling & Metric Computation
├── test_data.csv                    <-- Sample Test Dataset (Generated during train-test split)
├── requirements.txt                <-- Streamlit Cloud Dependency Declarations
├── README.md                       <-- Comprehensive Markdown Documentation
└── model/                          <-- Model Directory (Saved artifacts)
    ├── Model.ipynb                 <-- Training Notebook Artifact
    ├── logistic_regression.pkl     <-- Saved Scikit-Learn Pipeline
    ├── decision_tree.pkl           <-- Saved Scikit-Learn Pipeline
    ├── knn.pkl                     <-- Saved Scikit-Learn Pipeline
    ├── naive_bayes.pkl             <-- Saved Scikit-Learn Pipeline
    └── random_forest_ensemble.pkl  <-- Saved Scikit-Learn Pipeline
