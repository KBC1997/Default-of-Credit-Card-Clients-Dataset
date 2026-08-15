import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Page Configuration
st.set_page_config(
    page_title="Asset Management Credit Default App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model File Directory Map
model_file_map = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl"
}

# Sidebar Controls
st.sidebar.header("⚙️ User Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data File (.csv or .xlsx)", 
    type=["csv", "xlsx", "xls"],
    help="Upload custom test dataset. If left empty, default 'test_data.csv' is evaluated automatically."
)

model_choice = st.sidebar.selectbox(
    "Select ML Model for Deep-Dive",
    list(model_file_map.keys()),
    index=4  # Default to Random Forest (Ensemble)
)

# Flexible Data Ingestion Function (Supports CSV and Excel)
def load_eval_data():
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, header=0)
        else:
            try:
                df = pd.read_excel(uploaded_file, header=0, engine='openpyxl')
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
        source_label = f"Uploaded Dataset: `{uploaded_file.name}`"
        dataset_name = uploaded_file.name
    
    else:
        found_file = None
        for candidate in ['test_data.csv', 'test_data.xls', 'test_data.xlsx']:
            if os.path.exists(candidate):
                found_file = candidate
                break

        if found_file is None:
            st.error("❌ No default test file (`test_data.csv`) found in root directory!")
            st.stop()

        try:
            df = pd.read_csv(found_file)
        except Exception:
            df = pd.read_excel(found_file, header=0)
            
        source_label = f"Default Baseline: `{found_file}`"
        dataset_name = found_file
    
    # Clean column headers
    df.columns = [str(c).strip() for c in df.columns]

    # Automatically drop primary key / ID columns
    id_cols = [c for c in df.columns if c.lower() in ['id', 'unnamed: 0', 'index', 'customer_id']]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)

    # Dynamic target column matching
    target_cols = [c for c in df.columns if 'default' in c.lower()]
    if not target_cols:
        st.error("❌ Target column containing 'default' keyword was not found in dataset.")
        st.stop()
    
    df.rename(columns={target_cols[0]: 'default'}, inplace=True)
    return df, source_label, dataset_name

df_test, data_source, active_file_name = load_eval_data()

# Dynamic Title based on active dataset
st.title("💳 Asset Management: Credit Card Default Risk Classifier")
st.markdown(f"**Current Evaluation Target:** Running live model inference on **`{active_file_name}`** ({len(df_test):,} instances, {df_test.shape[1]-1} features).")
st.divider()

st.sidebar.info(f"📊 Active File: **{active_file_name}**\n\nTotal Records: **{len(df_test):,}**")

X_test = df_test.drop(columns=['default'])
y_test = df_test['default']

# Compute all 6 metrics live across all 5 models directly from active dataset
@st.cache_data
def evaluate_all_models_live(_X, _y):
    summary_results = []
    for model_name, path in model_file_map.items():
        if os.path.exists(path):
            pipe = joblib.load(path)
            y_pred = pipe.predict(_X)
            y_proba = pipe.predict_proba(_X)[:, 1] if hasattr(pipe, "predict_proba") else y_pred
            
            summary_results.append({
                'ML Model Name': model_name,
                'Accuracy': round(accuracy_score(_y, y_pred), 4),
                'AUC': round(roc_auc_score(_y, y_proba), 4),
                'Precision': round(precision_score(_y, y_pred, zero_division=0), 4),
                'Recall': round(recall_score(_y, y_pred, zero_division=0), 4),
                'F1': round(f1_score(_y, y_pred, zero_division=0), 4),
                'MCC': round(matthews_corrcoef(_y, y_pred), 4)
            })
    return pd.DataFrame(summary_results)

# Custom High-Contrast Styling Function for Metric Table
def style_high_contrast(df):
    def highlight_max_contrast(s):
        is_max = s == s.max()
        return ['background-color: #1b5e20; color: #ffffff; font-weight: bold;' if v else '' for v in is_max]
    
    return df.style.apply(highlight_max_contrast, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'])

# Dashboard Layout Tabs
tab1, tab2, tab3 = st.tabs(["📊 Model Comparison Table", "🔍 Single Model Inspection", "📋 Dataset Preview"])

# TAB 1: LIVE MODEL COMPARISON TABLE
with tab1:
    st.subheader(f"🏆 Model Performance Comparison — `{active_file_name}`")
    st.markdown("Live evaluation of all 5 classification pipelines calculated directly on the active dataset:")
    
    summary_df = evaluate_all_models_live(X_test, y_test)
    
    if not summary_df.empty:
        # High-contrast styled table display
        styled_table = style_high_contrast(summary_df)
        st.dataframe(
            styled_table,
            use_container_width=True,
            hide_index=True
        )
        
        # Best performing model callout
        best_idx = summary_df['AUC'].idxmax()
        winner_name = summary_df.loc[best_idx, 'ML Model Name']
        winner_auc = summary_df.loc[best_idx, 'AUC']
        winner_acc = summary_df.loc[best_idx, 'Accuracy']
        
        st.success(f"🌟 **Overall Winner for `{active_file_name}`:** **{winner_name}** with **AUC = {winner_auc:.4f}** and **Accuracy = {winner_acc:.4f}**.")
    else:
        st.warning("No model `.pkl` files found inside `model/` folder.")

# TAB 2: DETAILED SINGLE MODEL DIAGNOSTICS
with tab2:
    st.subheader(f"🎯 Detailed Diagnostics: {model_choice}")
    
    model_path = model_file_map[model_choice]
    if not os.path.exists(model_path):
        st.error(f"Model file `{model_path}` not found.")
    else:
        pipeline = joblib.load(model_path)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Display 6 Metric Cards
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        st.divider()
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("##### 📌 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Paid (0)', 'Default (1)'],
                yticklabels=['Paid (0)', 'Default (1)']
            )
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            st.pyplot(fig)
            
        with c2:
            st.markdown("##### 📄 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

# TAB 3: DATASET PREVIEW
with tab3:
    st.subheader(f"📋 Evaluation Dataset Preview (`{active_file_name}`)")
    st.markdown(f"Displaying first 10 rows of active dataset (Total: **{len(df_test):,}** instances, **{df_test.shape[1]}** features):")
    st.dataframe(df_test.head(10), use_container_width=True)