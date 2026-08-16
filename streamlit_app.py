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

# Page Setup
st.set_page_config(
    page_title="Asset Management Credit Default App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model Directory Mapping
model_file_map = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl"
}

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

# Robust CSV reader guard against quoted single-column Excel re-saves
def _read_csv_robust(file_obj_or_path):
    df = pd.read_csv(file_obj_or_path)
    if df.shape[1] == 1:
        col_name = df.columns[0]
        if ',' in col_name:
            if hasattr(file_obj_or_path, 'seek'):
                file_obj_or_path.seek(0)
            df = pd.read_csv(
                file_obj_or_path,
                quotechar='"',
                skipinitialspace=True,
            )
            if df.shape[1] == 1:
                split_cols = [c.strip() for c in col_name.split(',')]
                values = df[col_name].astype(str).str.split(',', expand=True)
                values.columns = split_cols
                df = values
    return df

# Data Ingestion Function
def load_eval_data():
    # 1. Custom Uploaded File
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.csv'):
            try:
                df = _read_csv_robust(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, header=0)
        else:
            try:
                df = pd.read_excel(uploaded_file, header=0)
            except Exception:
                uploaded_file.seek(0)
                df = _read_csv_robust(uploaded_file)
        source_label = f"Uploaded File (`{uploaded_file.name}`)"
        dataset_name = uploaded_file.name
    
    # 2. Default Display Fallback (test_data.csv)
    else:
        found_file = None
        for candidate in ['test_data.csv', 'test_data.xls', 'test_data.xlsx']:
            if os.path.exists(candidate):
                found_file = candidate
                break

        if found_file is None:
            st.error("❌ `test_data.csv` was not found in the root directory! Run model training first.")
            st.stop()

        try:
            df = _read_csv_robust(found_file)
        except Exception:
            df = pd.read_excel(found_file, header=0)
            
        source_label = f"Default Baseline (`{found_file}`)"
        dataset_name = found_file
    
    # Clean column headers
    df.columns = [str(c).strip() for c in df.columns]

    # Drop explicit primary key / ID columns if present
    id_cols = [c for c in df.columns if c.lower() in ['id', 'unnamed: 0', 'index', 'customer_id']]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)

    # Flexible Target Column Matching Logic:
    # 1. Look for exact match 'default'
    # 2. Look for fuzzy substring containing 'default'
    # 3. Fallback to using the LAST column as the target variable
    exact_matches = [c for c in df.columns if c.lower() == 'default']
    if exact_matches:
        target_col = exact_matches[0]
    else:
        fuzzy_matches = [c for c in df.columns if 'default' in c.lower()]
        if fuzzy_matches:
            target_col = fuzzy_matches[0]
        else:
            target_col = df.columns[-1]

    df.rename(columns={target_col: 'default'}, inplace=True)

    if df.shape[1] < 2:
        st.error(
            f"❌ Only 1 column was detected in `{dataset_name}`. "
            "Please ensure the file has valid features and comma delimiting."
        )
        st.stop()

    return df, source_label, dataset_name

df_test, data_source, active_file_name = load_eval_data()

# Dynamic Header Title
st.title("💳 Asset Management: Credit Card Default Risk Classifier")
st.markdown(f"**Current Evaluation Target:** Running live model inference on **`{active_file_name}`** ({len(df_test):,} records).")
st.divider()

st.sidebar.info(f"📊 Active File: **{active_file_name}**\n\nTotal Records: **{len(df_test):,}**")

# Separate Features (X) and Target (y)
X_test_raw = df_test.drop(columns=['default'])
y_test = pd.to_numeric(df_test['default'], errors='coerce').fillna(0).astype(int).to_numpy()

# Clean input matrix as float64 contiguous numpy array
X_test_matrix = np.ascontiguousarray(
    X_test_raw.apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
)

# Helper function to align features and perform safe model prediction
def safe_predict(pipeline, X_input):
    if isinstance(X_input, pd.DataFrame):
        arr = X_input.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(X_input, dtype=np.float64)
        
    arr = np.nan_to_num(arr, nan=0.0)
    
    if hasattr(pipeline, "feature_names_in_"):
        expected_cols = pipeline.feature_names_in_
        if len(expected_cols) == arr.shape[1]:
            X_eval = pd.DataFrame(arr, columns=expected_cols, dtype=np.float64)
        else:
            X_eval = arr
    else:
        X_eval = arr

    y_pred = pipeline.predict(X_eval)
    
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_eval)[:, 1]
    else:
        y_proba = y_pred
        
    return y_pred, y_proba

# Compute metrics across all 5 models
def evaluate_all_models_live(X_mat, y_true):
    summary_results = []
    for model_name, path in model_file_map.items():
        if os.path.exists(path):
            pipe = joblib.load(path)
            
            y_pred, y_proba = safe_predict(pipe, X_mat)
            
            summary_results.append({
                'ML Model Name': model_name,
                'Accuracy': round(accuracy_score(y_true, y_pred), 4),
                'AUC': round(roc_auc_score(y_true, y_proba), 4),
                'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
                'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
                'F1': round(f1_score(y_true, y_pred, zero_division=0), 4),
                'MCC': round(matthews_corrcoef(y_true, y_pred), 4)
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
    st.markdown("Live evaluation of all 5 classification pipelines calculated directly on test data:")
    
    summary_df = evaluate_all_models_live(X_test_matrix, y_test)
    
    if not summary_df.empty:
        styled_table = style_high_contrast(summary_df)
        st.dataframe(
            styled_table,
            use_container_width=True,
            hide_index=True
        )
        
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
        y_pred, y_proba = safe_predict(pipeline, X_test_matrix)
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
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
