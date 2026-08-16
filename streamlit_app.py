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
    "Upload Custom Test Dataset (.csv or .xlsx)", 
    type=["csv", "xlsx", "xls"],
    help="Upload custom test dataset. If left empty, default 'test_data.csv' is evaluated automatically."
)

model_choice = st.sidebar.selectbox(
    "Select ML Model for Deep-Dive",
    list(model_file_map.keys()),
    index=4  # Default to Random Forest (Ensemble)
)

# Robust reader against quoted single-column Excel re-saves
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

# Flexible Ingestion Function (Default Baseline vs Custom Upload)
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
    
    # 2. Fallback to Default Local Baseline (test_data.csv)
    else:
        found_file = None
        for candidate in ['test_data.csv', 'test_data.xls', 'test_data.xlsx']:
            if os.path.exists(candidate):
                found_file = candidate
                break

        if found_file is None:
            st.error("❌ `test_data.csv` was not found in the root directory!")
            st.stop()

        try:
            df = _read_csv_robust(found_file)
        except Exception:
            df = pd.read_excel(found_file, header=0)
            
        source_label = f"Default Baseline (`{found_file}`)"
        dataset_name = found_file
    
    # Standardize Column Headers
    df.columns = [str(c).strip() for c in df.columns]

    # Drop explicit primary key / ID / index columns if present
    id_cols = [c for c in df.columns if c.lower() in ['id', 'unnamed: 0', 'index', 'customer_id']]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)

    # Dynamic Target Column Matching
    exact_matches = [c for c in df.columns if c.lower() == 'default']
    if exact_matches:
        target_col = exact_matches[0]
    else:
        fuzzy_matches = [c for c in df.columns if 'default' in c.lower() or 'target' in c.lower() or 'diagnosis' in c.lower()]
        if fuzzy_matches:
            target_col = fuzzy_matches[0]
        else:
            target_col = df.columns[-1]

    df.rename(columns={target_col: 'default'}, inplace=True)

    if df.shape[1] < 2:
        st.error(
            f"❌ Only 1 column detected in `{dataset_name}`. "
            "Please ensure the file has valid features and comma delimiting."
        )
        st.stop()

    return df, source_label, dataset_name

df_test, data_source, active_file_name = load_eval_data()

# Separate Features (X) and Target (y)
X_test_df = df_test.drop(columns=['default'])
y_test = pd.to_numeric(df_test['default'], errors='coerce').fillna(0).astype(int).to_numpy()

# Dashboard Header Title
st.title("💳 Asset Management: Credit Card Default Risk Classifier")
st.markdown(f"**Current Evaluation Target:** Running live model inference on **`{active_file_name}`** ({len(df_test):,} records, {X_test_df.shape[1]} raw features).")
st.divider()

st.sidebar.info(f"📊 Active Data Source:\n\n**{data_source}**\n\nTotal Records: **{len(df_test):,}**\n\nFeatures: **{X_test_df.shape[1]}**")

# Robust Helper Function to Predict and Auto-Align Features
def safe_predict(pipeline, X_df):
    X_pred = X_df.copy()

    # Convert all columns to numeric float64
    for c in X_pred.columns:
        X_pred[c] = pd.to_numeric(X_pred[c], errors='coerce').fillna(0.0)

    # Align features if the pipeline was trained with feature names
    if hasattr(pipeline, "feature_names_in_"):
        expected_cols = list(pipeline.feature_names_in_)
        
        # Exact expected feature names present -> reorder
        if set(expected_cols).issubset(set(X_pred.columns)):
            X_pred = X_pred[expected_cols]
        # Same column count -> rename to match model
        elif len(X_pred.columns) == len(expected_cols):
            X_pred.columns = expected_cols
        # Uploaded dataset has extra features -> select expected columns or slice first N
        elif len(X_pred.columns) > len(expected_cols):
            cols_lower = {c.lower(): c for c in X_pred.columns}
            matched_cols = [cols_lower[exp.lower()] for exp in expected_cols if exp.lower() in cols_lower]
            if len(matched_cols) == len(expected_cols):
                X_pred = X_pred[matched_cols]
                X_pred.columns = expected_cols
            else:
                X_pred = X_pred.iloc[:, :len(expected_cols)]
                X_pred.columns = expected_cols
        # Uploaded dataset has fewer features -> pad missing with 0.0
        else:
            for col in expected_cols:
                if col not in X_pred.columns:
                    X_pred[col] = 0.0
            X_pred = X_pred[expected_cols]

    # Convert to clean numpy float64 matrix
    arr = X_pred.to_numpy(dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0)

    # Wrap in DataFrame with expected feature names if available
    if hasattr(pipeline, "feature_names_in_"):
        X_eval = pd.DataFrame(arr, columns=pipeline.feature_names_in_, dtype=np.float64)
    else:
        X_eval = arr

    y_pred = pipeline.predict(X_eval)
    
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_eval)[:, 1]
    else:
        y_proba = y_pred
        
    return y_pred, y_proba

# Compute metrics across all 5 models
def evaluate_all_models_live(X_df_input, y_true):
    summary_results = []
    for model_name, path in model_file_map.items():
        if os.path.exists(path):
            pipe = joblib.load(path)
            
            y_pred, y_proba = safe_predict(pipe, X_df_input)
            
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
    st.markdown("Live evaluation of all 5 classification pipelines calculated directly on active test data:")
    
    summary_df = evaluate_all_models_live(X_test_df, y_test)
    
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
        y_pred, y_proba = safe_predict(pipeline, X_test_df)
        
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
            
            # Display Legend explaining 0 and 1
            st.caption("🏷️ **Axis Key:** `0` = Negative Class (e.g., Non-Default / Benign) | `1` = Positive Class (e.g., Default / Malignant)")

            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                ax=ax, 
                cbar=False,
                xticklabels=['0', '1'],
                yticklabels=['0', '1']
            )
            plt.ylabel('Actual Class')
            plt.xlabel('Predicted Class')
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