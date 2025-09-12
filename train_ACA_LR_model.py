"""
ACA Logistic Regression Training Pipeline

This module implements the ACA LR pipeline described in paper_english.pdf:
- Load ACA/normal datasets and merge male/female cohorts
- Normalize chromosome read counts to 3M per sample (UR/ATRC based)
- Build features: GC (scaled x100), FF, chrN_count_3m
- Standardize features using training statistics only
- Train Logistic Regression (class_weight='balanced', C=100)
- Evaluate via ROC with Youden's J thresholding, show CM/metrics
- Save per-chromosome model/scaler artifacts

Inputs
- CSVs with columns: sample_id, result, GC, FF,
  T{13,18,21}_UR, T{13,18,21}_RC (ACA), ATRC, chr{13,18,21}_count (normal)

Outputs
- lr_model_T{N}.joblib, lr_scaler_T{N}.joblib per chromosome
- Printed metrics and confusion matrices
"""

from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Pipeline overview:
# 1) Load ACA/normal datasets and merge male/female .
# 2) Normalize chromosome read counts to 3M total per sample (using UR/ATRC).
# 3) Build features: GC (scaled x100), FF, chrN_count_3m.
# 4) Standardize features on training data only.
# 5) Train Logistic Regression (class_weight='balanced', C=100).
# 6) Evaluate with ROC; choose threshold via Youden's J; report CM and metrics.
# 7) Save per-chromosome model and scaler artifacts.


# Standardize features and train Logistic Regression, then predict on test
def fit_lr_model(x_train_data, y_train_data, x_test_data):
    """Standardize features on train, fit LR, and predict on test."""
    lr_scaler = StandardScaler()
    lr_scaler.fit(x_train_data)
    
    train = pd.DataFrame(lr_scaler.transform(x_train_data))
    test = pd.DataFrame(lr_scaler.transform(x_test_data.copy()))
    
    # Create logistic regression model
    lr_model = LogisticRegression(random_state=42,
                                  C=100,
                                  class_weight='balanced',
                                  )
    lr_model.fit(train, y_train_data)

    lr_pred = lr_model.predict(test)
    
    return lr_pred, lr_model, lr_scaler


# Plot confusion matrix (used during evaluation)
def plot_cm(cm, ax, label="Confusion Matrix", title="Confusion Matrix", fontsize=14, annot_kws=None):
    # Default annot_kws
    if annot_kws is None:
        annot_kws = {'size': 12}

    # Plot confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues', values_format='d', ax=ax, colorbar=False)

    # Adjust annotation font size
    for text in ax.texts:
        text.set_fontsize(annot_kws.get('size', 12))

    # Set title and axis labels
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('Predicted', fontsize=fontsize-2)
    ax.set_ylabel('True', fontsize=fontsize-2)

    # Configure tick label font size
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)

    ax.grid(False)

    # Tweak layout
    plt.tight_layout()


# Load CSVs, merge male/female, and rename columns to consistent schema
def load_data():
    """Load CSVs, merge cohorts, and align column names."""
    aca_cols = ['sample_id','result','GC','FF','T13_UR','T18_UR','T21_UR','T13_RC','T18_RC','T21_RC']
    normal_cols = ['sample_id','result','GC','FF','ATRC','chr13_count','chr18_count','chr21_count']

    # Load CSV datasets (insert your own dataset location here)
    ACA_male_train = pd.read_csv(r'./ACA_train_male.csv', usecols=ACA_load_cols)
    ACA_female_train = pd.read_csv(r'./ACA_train_female.csv', usecols=ACA_load_cols)
    normal_male_train = pd.read_csv(r'./normal_train_male.csv', usecols=normal_load_cols)
    normal_female_train = pd.read_csv(r'./normal_train_female.csv', usecols=normal_load_cols)

    ACA_male_test = pd.read_csv(r'./ACA_test_male.csv', usecols=ACA_load_cols)
    ACA_female_test = pd.read_csv(r'./ACA_test_female.csv', usecols=ACA_load_cols)
    normal_male_test = pd.read_csv(r'./normal_test_male.csv', usecols=normal_load_cols)
    normal_female_test = pd.read_csv(r'./normal_test_female.csv', usecols=normal_load_cols)

    # Merge male/female cohorts
    aca_train = pd.concat([aca_train_m, aca_train_f], ignore_index=True)
    normal_train = pd.concat([normal_train_m, normal_train_f], ignore_index=True)
    aca_test = pd.concat([aca_test_m, aca_test_f], ignore_index=True)
    normal_test = pd.concat([normal_test_m, normal_test_f], ignore_index=True)
    
    # Rename columns to consistent chromosome naming
    chromosomes = [13, 18, 21]
    for chr_num in chromosomes:
        aca_train = aca_train.rename(columns={f"T{chr_num}_RC": f"chr{chr_num}_count"})
        aca_test = aca_test.rename(columns={f"T{chr_num}_RC": f"chr{chr_num}_count"})

    return aca_train, aca_test, normal_train, normal_test, chromosomes


# Vectorized 3M normalization of chromosome counts per sample
def normalize_data(aca_train, aca_test, normal_train, normal_test, chromosomes):
    """Normalize chromosome counts to 3M per sample (vectorized)."""
    for chr_num in tqdm(chromosomes, desc="Processing chromosomes"):
        # 3M normalization for ACA (vectorized)
        factor_train = 3000000 / aca_train[f'T{chr_num}_UR']
        factor_test = 3000000 / aca_test[f'T{chr_num}_UR']
        
        aca_train[f'chr{chr_num}_count_3m'] = (aca_train[f'chr{chr_num}_count'] * factor_train).astype(int)
        aca_test[f'chr{chr_num}_count_3m'] = (aca_test[f'chr{chr_num}_count'] * factor_test).astype(int)
        
        # 3M normalization for Normal (vectorized)
        factor_normal_train = 3000000 / normal_train['ATRC']
        factor_normal_test = 3000000 / normal_test['ATRC']
        
        normal_train[f'chr{chr_num}_count_3m'] = (normal_train[f'chr{chr_num}_count'] * factor_normal_train).astype(int)
        normal_test[f'chr{chr_num}_count_3m'] = (normal_test[f'chr{chr_num}_count'] * factor_normal_test).astype(int)
    
    return aca_train, aca_test, normal_train, normal_test


# Assemble training sets: label ACA as positive (1) and normal as negative (0)
# GC is scaled by 100 as described in the paper
def prep_train_data(aca_train, normal_train, chromosomes):
    """Build per-chromosome training tables with labels and GC scaling."""
    train_data = {}
    
    for chr_num in chromosomes:
        pos_data = aca_train.copy()
        neg_data = normal_train.copy()
        
        cols = ['sample_id', 'GC', 'FF', 'result', f'chr{chr_num}_count_3m']
        pos_data = pos_data[cols]
        neg_data = neg_data[cols]
        
        pos_data['result'] = 1
        neg_data['result'] = 0
        
        pos_data['GC'] = pos_data['GC'] * 100
        neg_data['GC'] = neg_data['GC'] * 100
        
        train_data[chr_num] = pd.concat([pos_data, neg_data], ignore_index=True)
    
    return train_data


# Prepare held-out test sets per chromosome (same feature schema as training)
def prep_test_data(aca_test, normal_test, chr_num):
    """Build per-chromosome test table with the same schema as training."""
    pos_data = aca_test.copy()
    neg_data = normal_test.copy()
    
    cols = ['sample_id', 'GC', 'FF', 'result', f'chr{chr_num}_count_3m']
    pos_data = pos_data[cols]
    neg_data = neg_data[cols]
    
    pos_data['result'] = 1
    neg_data['result'] = 0
    
    pos_data['GC'] = pos_data['GC'] * 100
    neg_data['GC'] = neg_data['GC'] * 100
    
    test_data = pd.concat([pos_data, neg_data], ignore_index=True)
    return test_data


# End-to-end training/evaluation for a single chromosome
def train_model(train_data, test_data, chr_num, upscaler=0):
    """Train and evaluate LR for a single chromosome, then persist artifacts."""
    # Split features and target
    features = ['GC', 'FF', f'chr{chr_num}_count_3m']
    X_train = train_data[features]
    y_train = train_data['result']
    X_test = test_data[features]
    y_test = test_data['result']
    
    # Train model and predict
    y_pred, model, scaler = fit_lr_model(
        x_train_data=X_train, 
        y_train_data=y_train, 
        x_test_data=X_test
    )
    
    # Collect results
    results = {
        '정답': list(y_test),
        '예측_LR': list(y_pred)
    }
    
    # Compute ROC metrics and visualize
    metrics_df = calc_metrics(y_test, y_pred, chr_num, upscaler, 'Logistic_Regression', vis_mode='cm')
    
    # Persist model artifacts
    joblib.dump(model, f'./lr_model_T{chr_num}.joblib')
    joblib.dump(scaler, f'./lr_scaler_T{chr_num}.joblib')
    
    return results, metrics_df, model, scaler


# Compute ROC, pick optimal threshold via Youden's J, summarize metrics and CM
def calc_metrics(y_test, y_pred, disease, upscaler, model, vis_mode='cm'):
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Youden's J Statistic으로 최적 threshold 찾기
    youdens_j = tpr - fpr
    optimal_idx = youdens_j.argmax()  # J를 최대화하는 인덱스
    if model != 'Z-Score':
        threshold = thresholds[optimal_idx]
    else:
        threshold = 3

    y_pred = (y_pred >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    TP, FN = cm[1, 1], cm[1, 0]   # True Positive, False Negative
    TN, FP = cm[0, 0], cm[0, 1]

    accuracy_lr = (TP + TN) / (TP + TN + FP + FN)
    sensitivity_lr = TP / (TP + FN)
    specificity_lr = TN / (TN + FP)
    NPV_lr = TN / (TN + FN)
    PPV_lr = TP / (TP + FP)

    result_df = pd.DataFrame({
        'disease': [disease],
        'method': [model],
        'accuracy': [accuracy_lr],
        'sensitivity': [sensitivity_lr],
        'specificity': [specificity_lr],
        'NPV': [NPV_lr],
        'PPV': [PPV_lr]
    })

    if vis_mode == 'cm':
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        if upscaler > 0:
            plot_cm(cm, ax, label=f"{model} CM", title=f"{model}: Trisomy {disease} (x{upscaler})")
        else:
            plot_cm(cm, ax, label=f"{model} CM", title=f"{model}: Trisomy {disease}")
    elif vis_mode == 'all':
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if upscaler > 0:
            plot_cm(cm, axes[0], label=f"{model} CM", title=f"{model}: Trisomy {disease} (x{upscaler})")
        else:
            plot_cm(cm, axes[0], label=f"{model} CM", title=f"{model}: Trisomy {disease}")
    else:
        pass
    return result_df


# Orchestrate the full ACA LR training pipeline across chromosomes
def main():
    """Run the full ACA LR training pipeline across chromosomes."""
    print("Start ACA Logistic Regression training...")
    
    # 1) Load data
    print("1) Loading data...")
    aca_train, aca_test, normal_train, normal_test, chromosomes = load_data()
    
    # 2) Normalize counts to 3M
    print("2) Normalizing counts to 3M...")
    aca_train, aca_test, normal_train, normal_test = normalize_data(
        aca_train, aca_test, normal_train, normal_test, chromosomes
    )
    
    # 3) Prepare training tables
    print("3) Preparing training data...")
    train_data = prep_train_data(aca_train, normal_train, chromosomes)
    
    # 4) Train and evaluate per chromosome
    print("4) Training and evaluating...")
    results_data = {}
    
    for chr_num in chromosomes:
        print(f"\nProcessing Trisomy {chr_num}...")
        
        # Prepare test table
        test_data = prep_test_data(aca_test, normal_test, chr_num)
        
        # Train and evaluate
        results, metrics_df, model, scaler = train_model(
            train_data[chr_num], test_data, chr_num
        )
        
        # Save results snapshot
        results_data[chr_num] = {0: results}
        
        print(f"Trisomy {chr_num} training done!")
        print(metrics_df)
        print("-" * 50)
    
    print("All models trained!")
    return results_data


if __name__ == "__main__":
    results = main()