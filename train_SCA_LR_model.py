"""
SCA Logistic Regression Training Pipeline

This module mirrors the ACA LR pipeline structure for SCA multiclass tasks:
- Load SCA train/test backdata CSVs and align labels (M/F -> XY/XX)
- Build features: GC (scaled x100), UR, snp_FF, chr23_count, chr24_count
- Standardize features using training statistics only
- Train Logistic Regression (multiclass) with class_weight='balanced'
- Evaluate with confusion matrix and macro metrics
- Save model and scaler artifacts
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay


# Standardize features and train Logistic Regression, then predict on test
def fit_lr_model(x_train, y_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    Xtr = pd.DataFrame(scaler.transform(x_train))
    Xte = pd.DataFrame(scaler.transform(x_test))

    model = LogisticRegression(random_state=42, C=1, class_weight='balanced', max_iter=1000)
    model.fit(Xtr, y_train)

    y_pred = model.predict(Xte)
    return y_pred, model, scaler


# Plot confusion matrix (used during evaluation)
def plot_cm(cm, labels, title="Confusion Matrix"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d', ax=ax, colorbar=False)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.show()


# Load train/test SCA backdata and harmonize labels/columns
def load_data():
    cols = ['result', 'GC', 'UR', 'snp_FF', 'chr23_count', 'chr24_count']

    train_df = pd.read_csv('/BiO_3/Paper_Data_Preparation/_SCA_All_Data/SCA_train_backdata.csv', usecols=cols)
    test_df = pd.read_csv('/BiO_3/Paper_Data_Preparation/_SCA_All_Data/SCA_test_backdata.csv', usecols=cols)

    # Align labels to XY/XX for normals
    train_df['result'] = train_df['result'].replace(['M', 'F'], ['XY', 'XX'])
    test_df['result'] = test_df['result'].replace(['M', 'F'], ['XY', 'XX'])

    # Scale GC by 100 per paper convention
    train_df['GC'] = train_df['GC'] * 100
    test_df['GC'] = test_df['GC'] * 100

    return train_df, test_df


# Train and evaluate LR for SCA multiclass classification
def train_model(train_df, test_df):
    features = ['GC', 'UR', 'snp_FF', 'chr23_count', 'chr24_count']
    X_train = train_df[features]
    y_train = train_df['result']
    X_test = test_df[features]
    y_test = test_df['result']

    y_pred, model, scaler = fit_lr_model(X_train, y_train, X_test)

    # Metrics
    sensitivity = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    print('\nEvaluation Metrics')
    print(f'sensitivity: {sensitivity:.4f}')
    print(f'precision  : {precision:.4f}')
    print(f'accuracy   : {accuracy:.4f}')

    # Confusion matrix
    labels = sorted(list(set(y_test) | set(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, labels=labels, digits=4))
    print('\nConfusion Matrix\n')
    print(cm)
    plot_cm(cm, labels, title='Confusion Matrix: SCA Evaluation')

    # Persist model artifacts
    joblib.dump(model, './sca_lr_model.joblib')
    joblib.dump(scaler, './sca_lr_scaler.joblib')
    print('\nSaved artifacts: sca_lr_model.joblib, sca_lr_scaler.joblib')

    return y_pred, model, scaler


def main():
    print('Start SCA Logistic Regression training...')
    print('1) Loading data...')
    train_df, test_df = load_data()

    print('2) Training and evaluating...')
    y_pred, model, scaler = train_model(train_df, test_df)

    print('Done.')
    return y_pred


if __name__ == '__main__':
    _ = main()