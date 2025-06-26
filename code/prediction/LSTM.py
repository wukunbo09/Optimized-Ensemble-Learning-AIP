import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    matthews_corrcoef,
    average_precision_score,
    roc_curve,
    auc
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import Counter
from tensorflow.keras.layers import LSTM, Reshape

file_paths = [
    '/Optimized Ensemble Learning-AIP/data/external_data/training+test.xlsx'
]


def process_file(file_path):
    df = pd.read_excel(file_path)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    X_train, X_test = X[:3583], X[3584:]
    y_train, y_test = y[:3583], y[3584:]
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    X_train_res, y_train_res = under_sampler.fit_resample(X_train_res, y_train_res)
    print(Counter(y_train_res))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_res, y_train_res)
    feature_importance = rf.feature_importances_

    important_features = np.argsort(feature_importance)[::-1][:int(len(feature_importance) * 0.2)]
    X_train_selected = X_train_res[:, important_features]
    X_test_selected = X_test[:, important_features]

    X_train_lstm = X_train_selected.reshape((X_train_selected.shape[0], X_train_selected.shape[1], 1))
    X_test_lstm = X_test_selected.reshape((X_test_selected.shape[0], X_test_selected.shape[1], 1))

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train_selected.shape[1], 1), return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_lstm, y_train_res, epochs=50, batch_size=32, verbose=0, validation_split=0.2)

    y_pred_proba = model.predict(X_test_lstm).flatten()
    y_pred = (y_pred_proba > 0.4).astype(int)

    TP = ((y_test == 1) & (y_pred == 1)).sum()
    TN = ((y_test == 0) & (y_pred == 0)).sum()
    FP = ((y_test == 0) & (y_pred == 1)).sum()
    FN = ((y_test == 1) & (y_pred == 0)).sum()

    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(f"File: {file_path}")
    print(f"Selected Features: {X_train_selected.shape[1]}")
    print("True Positives (TP):", TP)
    print("False Negatives (FN):", FN)
    print("True Negatives (TN):", TN)
    print("False Positive (FP):", FP)
    print("Sensitivity:%.4f" %sensitivity)
    print("Specificity:%.4f" %specificity)
    print("Precision:%.4f" %precision)
    print("Accuracy:%.4f" %accuracy)
    print("F1:%.4f" %f1)
    print("MCC:%.4f" %mcc)
    print("AP:%.4f" %ap)
    print(f"AUC: {auc_score}")
    print()

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision_recall, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    return fpr, tpr, precision_recall, recall, sensitivity, specificity, precision, accuracy, f1, mcc, ap, auc_score

roc_data = []
prc_data = []
boxplot_data = {
    "Sensitivity": [],
    "Specificity": [],
    "Precision": [],
    "Accuracy": [],
    "F1": [],
    "MCC": [],
    "AP": []
}
labels = []

# 处理每个文件
for file_path in file_paths:
    fpr, tpr, precision_recall, recall, sensitivity, specificity, precision, accuracy, f1, mcc, ap, auc_score= process_file(file_path)
