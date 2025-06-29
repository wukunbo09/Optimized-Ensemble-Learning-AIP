import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    matthews_corrcoef,
    average_precision_score,
    roc_curve,
    auc,
)
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import Counter


file_paths = [
    '/Optimized-Ensemble-Learning-AIP/data/external_data/training+test.xlsx'
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_res)

    input_dim = X_scaled.shape[1]
    encoding_dim = 300

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer=Adam(), loss='mae')

    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, shuffle=True, validation_data=(X_scaled, X_scaled))

    X_train_encoded = encoder.predict(X_scaled)
    X_test_scaled = scaler.transform(X_test)
    X_test_encoded = encoder.predict(X_test_scaled)

    xgb_model = XGBClassifier(
        booster='gbtree',
        objective='binary:logistic',
        eval_metric='logloss',
        learning_rate=0.01,
        max_depth=10,
        subsample=0.2,
        n_estimators=500,
        colsample_bytree=0.7,
        min_child_weight=20,
        scale_pos_weight=1.5
    )
    xgb_model.fit(X_train_encoded, y_train_res)

    y_pred = xgb_model.predict(X_test_encoded)
    y_pred_proba = xgb_model.predict_proba(X_test_encoded)[:, 1]

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
    print("True Positives (TP):", TP)
    print("False Negatives (FN):", FN)
    print("True Negatives (TN):", TN)
    print("False Positive (FP):", FP)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("F1:", f1)
    print("MCC:", mcc)
    print("AP:", ap)
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

for file_path in file_paths:
    fpr, tpr, precision_recall, recall, sensitivity, specificity, precision, accuracy, f1, mcc,  ap, auc_score= process_file(file_path)
