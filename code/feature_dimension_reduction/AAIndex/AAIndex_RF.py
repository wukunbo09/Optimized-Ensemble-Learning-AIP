import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

file_paths = [
    '/Optimized-Ensemble-Learning-AIP/data/external_data/feature_dimension_reduction/AAIndex_original.csv'
]

def process_file(file_path):
    df = pd.read_csv(file_path)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    #under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    X_train, X_test = X[:3583], X[3584:]
    y_train, y_test = y[:3583], y[3584:]
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    #X_train_res, y_train_res = under_sampler.fit_resample(X_train_res, y_train_res)


    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_res, y_train_res)

    feature_importance = rf.feature_importances_
    important_features = np.argsort(feature_importance)[::-1][:int(len(feature_importance) * 0.1)]
    X_train_selected = X_train_res[:, important_features]
    X_test_selected = X_test[:, important_features]

    column_names = df.columns.tolist()
    selected_column_names = [column_names[i + 1] for i in important_features]
    selected_column_names = [column_names[0]] + selected_column_names

    df = df.apply(pd.to_numeric, errors='coerce')

    selected_df = df[selected_column_names]
    #selected_df.to_excel('AAIndex_RF.xlsx', index=False, engine='openpyxl')

    return X_train_selected, X_test_selected

for file_path in file_paths:
    process_file(file_path)
