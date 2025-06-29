import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import Lasso, LogisticRegression
from collections import Counter
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings("ignore")
file_paths = [
     '/Optimized-Ensemble-Learning-AIP/data/external_data/17 new sequences/17_sequences_train+test.xlsx'
]


def process_file(file_path):
    df = pd.read_excel(file_path)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    X_train, X_test = X[:3582], X[3582:3600]
    y_train, y_test = y[:3582], y[3582:3600]
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    X_train_res, y_train_res = under_sampler.fit_resample(X_train_res, y_train_res)
    print(Counter(y_train_res))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_res, y_train_res)
    feature_importance = rf.feature_importances_

    important_features = np.argsort(feature_importance)[::-1][:int(len(feature_importance) * 0.2)]
    X_train_selected = X_train_res[:, important_features]
    X_test_selected = X_test[:, important_features]

    xgb_model = XGBClassifier(
        booster='gbtree',
        objective='binary:logistic',
        eval_metric='logloss',
        learning_rate=0.00156,
        max_depth=10,
        subsample=0.2,
        n_estimators=500,
        colsample_bytree=0.7,
        min_child_weight=20,
        scale_pos_weight =1.5
    )
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        random_state=42,
        max_depth=20,
        max_features='sqrt'
    )
    ab_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=500,
        random_state=42,
        learning_rate=0.01,
    )
    lgbm_model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=1,
        random_state=42,
        class_weight='balanced',
        min_samples_split=10,
        subsample=0.2,
        verbose = -1,
    )
    gbdt_model = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.001,
        max_depth=30,
        random_state=42,
        max_features='sqrt',
        min_samples_split=10,
        subsample=0.2
    )

    #Voting Classifier
    voting_model = VotingClassifier(estimators=[
        ('xgb', xgb_model), ('rf', rf_model),('ab',ab_model),('lgbm',lgbm_model),('gbdt',gbdt_model)
    ], voting='soft')
    voting_model.fit(X_train_selected, y_train_res)

    y_pred_proba = voting_model.predict_proba(X_test_selected)[:, 1]
    #y_pred = voting_model.predict(X_test_selected)

    results_df = pd.DataFrame({
        'Index': df.iloc[y_test.index].index if isinstance(y_test, pd.Series) else np.arange(len(y_test)),
        'True Label': y_test,
        'Predicted Probability': y_pred_proba
    })
    results_df.to_csv(f'prediction_results_{os.path.basename(file_path).split(".")[0]}.csv', index=False)
    print(results_df.head())

for file_path in file_paths:
     process_file(file_path)

