# Imports
import pandas as pd
import numpy as np
import sqlite3
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
import pickle
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("All libraries imported successfully")

#  Load Data and Preprocessing
print("\n" + "="*70)
print("LOADING DATA AND PREPROCESSING")
print("="*70)

db_path = '/Users/karthika/housing_app_fall25/db/housing_classification.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql("SELECT * FROM housing_data", conn)
conn.close()

X = df.drop('PriceCategory', axis=1)
y = df['PriceCategory']

models_path = Path('/Users/karthika/housing_app_fall25/api/models')

# Load encoders
with open(models_path / 'label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = label_encoders[col].transform(X[col].astype(str))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
with open(models_path / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load PCA
with open(models_path / 'pca.pkl', 'rb') as f:
    pca = pickle.load(f)

X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f" Data loaded and preprocessed")
print(f"   Train shape: {X_train_scaled.shape}")
print(f"   Test shape: {X_test_scaled.shape}")
print(f"   PCA shape: {X_train_pca.shape}")

results = []

#  RandomForest - No PCA + Optuna
print("\n" + "="*70)
print("EXPERIMENT 1/8: RandomForest + No PCA + Optuna")
print("="*70)

def objective_rf_nopca(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train_scaled, y_train, cv=3, 
                           scoring='f1_weighted', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective_rf_nopca, n_trials=20, show_progress_bar=True)

print(f" Best CV F1 Score: {study.best_value:.4f}")

best_rf = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_rf.fit(X_train_scaled, y_train)
y_pred = best_rf.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred, average='weighted')

with open(models_path / 'RandomForest_NoPCA_Optuna.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

results.append({
    'experiment_name': 'RandomForest_NoPCA_Optuna',
    'model_type': 'RandomForest',
    'pca': False,
    'optuna': True,
    'f1_score': f1
})

print(f" Test F1 Score: {f1:.4f}\n")


# GradientBoosting - No PCA + Optuna
print("\n" + "="*70)
print("EXPERIMENT 2/8: GradientBoosting + No PCA + Optuna")
print("="*70)

def objective_gb_nopca(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'random_state': 42
    }
    model = GradientBoostingClassifier(**params)
    score = cross_val_score(model, X_train_scaled, y_train, cv=3,
                           scoring='f1_weighted', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective_gb_nopca, n_trials=20, show_progress_bar=True)

print(f" Best CV F1 Score: {study.best_value:.4f}")

best_gb = GradientBoostingClassifier(**study.best_params, random_state=42)
best_gb.fit(X_train_scaled, y_train)
y_pred = best_gb.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred, average='weighted')

with open(models_path / 'GradientBoosting_NoPCA_Optuna.pkl', 'wb') as f:
    pickle.dump(best_gb, f)

results.append({
    'experiment_name': 'GradientBoosting_NoPCA_Optuna',
    'model_type': 'GradientBoosting',
    'pca': False,
    'optuna': True,
    'f1_score': f1
})

print(f" Test F1 Score: {f1:.4f}\n")

# CELL 5: XGBoost - No PCA + Optuna
print("\n" + "="*70)
print("EXPERIMENT 3/8: XGBoost + No PCA + Optuna")
print("="*70)

def objective_xgb_nopca(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'n_jobs': -1
    }
    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train_scaled, y_train, cv=3,
                           scoring='f1_weighted', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective_xgb_nopca, n_trials=20, show_progress_bar=True)

print(f" Best CV F1 Score: {study.best_value:.4f}")

best_xgb = XGBClassifier(**study.best_params, random_state=42, eval_metric='mlogloss', 
                         use_label_encoder=False, n_jobs=-1)
best_xgb.fit(X_train_scaled, y_train)
y_pred = best_xgb.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred, average='weighted')

with open(models_path / 'XGBoost_NoPCA_Optuna.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

results.append({
    'experiment_name': 'XGBoost_NoPCA_Optuna',
    'model_type': 'XGBoost',
    'pca': False,
    'optuna': True,
    'f1_score': f1
})

print(f" Test F1 Score: {f1:.4f}\n")


# LightGBM - No PCA + Optuna
print("\n" + "="*70)
print("EXPERIMENT 4/8: LightGBM + No PCA + Optuna")
print("="*70)

def objective_lgb_nopca(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    model = LGBMClassifier(**params)
    score = cross_val_score(model, X_train_scaled, y_train, cv=3,
                           scoring='f1_weighted', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective_lgb_nopca, n_trials=20, show_progress_bar=True)

print(f" Best CV F1 Score: {study.best_value:.4f}")

best_lgb = LGBMClassifier(**study.best_params, random_state=42, verbose=-1, n_jobs=-1)
best_lgb.fit(X_train_scaled, y_train)
y_pred = best_lgb.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred, average='weighted')

with open(models_path / 'LightGBM_NoPCA_Optuna.pkl', 'wb') as f:
    pickle.dump(best_lgb, f)

results.append({
    'experiment_name': 'LightGBM_NoPCA_Optuna',
    'model_type': 'LightGBM',
    'pca': False,
    'optuna': True,
    'f1_score': f1
})

print(f" Test F1 Score: {f1:.4f}\n")


# RandomForest - PCA + Optuna
print("\n" + "="*70)
print("EXPERIMENT 5/8: RandomForest + PCA + Optuna")
print("="*70)

def objective_rf_pca(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train_pca, y_train, cv=3,
                           scoring='f1_weighted', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective_rf_pca, n_trials=20, show_progress_bar=True)

print(f" Best CV F1 Score: {study.best_value:.4f}")

best_rf_pca = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_rf_pca.fit(X_train_pca, y_train)
y_pred = best_rf_pca.predict(X_test_pca)
f1 = f1_score(y_test, y_pred, average='weighted')

with open(models_path / 'RandomForest_PCA_Optuna.pkl', 'wb') as f:
    pickle.dump(best_rf_pca, f)

results.append({
    'experiment_name': 'RandomForest_PCA_Optuna',
    'model_type': 'RandomForest',
    'pca': True,
    'optuna': True,
    'f1_score': f1
})

print(f" Test F1 Score: {f1:.4f}\n")


# GradientBoosting - PCA + Optuna
print("\n" + "="*70)
print("EXPERIMENT 6/8: GradientBoosting + PCA + Optuna")
print("="*70)

def objective_gb_pca(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'random_state': 42
    }
    model = GradientBoostingClassifier(**params)
    score = cross_val_score(model, X_train_pca, y_train, cv=3,
                           scoring='f1_weighted', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective_gb_pca, n_trials=20, show_progress_bar=True)

print(f" Best CV F1 Score: {study.best_value:.4f}")

best_gb_pca = GradientBoostingClassifier(**study.best_params, random_state=42)
best_gb_pca.fit(X_train_pca, y_train)
y_pred = best_gb_pca.predict(X_test_pca)
f1 = f1_score(y_test, y_pred, average='weighted')

with open(models_path / 'GradientBoosting_PCA_Optuna.pkl', 'wb') as f:
    pickle.dump(best_gb_pca, f)

results.append({
    'experiment_name': 'GradientBoosting_PCA_Optuna',
    'model_type': 'GradientBoosting',
    'pca': True,
    'optuna': True,
    'f1_score': f1
})

print(f" Test F1 Score: {f1:.4f}\n")


# XGBoost - PCA + Optuna
print("\n" + "="*70)
print("EXPERIMENT 7/8: XGBoost + PCA + Optuna")
print("="*70)

def objective_xgb_pca(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'n_jobs': -1
    }
    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train_pca, y_train, cv=3,
                           scoring='f1_weighted', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective_xgb_pca, n_trials=20, show_progress_bar=True)

print(f" Best CV F1 Score: {study.best_value:.4f}")

best_xgb_pca = XGBClassifier(**study.best_params, random_state=42, eval_metric='mlogloss',
                             use_label_encoder=False, n_jobs=-1)
best_xgb_pca.fit(X_train_pca, y_train)
y_pred = best_xgb_pca.predict(X_test_pca)
f1 = f1_score(y_test, y_pred, average='weighted')

with open(models_path / 'XGBoost_PCA_Optuna.pkl', 'wb') as f:
    pickle.dump(best_xgb_pca, f)

results.append({
    'experiment_name': 'XGBoost_PCA_Optuna',
    'model_type': 'XGBoost',
    'pca': True,
    'optuna': True,
    'f1_score': f1
})

print(f" Test F1 Score: {f1:.4f}\n")

#  LightGBM - PCA + Optuna
print("\n" + "="*70)
print("EXPERIMENT 8/8: LightGBM + PCA + Optuna")
print("="*70)

def objective_lgb_pca(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    model = LGBMClassifier(**params)
    score = cross_val_score(model, X_train_pca, y_train, cv=3,
                           scoring='f1_weighted', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective_lgb_pca, n_trials=20, show_progress_bar=True)

print(f" Best CV F1 Score: {study.best_value:.4f}")

best_lgb_pca = LGBMClassifier(**study.best_params, random_state=42, verbose=-1, n_jobs=-1)
best_lgb_pca.fit(X_train_pca, y_train)
y_pred = best_lgb_pca.predict(X_test_pca)
f1 = f1_score(y_test, y_pred, average='weighted')

with open(models_path / 'LightGBM_PCA_Optuna.pkl', 'wb') as f:
    pickle.dump(best_lgb_pca, f)

results.append({
    'experiment_name': 'LightGBM_PCA_Optuna',
    'model_type': 'LightGBM',
    'pca': True,
    'optuna': True,
    'f1_score': f1
})

print(f" Test F1 Score: {f1:.4f}\n")


# Combine and Save All Results
print("\n" + "="*70)
print("COMBINING ALL RESULTS")
print("="*70)

results_df_part2 = pd.DataFrame(results)

# Load part 1 results
results_part1 = pd.read_csv(models_path / 'experiment_results_part1.csv')

# Combine all results
all_results = pd.concat([results_part1, results_df_part2], ignore_index=True)
all_results = all_results.sort_values('f1_score', ascending=False).reset_index(drop=True)
all_results.to_csv(models_path / 'experiment_results.csv', index=False)

print("\n" + "="*70)
print(" ALL 16 EXPERIMENTS COMPLETED!")
print("="*70)
print(all_results.to_string(index=False))
print(f"\n Final results saved to: {models_path / 'experiment_results.csv'}")
print(f"\n BEST MODEL: {all_results.loc[0, 'experiment_name']}")
print(f"   F1 Score: {all_results.loc[0, 'f1_score']:.4f}")
print(f"   Model Type: {all_results.loc[0, 'model_type']}")
print(f"   PCA: {all_results.loc[0, 'pca']}")
print(f"   Optuna: {all_results.loc[0, 'optuna']}")

print("\n" + "="*70)
print(" MODEL TRAINING COMPLETE!")
print("="*70)



