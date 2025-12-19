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
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully")

#Load Data from Database
print("\n" + "="*70)
print("LOADING DATA FROM DATABASE")
print("="*70)

db_path = '/Users/karthika/housing_app_fall25/db/housing_classification.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql("SELECT * FROM housing_data", conn)
conn.close()

print(f"📊 Data loaded: {df.shape}")
print(f"\n🎯 Target distribution:")
dist = df['PriceCategory'].value_counts().sort_index()
for idx, count in dist.items():
    print(f"   Class {idx}: {count} samples ({count/len(df)*100:.1f}%)")


# Preprocessing
print("\n" + "="*70)
print("PREPROCESSING DATA")
print("="*70)

# Separate features and target
X = df.drop('PriceCategory', axis=1)
y = df['PriceCategory']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}

print(f"\n Encoding {len(categorical_cols)} categorical columns...")
for i, col in enumerate(categorical_cols, 1):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    if i % 10 == 0 or i == len(categorical_cols):
        print(f"   Encoded {i}/{len(categorical_cols)} columns")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Data Split:")
print(f"   Train set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")


# Scaling and Save Preprocessing Objects
print("\n" + "="*70)
print("SCALING FEATURES")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f" Features scaled")
print(f"   Train shape: {X_train_scaled.shape}")
print(f"   Test shape: {X_test_scaled.shape}")

# Save scaler and encoders
models_path = Path('/Users/karthika/housing_app_fall25/api/models')
models_path.mkdir(exist_ok=True, parents=True)

with open(models_path / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(models_path / 'label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Save feature names for later use
feature_names = X.columns.tolist()
with open(models_path / 'feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print(" Scaler, encoders, and feature names saved")


# Train Models - No PCA, No Optuna (4 experiments)
print("\n" + "="*70)
print("EXPERIMENT SET 1: No PCA + No Optuna (4 models)")
print("="*70)

results = []

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', n_jobs=-1, use_label_encoder=False),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
}

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"🔄 Training {name}...")
    print(f"{'='*70}")
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Save model
    model_filename = f"{name}_NoPCA_NoOptuna.pkl"
    with open(models_path / model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    results.append({
        'experiment_name': f"{name}_NoPCA_NoOptuna",
        'model_type': name,
        'pca': False,
        'optuna': False,
        'f1_score': f1
    })
    
    print(f" {name} - F1 Score: {f1:.4f}")
    print(f"\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High', 'Very High']))


# Train Models - With PCA, No Optuna (4 experiments)
print("\n" + "="*70)
print("EXPERIMENT SET 2: With PCA + No Optuna (4 models)")
print("="*70)

# Apply PCA
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"📉 PCA Transformation:")
print(f"   Original features: {X_train_scaled.shape[1]}")
print(f"   Reduced features: {X_train_pca.shape[1]}")
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.2%}\n")

# Save PCA
with open(models_path / 'pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

for name in models.keys():
    print(f"\n{'='*70}")
    print(f" Training {name} with PCA...")
    print(f"{'='*70}")
    
    # Create new model instance
    if name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif name == 'GradientBoosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif name == 'XGBoost':
        model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', n_jobs=-1, use_label_encoder=False)
    else:  # LightGBM
        model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Save model
    model_filename = f"{name}_PCA_NoOptuna.pkl"
    with open(models_path / model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    results.append({
        'experiment_name': f"{name}_PCA_NoOptuna",
        'model_type': name,
        'pca': True,
        'optuna': False,
        'f1_score': f1
    })
    
    print(f" {name} with PCA - F1 Score: {f1:.4f}")


# Save Results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
results_df.to_csv(models_path / 'experiment_results_part1.csv', index=False)

print(f" Results saved to: {models_path / 'experiment_results_part1.csv'}")

print("\n" + "="*70)
print(" SUMMARY - 8 Experiments Completed")
print("="*70)
print(results_df.to_string(index=False))
print(f"\n🏆 Best Model So Far: {results_df.loc[results_df['f1_score'].idxmax(), 'experiment_name']}")
print(f"   F1 Score: {results_df['f1_score'].max():.4f}")

print("\n" + "="*70)
print(" TRAINING WITHOUT OPTUNA COMPLETE!")
print("="*70)


