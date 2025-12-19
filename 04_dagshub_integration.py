# Install DagsHub
!pip install dagshub mlflow

#  Setup DagsHub
import dagshub
import mlflow
import pandas as pd
from pathlib import Path

# Initialize DagsHub 
DAGSHUB_USERNAME = "karthikaon0412"  
DAGSHUB_REPO = "housing_fall2025"

print(f"Initializing DagsHub...")
print(f"Repository: {DAGSHUB_USERNAME}/{DAGSHUB_REPO}")

dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)

mlflow.set_experiment("housing_price_classification")

print(" DagsHub initialized")
print(f" View at: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}")

# Log All Experiments
models_path = Path('/Users/karthika/housing_app_fall25/api/models')
results_df = pd.read_csv(models_path / 'experiment_results.csv')

print(f"\n{'='*70}")
print(f"Logging {len(results_df)} experiments to DagsHub...")
print(f"{'='*70}\n")

for idx, row in results_df.iterrows():
    experiment_name = row['experiment_name']
    
    print(f"[{idx+1}/{len(results_df)}] Logging: {experiment_name}")
    
    with mlflow.start_run(run_name=experiment_name):
        # Log parameters
        mlflow.log_param("model_type", row['model_type'])
        mlflow.log_param("pca", row['pca'])
        mlflow.log_param("optuna", row['optuna'])
        mlflow.log_param("experiment_name", experiment_name)
        
        # Log metrics
        mlflow.log_metric("f1_score", row['f1_score'])
        mlflow.log_metric("f1_score_percentage", row['f1_score'] * 100)
        
        # Log model file if it exists
        model_path = models_path / f"{experiment_name}.pkl"
        if model_path.exists():
            mlflow.log_artifact(str(model_path), artifact_path="models")
        
        # Log tags
        mlflow.set_tag("model_family", row['model_type'])
        mlflow.set_tag("preprocessing", "PCA" if row['pca'] else "No PCA")
        mlflow.set_tag("tuning", "Optuna" if row['optuna'] else "No Tuning")
        
        print(f"    F1 Score: {row['f1_score']:.4f}")

print(f"\n{'='*70}")
print(" All experiments logged to DagsHub!")
print(f"{'='*70}")
print(f"\n View your experiments at:")
print(f"   https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}")

# Log Best Model Information
best_model = results_df.loc[results_df['f1_score'].idxmax()]

print(f"\n{'='*70}")
print("BEST MODEL SUMMARY")
print(f"{'='*70}")
print(f"Model: {best_model['experiment_name']}")
print(f"Type: {best_model['model_type']}")
print(f"F1 Score: {best_model['f1_score']:.4f}")
print(f"PCA: {best_model['pca']}")
print(f"Optuna: {best_model['optuna']}")
print(f"{'='*70}")

# Log summary
with mlflow.start_run(run_name="BEST_MODEL_SUMMARY"):
    mlflow.log_param("best_model", best_model['experiment_name'])
    mlflow.log_metric("best_f1_score", best_model['f1_score'])
    mlflow.log_metric("average_f1_score", results_df['f1_score'].mean())
    mlflow.log_metric("std_f1_score", results_df['f1_score'].std())
    
    # Log experiment results CSV
    mlflow.log_artifact(str(models_path / 'experiment_results.csv'))

print("\n DagsHub integration complete!")



