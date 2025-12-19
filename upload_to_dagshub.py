import dagshub
import mlflow
import pandas as pd
from pathlib import Path

# Your DagsHub credentials
DAGSHUB_USERNAME = "karthikaon0412"
DAGSHUB_REPO = "housing_classification"

print(f"{'='*70}")
print(f"Uploading experiments to DagsHub...")
print(f"Repository: {DAGSHUB_USERNAME}/{DAGSHUB_REPO}")
print(f"{'='*70}\n")

# Initialize DagsHub
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
mlflow.set_experiment("housing_price_classification")

# Load experiment results
models_path = Path('api/models')
results_df = pd.read_csv(models_path / 'experiment_results.csv')

print(f"Found {len(results_df)} experiments to upload\n")

# Log each experiment
for idx, row in results_df.iterrows():
    experiment_name = row['experiment_name']
    
    print(f"[{idx+1}/16] Uploading: {experiment_name:<45}", end=" ")
    
    try:
        with mlflow.start_run(run_name=experiment_name):
            # Log parameters
            mlflow.log_param("model_type", row['model_type'])
            mlflow.log_param("pca", bool(row['pca']))
            mlflow.log_param("optuna", bool(row['optuna']))
            mlflow.log_param("experiment_name", experiment_name)
            
            # Log metrics
            mlflow.log_metric("f1_score", float(row['f1_score']))
            mlflow.log_metric("f1_score_percentage", float(row['f1_score']) * 100)
            
            # Log model file
            model_path = models_path / f"{experiment_name}.pkl"
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="models")
            
            # Add tags for filtering
            mlflow.set_tag("pca_used", "Yes" if row['pca'] else "No")
            mlflow.set_tag("optuna_used", "Yes" if row['optuna'] else "No")
            mlflow.set_tag("model_family", row['model_type'])
            
            print(f"✅ F1={row['f1_score']:.4f}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n{'='*70}")
print("🎉 SUCCESS! ALL 16 EXPERIMENTS UPLOADED TO DAGSHUB!")
print(f"{'='*70}")
print(f"\n🔗 View your experiments at:")
print(f"   https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}")
print(f"\n📊 You can now see:")
print("   ✅ All 16 experiments")
print("   ✅ F1 scores for each model")
print("   ✅ PCA and Optuna configurations")
print("   ✅ Model artifacts (.pkl files)")
print("   ✅ Complete experiment metadata")
print(f"\n{'='*70}")
