from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import json

app = FastAPI(
    title="Housing Price Category Prediction API",
    description="Predicts housing price categories using machine learning",
    version="1.0.0"
)

# Load models and preprocessing objects
models_path = Path("/app/models")

class ModelLoader:
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.label_encoders = None
        self.feature_names = None
        self.best_model = None
        self.experiment_results = None
        self.best_model_info = None
        
    def load_all(self):
        try:
            print("Loading preprocessing objects...")
            
            with open(models_path / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(models_path / 'pca.pkl', 'rb') as f:
                self.pca = pickle.load(f)
            
            with open(models_path / 'label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            with open(models_path / 'feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            self.experiment_results = pd.read_csv(models_path / 'experiment_results.csv')
            
            best_experiment = self.experiment_results.loc[
                self.experiment_results['f1_score'].idxmax()
            ]
            
            best_model_name = best_experiment['experiment_name'] + '.pkl'
            
            print(f"Loading best model: {best_model_name}")
            with open(models_path / best_model_name, 'rb') as f:
                self.best_model = pickle.load(f)
            
            self.best_model_info = {
                'name': best_experiment['experiment_name'],
                'type': best_experiment['model_type'],
                'pca': bool(best_experiment['pca']),
                'optuna': bool(best_experiment['optuna']),
                'f1_score': float(best_experiment['f1_score'])
            }
            
            print(f"✅ Loaded best model: {self.best_model_info['name']}")
            print(f"   F1 Score: {self.best_model_info['f1_score']:.4f}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

loader = ModelLoader()

@app.on_event("startup")
async def startup_event():
    print("="*70)
    print("Starting Housing Price Category Prediction API...")
    print("="*70)
    loader.load_all()
    print("="*70)
    print("✅ API Ready!")
    print("="*70)

class PredictionInput(BaseModel):
    features: Dict[str, Any]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": {
                    "MS SubClass": 60,
                    "MS Zoning": "RL",
                    "Lot Area": 8450,
                    "Overall Qual": 7,
                    "Year Built": 2003
                }
            }
        }
    )

class PredictionOutput(BaseModel):
    predicted_category: int
    category_label: str
    probability: List[float]
    confidence: float
    model_info: Dict[str, Any]

@app.get("/")
async def root():
    return {
        "message": "Housing Price Category Prediction API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "experiments": "/experiments"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": loader.best_model is not None,
        "scaler_loaded": loader.scaler is not None,
        "encoders_loaded": loader.label_encoders is not None
    }

@app.get("/model-info")
async def model_info():
    if loader.best_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "best_model": loader.best_model_info,
        "categories": {
            "0": "Low Price Range",
            "1": "Medium Price Range",
            "2": "High Price Range",
            "3": "Very High Price Range"
        },
        "total_features": len(loader.feature_names)
    }

@app.get("/experiments")
async def get_experiments():
    if loader.experiment_results is None:
        raise HTTPException(status_code=503, detail="Experiment results not loaded")
    
    return {
        "total_experiments": len(loader.experiment_results),
        "experiments": loader.experiment_results.to_dict(orient='records'),
        "best_model": loader.best_model_info
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        if loader.best_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        input_dict = input_data.features
        df = pd.DataFrame([input_dict])
        
        missing_cols = set(loader.feature_names) - set(df.columns)
        if missing_cols:
            for col in missing_cols:
                if col in loader.label_encoders:
                    df[col] = 'Missing'
                else:
                    df[col] = 0
        
        df = df[loader.feature_names]
        
        for col, encoder in loader.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    df[col] = encoder.transform(['Missing'])
        
        X_scaled = loader.scaler.transform(df)
        
        if loader.best_model_info['pca']:
            X_scaled = loader.pca.transform(X_scaled)
        
        prediction = loader.best_model.predict(X_scaled)[0]
        
        if hasattr(loader.best_model, 'predict_proba'):
            probabilities = loader.best_model.predict_proba(X_scaled)[0].tolist()
            confidence = float(max(probabilities))
        else:
            probabilities = [1.0 if i == prediction else 0.0 for i in range(4)]
            confidence = 1.0
        
        category_labels = {
            0: "Low Price Range",
            1: "Medium Price Range",
            2: "High Price Range",
            3: "Very High Price Range"
        }
        
        return PredictionOutput(
            predicted_category=int(prediction),
            category_label=category_labels[int(prediction)],
            probability=probabilities,
            confidence=confidence,
            model_info=loader.best_model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
