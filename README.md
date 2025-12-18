# Housing Price Category Classification

**Final Project - Data Science Course**

Multi-class classification predicting housing prices into 4 categories using the Ames Housing Dataset.

## 🎯 Project Overview

Classifies houses into price categories:
- **Class 0 - Low**: ≤$129,975
- **Class 1 - Medium**: $129,976 - $163,000
- **Class 2 - High**: $163,001 - $214,000
- **Class 3 - Very High**: >$214,000

## 📊 Experiments & Results

**16 Total Experiments Conducted:**
- 4 Classification Models: RandomForest, GradientBoosting, XGBoost, LightGBM
- 4 Configurations per model:
  - No PCA + No Hyperparameter Tuning
  - No PCA + Optuna Tuning
  - With PCA + No Tuning
  - With PCA + Optuna Tuning

**Best Model:** RandomForest (No PCA, Optuna) - **F1 Score: 0.8203**

All experiment results tracked in DagsHub with complete metrics.

## 🛠️ Technology Stack

- **ML/Data**: Scikit-learn, XGBoost, LightGBM, Pandas, NumPy
- **Optimization**: Optuna (hyperparameter tuning)
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Database**: SQLite
- **Deployment**: Docker, Docker Compose, DigitalOcean
- **Tracking**: DagsHub, MLflow

## 🚀 Quick Start

### Local Deployment
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/housing_classification.git
cd housing_classification

# Run with Docker
docker compose up -d

# Access services
# API: http://localhost:8000
# Streamlit UI: http://localhost:8501
```

### API Usage
```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model-info

# View all experiments
curl http://localhost:8000/experiments
```

## 📁 Project Structure
```
housing_classification/
├── api/
│   ├── app.py                 # FastAPI backend
│   ├── Dockerfile
│   ├── requirements.txt
│   └── models/
│       ├── *.pkl              # 16 trained models
│       └── experiment_results.csv
├── streamlit/
│   ├── app.py                 # Streamlit frontend
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   ├── 01_create_database.ipynb
│   ├── 02_train_model_without_optuna.ipynb
│   ├── 03_train_models_with_optuna.ipynb
│   └── 04_dagshub_integration.ipynb
├── data/
│   ├── train.csv              # Ames Housing Dataset
│   └── data_schema.json
├── db/
│   └── housing_classification.db
└── docker-compose.yml
```

## 🌐 Live Deployment

- **API**: http://YOUR_DROPLET_IP:8000
- **UI**: http://YOUR_DROPLET_IP:8501
- **API Docs**: http://YOUR_DROPLET_IP:8000/docs
- **DagsHub**: https://dagshub.com/YOUR_USERNAME/housing_classification

## 📈 Model Performance

| Model | Configuration | PCA | Optuna | F1 Score |
|-------|--------------|-----|--------|----------|
| RandomForest | No PCA + Optuna | ❌ | ✅ | **0.8203** |
| GradientBoosting | No PCA + Optuna | ❌ | ✅ | 0.8156 |
| XGBoost | No PCA + Optuna | ❌ | ✅ | 0.8145 |
| LightGBM | No PCA + Optuna | ❌ | ✅ | 0.8134 |

*Full results available in `api/models/experiment_results.csv`*

## 🗄️ Database Schema

SQLite database with normalized schema:
- **Table**: `housing_data`
- **Rows**: 2,930 housing records
- **Features**: 80+ attributes
- **Target**: `PriceCategory` (0-3)

## 🔬 Experiment Tracking

All 16 experiments logged to DagsHub including:
- Model type and configuration
- Hyperparameters (for Optuna runs)
- F1 scores (weighted average)
- Trained model artifacts
- PCA transformation status

## 🐳 Docker Deployment

### Services
- **API**: FastAPI on port 8000
- **Streamlit**: Frontend on port 8501

### Commands
```bash
# Build and start
docker compose up -d --build

# View logs
docker compose logs -f

# Stop services
docker compose down

# Check status
docker compose ps
```

## 📊 API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /model-info` - Model information
- `GET /experiments` - All experiment results
- `POST /predict` - Make prediction

## 🎓 Academic Context

**Course**: Data Science Final Project  
**Institution**: [Your University]  
**Semester**: Fall 2024  
**Requirements Met**:
- ✅ Classification problem (4 classes)
- ✅ Normalized database schema
- ✅ 16 experiments with proper tracking
- ✅ DagsHub integration
- ✅ FastAPI + Streamlit deployment
- ✅ Docker containerization
- ✅ Cloud deployment (DigitalOcean)
