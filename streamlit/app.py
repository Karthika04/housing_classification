import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# API endpoint
API_URL = "http://api:8000"

st.set_page_config(
    page_title="Housing Price Category Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏠 Housing Price Category Predictor</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Model Information
st.sidebar.title("📊 Model Dashboard")

try:
    model_info_response = requests.get(f"{API_URL}/model-info", timeout=5)
    if model_info_response.status_code == 200:
        model_info = model_info_response.json()
        st.sidebar.success("✅ API Connected")
        st.sidebar.metric("Best Model", model_info['best_model']['type'])
        st.sidebar.metric("F1 Score", f"{model_info['best_model']['f1_score']:.4f}")
        
        with st.sidebar.expander("📋 Model Details"):
            st.write(f"**Full Name:** {model_info['best_model']['name']}")
            st.write(f"**PCA:** {'Yes' if model_info['best_model']['pca'] else 'No'}")
            st.write(f"**Hyperparameter Tuning:** {'Yes' if model_info['best_model']['optuna'] else 'No'}")
            st.write(f"**Features:** {model_info['total_features']}")
    else:
        st.sidebar.error("❌ API Not Connected")
except Exception as e:
    st.sidebar.warning("⚠️ Unable to reach API")
    st.sidebar.text(f"Error: {str(e)}")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Make Prediction", "📈 View Experiments", "📊 Model Comparison", "ℹ️ About"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    st.header("Enter Property Details for Prediction")
    
    st.info("💡 **Tip:** Fill in as many details as possible for accurate predictions. Default values are provided for quick testing.")
    
    # Simple form for basic features
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ms_subclass = st.number_input("MS SubClass", value=60, min_value=20, max_value=190)
            ms_zoning = st.selectbox("MS Zoning", ["RL", "RM", "C (all)", "FV", "RH"])
            lot_area = st.number_input("Lot Area (sq ft)", value=10000, min_value=1000, max_value=100000)
            overall_qual = st.slider("Overall Quality", 1, 10, 7)
            overall_cond = st.slider("Overall Condition", 1, 10, 5)
        
        with col2:
            year_built = st.number_input("Year Built", value=2000, min_value=1872, max_value=2024)
            gr_liv_area = st.number_input("Above Ground Living Area", value=1500, min_value=300, max_value=6000)
            total_bsmt_sf = st.number_input("Total Basement SF", value=1000, min_value=0, max_value=6000)
            garage_cars = st.number_input("Garage Cars", value=2, min_value=0, max_value=5)
            garage_area = st.number_input("Garage Area", value=500, min_value=0, max_value=1500)
        
        with col3:
            full_bath = st.number_input("Full Bathrooms", value=2, min_value=0, max_value=4)
            bedroom_abvgr = st.number_input("Bedrooms", value=3, min_value=0, max_value=10)
            kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"])
            fireplaces = st.number_input("Fireplaces", value=1, min_value=0, max_value=4)
            neighborhood = st.selectbox("Neighborhood", 
                ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt", "Other"])
        
        submit_button = st.form_submit_button("🔮 Predict Price Category", use_container_width=True)
    
    if submit_button:
        with st.spinner("🔄 Making prediction..."):
            try:
                # Prepare minimal feature set
                input_data = {
                    "features": {
                        "MS SubClass": int(ms_subclass),
                        "MS Zoning": ms_zoning,
                        "Lot Area": int(lot_area),
                        "Overall Qual": int(overall_qual),
                        "Overall Cond": int(overall_cond),
                        "Year Built": int(year_built),
                        "Gr Liv Area": int(gr_liv_area),
                        "Total Bsmt SF": float(total_bsmt_sf),
                        "Garage Cars": int(garage_cars),
                        "Garage Area": int(garage_area),
                        "Full Bath": int(full_bath),
                        "Bedroom AbvGr": int(bedroom_abvgr),
                        "Kitchen Qual": kitchen_qual,
                        "Fireplaces": int(fireplaces),
                        "Neighborhood": neighborhood
                    }
                }
                
                # Make API call
                response = requests.post(f"{API_URL}/predict", json=input_data, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Prediction Successful!")
                    
                    # Display results in cards
                    col_r1, col_r2, col_r3 = st.columns(3)
                    
                    with col_r1:
                        st.metric(
                            label="🏷️ Predicted Category",
                            value=f"Class {result['predicted_category']}",
                            delta=result['category_label']
                        )
                    
                    with col_r2:
                        st.metric(
                            label="📊 Confidence",
                            value=f"{result['confidence']:.1%}"
                        )
                    
                    with col_r3:
                        st.metric(
                            label="🤖 Model Used",
                            value=result['model_info']['type']
                        )
                    
                    # Probability visualization
                    st.subheader("📊 Prediction Probabilities")
                    
                    categories = ["Low", "Medium", "High", "Very High"]
                    probs = result['probability']
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=categories,
                            y=probs,
                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                            text=[f'{p:.1%}' for p in probs],
                            textposition='auto',
                            hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Probability Distribution Across Categories",
                        xaxis_title="Price Category",
                        yaxis_title="Probability",
                        yaxis=dict(tickformat='.0%'),
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional info
                    with st.expander("🔍 View Detailed Results"):
                        st.json(result)
                    
                else:
                    st.error(f"❌ Prediction failed: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ API Connection Error: {str(e)}")
                st.info("💡 Make sure the API is running with `docker-compose up`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# TAB 2: EXPERIMENTS
# ============================================================================
with tab2:
    st.header("📈 Model Experiments Dashboard")
    
    try:
        exp_response = requests.get(f"{API_URL}/experiments", timeout=5)
        
        if exp_response.status_code == 200:
            exp_data = exp_response.json()
            experiments_df = pd.DataFrame(exp_data['experiments'])
            
            st.success(f"✅ Loaded {exp_data['total_experiments']} experiments")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Experiments", len(experiments_df))
            with col2:
                st.metric("🏆 Best F1 Score", f"{experiments_df['f1_score'].max():.4f}")
            with col3:
                st.metric("📈 Average F1 Score", f"{experiments_df['f1_score'].mean():.4f}")
            with col4:
                st.metric("📉 Worst F1 Score", f"{experiments_df['f1_score'].min():.4f}")
            
            st.markdown("---")
            
            # Visualizations
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                st.subheader("🎯 F1 Scores by Model Type")
                fig1 = px.box(
                    experiments_df,
                    x='model_type',
                    y='f1_score',
                    color='model_type',
                    title="Model Performance Distribution",
                    labels={'f1_score': 'F1 Score', 'model_type': 'Model Type'}
                )
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_v2:
                st.subheader("⚙️ Impact of PCA and Optuna")
                
                # Create comparison
                experiments_df['config'] = experiments_df.apply(
                    lambda row: f"{'PCA' if row['pca'] else 'No PCA'}\n{'Optuna' if row['optuna'] else 'No Optuna'}",
                    axis=1
                )
                
                config_avg = experiments_df.groupby('config')['f1_score'].mean().reset_index()
                
                fig2 = px.bar(
                    config_avg,
                    x='config',
                    y='f1_score',
                    title="Average F1 Score by Configuration",
                    labels={'f1_score': 'Average F1 Score', 'config': 'Configuration'},
                    color='f1_score',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Detailed comparison
            st.subheader("📊 All Experiments Comparison")
            
            fig3 = px.bar(
                experiments_df.sort_values('f1_score', ascending=True),
                y='experiment_name',
                x='f1_score',
                color='model_type',
                title="F1 Scores - All Experiments",
                labels={'f1_score': 'F1 Score', 'experiment_name': 'Experiment'},
                orientation='h',
                height=600
            )
            fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig3, use_container_width=True)
            
            # Detailed table
            st.subheader("📋 Detailed Results Table")
            
            # Format the dataframe
            display_df = experiments_df.copy()
            display_df['f1_score'] = display_df['f1_score'].apply(lambda x: f"{x:.4f}")
            display_df['pca'] = display_df['pca'].apply(lambda x: "✓" if x else "✗")
            display_df['optuna'] = display_df['optuna'].apply(lambda x: "✓" if x else "✗")
            display_df = display_df.sort_values('f1_score', ascending=False)
            
            st.dataframe(
                display_df[['experiment_name', 'model_type', 'pca', 'optuna', 'f1_score']],
                use_container_width=True,
                hide_index=True
            )
            
        else:
            st.error("❌ Failed to load experiments")
            
    except Exception as e:
        st.error(f"❌ Error loading experiments: {str(e)}")

# ============================================================================
# TAB 3: MODEL COMPARISON
# ============================================================================
with tab3:
    st.header("📊 Model Performance Comparison")
    
    try:
        exp_response = requests.get(f"{API_URL}/experiments", timeout=5)
        
        if exp_response.status_code == 200:
            exp_data = exp_response.json()
            experiments_df = pd.DataFrame(exp_data['experiments'])
            
            # Heatmap
            st.subheader("🔥 Performance Heatmap")
            
            # Create configuration labels
            experiments_df['pca_label'] = experiments_df['pca'].apply(lambda x: 'With PCA' if x else 'No PCA')
            experiments_df['optuna_label'] = experiments_df['optuna'].apply(lambda x: 'With Optuna' if x else 'No Optuna')
            experiments_df['config_full'] = experiments_df['pca_label'] + ' + ' + experiments_df['optuna_label']
            
            pivot_data = experiments_df.pivot_table(
                values='f1_score',
                index='model_type',
                columns='config_full',
                aggfunc='first'
            )
            
            fig_heat = px.imshow(
                pivot_data,
                labels=dict(x="Configuration", y="Model Type", color="F1 Score"),
                title="F1 Score Heatmap: Model × Configuration",
                color_continuous_scale='RdYlGn',
                aspect="auto",
                text_auto='.4f'
            )
            fig_heat.update_xaxes(side="bottom")
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # Statistical comparison
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.subheader("📈 PCA Impact Analysis")
                pca_comparison = experiments_df.groupby('pca')['f1_score'].agg(['mean', 'std', 'min', 'max']).round(4)
                pca_comparison.index = ['Without PCA', 'With PCA']
                st.dataframe(pca_comparison, use_container_width=True)
                
                # Calculate improvement
                improvement = ((pca_comparison.loc['With PCA', 'mean'] - pca_comparison.loc['Without PCA', 'mean']) / 
                              pca_comparison.loc['Without PCA', 'mean'] * 100)
                
                if improvement > 0:
                    st.success(f"✅ PCA improves performance by {improvement:.2f}% on average")
                else:
                    st.warning(f"⚠️ PCA decreases performance by {abs(improvement):.2f}% on average")
            
            with col_s2:
                st.subheader("🎯 Optuna Impact Analysis")
                optuna_comparison = experiments_df.groupby('optuna')['f1_score'].agg(['mean', 'std', 'min', 'max']).round(4)
                optuna_comparison.index = ['Without Optuna', 'With Optuna']
                st.dataframe(optuna_comparison, use_container_width=True)
                
                # Calculate improvement
                improvement = ((optuna_comparison.loc['With Optuna', 'mean'] - optuna_comparison.loc['Without Optuna', 'mean']) / 
                              optuna_comparison.loc['Without Optuna', 'mean'] * 100)
                
                if improvement > 0:
                    st.success(f"✅ Optuna improves performance by {improvement:.2f}% on average")
                else:
                    st.warning(f"⚠️ Optuna decreases performance by {abs(improvement):.2f}% on average")
            
            # Top 5 models
            st.subheader("🏆 Top 5 Models")
            top5 = experiments_df.nlargest(5, 'f1_score')[['experiment_name', 'model_type', 'f1_score', 'pca', 'optuna']]
            
            for idx, row in top5.iterrows():
                col_t1, col_t2, col_t3 = st.columns([3, 1, 1])
                with col_t1:
                    st.write(f"**{row['experiment_name']}**")
                with col_t2:
                    st.metric("F1 Score", f"{row['f1_score']:.4f}")
                with col_t3:
                    tags = []
                    if row['pca']:
                        tags.append("PCA")
                    if row['optuna']:
                        tags.append("Optuna")
                    st.write(" • ".join(tags) if tags else "Baseline")
                st.markdown("---")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ============================================================================
# TAB 4: ABOUT
# ============================================================================
with tab4:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🏠 Housing Price Category Predictor
    
    This application predicts housing price categories based on various property features from the **Ames Housing Dataset**.
    
    #### 💰 Price Categories:
    The model classifies houses into **4 price categories**:
    - **Class 0**: Low Price Range
    - **Class 1**: Medium Price Range
    - **Class 2**: High Price Range
    - **Class 3**: Very High Price Range
    
    #### 🔬 Machine Learning Pipeline:
    
    **16 Experiments Conducted:**
    - **4 Classification Models**: RandomForest, GradientBoosting, XGBoost, LightGBM
    - **4 Configurations Each**:
      1. ❌ No PCA + ❌ No Hyperparameter Tuning
      2. ❌ No PCA + ✅ Optuna Tuning
      3. ✅ With PCA + ❌ No Tuning
      4. ✅ With PCA + ✅ Optuna Tuning
    
    #### 🛠️ Technology Stack:
    """)
    
    col_tech1, col_tech2, col_tech3 = st.columns(3)
    
    with col_tech1:
        st.markdown("""
        **Backend:**
        - FastAPI
        - Python 3.10
        - SQLite Database
        """)
    
    with col_tech2:
        st.markdown("""
        **ML Libraries:**
        - Scikit-learn
        - XGBoost
        - LightGBM
        - Optuna
        """)
    
    with col_tech3:
        st.markdown("""
        **Deployment:**
        - Docker
        - Docker Compose
        - Streamlit Frontend
        """)
    
    st.markdown("""
    #### 📊 Key Features:
    - ✅ Real-time predictions via REST API
    - ✅ Interactive web interface
    - ✅ Comprehensive experiment tracking
    - ✅ Model performance visualization
    - ✅ Probability distributions for predictions
    - ✅ Automatic model selection (best F1 score)
    
    #### 📁 Project Structure:
```
    housing_app_fall25/
    ├── api/              # FastAPI backend
    ├── streamlit/        # Streamlit frontend
    ├── notebooks/        # Jupyter notebooks for training
    ├── db/              # SQLite database
    ├── data/            # Dataset and schemas
    └── docker-compose.yml
```
    
    #### 🎓 Academic Project:
    - **Course**: Data Science / Machine Learning
    - **Dataset**: Ames Housing Dataset
    - **Task**: Multi-class Classification
    - **Evaluation Metric**: F1 Score (Weighted)
    
    #### 🔗 Resources:
    - [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
    - [FastAPI Documentation](https://fastapi.tiangolo.com/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    """)
    
    st.info("💡 **Note**: This application automatically uses the best performing model from all experiments.")
    
    # System info
    with st.expander("🖥️ System Information"):
        st.write(f"**Current Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**API Endpoint**: {API_URL}")
        
        try:
            health_response = requests.get(f"{API_URL}/health", timeout=2)
            if health_response.status_code == 200:
                health_data = health_response.json()
                st.write("**API Status**: ✅ Healthy")
                st.json(health_data)
        except:
            st.write("**API Status**: ❌ Offline")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏠 Housing Price Category Predictor | Built with ❤️ using FastAPI & Streamlit</p>
        <p>Data Science Final Project 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)