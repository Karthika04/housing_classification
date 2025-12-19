
#Imports and Setup
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import json

print("All imports successful")

#Load Ames Housing Data
data_path = Path('/Users/karthika/housing_app_fall25/data')
df = pd.read_csv(data_path/'housing.csv')

print(f"📊 Original shape: {df.shape}")
print(f"📋 Columns: {len(df.columns)}")
print("\n🔍 First few rows:")
print(df.head())

#Create Classification Target
print("\n" + "="*70)
print("CREATING PRICE CATEGORIES")
print("="*70)

# Check if PriceCategory already exists
if 'PriceCategory' not in df.columns:
    # Calculate quartiles
    q1 = df['SalePrice'].quantile(0.25)
    q2 = df['SalePrice'].quantile(0.50)
    q3 = df['SalePrice'].quantile(0.75)
    
    print(f"Price Quartiles:")
    print(f"  Q1 (25%): ${q1:,.0f}")
    print(f"  Q2 (50%): ${q2:,.0f}")
    print(f"  Q3 (75%): ${q3:,.0f}")
    
    # Create categorical target
    def create_price_category(price):
        if price <= q1:
            return 0  # Low
        elif price <= q2:
            return 1  # Medium
        elif price <= q3:
            return 2  # High
        else:
            return 3  # Very High
    
    df['PriceCategory'] = df['SalePrice'].apply(create_price_category)
    print("\n PriceCategory created")
else:
    print("PriceCategory already exists in dataset")
    q1 = df[df['PriceCategory'] == 0]['SalePrice'].max()
    q2 = df[df['PriceCategory'] == 1]['SalePrice'].max()
    q3 = df[df['PriceCategory'] == 2]['SalePrice'].max()

# Verify distribution
print(f"\nClass Distribution:")
print(df['PriceCategory'].value_counts().sort_index())


#Feature Selection and Cleaning
print("\n" + "="*70)
print("FEATURE SELECTION AND CLEANING")
print("="*70)

# Exclude ID columns and SalePrice (we have PriceCategory now)
columns_to_exclude = ['Order', 'PID', 'SalePrice'] if 'Order' in df.columns else ['Id', 'SalePrice']

# Get all available feature columns
available_features = [col for col in df.columns if col not in columns_to_exclude]

print(f"Total features (including target): {len(available_features)}")

# Create clean dataframe
df_clean = df[available_features].copy()

print(f"Cleaned dataset shape: {df_clean.shape}")


# CELL 5: Handle Missing Values
print("\n" + "="*70)
print("HANDLING MISSING VALUES")
print("="*70)
print(f"Missing values before: {df_clean.isnull().sum().sum()}")

# For numeric columns: fill with median
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col != 'PriceCategory']

print(f"\n Numeric columns: {len(numeric_cols)}")
for col in numeric_cols:
    if df_clean[col].isnull().any():
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)

# For categorical columns: fill with 'Missing'
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

print(f"📊 Categorical columns: {len(categorical_cols)}")
for col in categorical_cols:
    if df_clean[col].isnull().any():
        df_clean[col].fillna('Missing', inplace=True)

print(f"\n Missing values after: {df_clean.isnull().sum().sum()}")

print("\n" + "="*70)
print("DATA SUMMARY")
print("="*70)
print(f"Total samples: {len(df_clean)}")
print(f"Total features: {len(df_clean.columns) - 1}")
print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Target classes: {df_clean['PriceCategory'].nunique()}")
print("="*70)



# CELL 6: Create SQLite Database
print("\n" + "="*70)
print("CREATING DATABASE")
print("="*70)

db_path = Path('/Users/karthika/housing_app_fall25/db/housing_classification.db')
db_path.parent.mkdir(exist_ok=True, parents=True)

# Remove old database if exists
if db_path.exists():
    db_path.unlink()
    print("🗑️  Removed old database")

# Create connection and table
conn = sqlite3.connect(db_path)
df_clean.to_sql('housing_data', conn, if_exists='replace', index=False)

print(f"Database created: {db_path}")
print(f"   Total records: {len(df_clean)}")
print(f"   Total columns: {len(df_clean.columns)}")

# Verify database
test_query = pd.read_sql("SELECT * FROM housing_data LIMIT 5", conn)
print("\n📋 First 5 records from database:")
print(test_query.head())

# Check target distribution
target_dist = pd.read_sql(
    "SELECT PriceCategory, COUNT(*) as count FROM housing_data GROUP BY PriceCategory ORDER BY PriceCategory",
    conn
)
print("\n📊 Target distribution in database:")
for _, row in target_dist.iterrows():
    print(f"   Class {row['PriceCategory']}: {row['count']} samples")

conn.close()
print("\n Database creation completed!")


# CELL 7: Save Data Schema
print("\n" + "="*70)
print("SAVING DATA SCHEMA")
print("="*70)

data_path = Path('/Users/karthika/housing_app_fall25/data')
data_path.mkdir(exist_ok=True, parents=True)

# Get feature lists
numeric_features = [col for col in df_clean.select_dtypes(include=[np.number]).columns if col != 'PriceCategory']
categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()

# Calculate price ranges for each category
price_ranges = []
for cat in range(4):
    cat_prices = df[df['PriceCategory'] == cat]['SalePrice']
    price_ranges.append({
        'category': cat,
        'min': float(cat_prices.min()),
        'max': float(cat_prices.max()),
        'mean': float(cat_prices.mean()),
        'count': int(len(cat_prices))
    })

# Create schema
schema = {
    "target": "PriceCategory",
    "target_mapping": {
        "0": f"Low (${price_ranges[0]['min']:,.0f} - ${price_ranges[0]['max']:,.0f})",
        "1": f"Medium (${price_ranges[1]['min']:,.0f} - ${price_ranges[1]['max']:,.0f})",
        "2": f"High (${price_ranges[2]['min']:,.0f} - ${price_ranges[2]['max']:,.0f})",
        "3": f"Very High (${price_ranges[3]['min']:,.0f} - ${price_ranges[3]['max']:,.0f})"
    },
    "price_ranges": price_ranges,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "total_features": len(df_clean.columns) - 1,
    "total_samples": len(df_clean),
    "database_path": "db/housing_classification.db",
    "table_name": "housing_data",
    "quartiles": {
        "q1": float(q1),
        "q2": float(q2),
        "q3": float(q3)
    }
}

# Save schema
schema_path = data_path / 'data_schema.json'
with open(schema_path, 'w') as f:
    json.dump(schema, f, indent=2)

print(f" Schema saved: {schema_path}")
print("\n Schema Summary:")
print(f"   Total Features: {schema['total_features']}")
print(f"   Numeric Features: {len(numeric_features)}")
print(f"   Categorical Features: {len(categorical_features)}")
print(f"   Target Classes: 4")
print("\n Price Categories:")
for key, value in schema['target_mapping'].items():
    print(f"   Class {key}: {value}")

print("\n" + "="*70)
print(" DATABASE SETUP COMPLETE!")
print("="*70)



