"""
Wine Cultivar Origin Model Building Script
CSC415 Holiday Assignment - Project 6 (Part A)
Author: SOMADE TOLUWANI (22CH032062)

This script builds a Random Forest Classifier to predict wine cultivar origin
using the UCI Wine Dataset with 6 selected features.
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

def build_and_train_model():
    """
    Build and train the Wine Cultivar Classification model.
    
    Selected Features (6 as per assignment):
    1. alcohol
    2. malic_acid
    3. flavanoids
    4. color_intensity
    5. hue
    6. proline
    """
    print("=" * 60)
    print("🍷 Wine Cultivar Origin Prediction - Model Building")
    print("=" * 60)
    print("\nAuthor: SOMADE TOLUWANI (22CH032062)")
    print("Algorithm: Random Forest Classifier")
    print("Model Persistence: Joblib")
    print("-" * 60)
    
    # Step 1: Load the Wine Dataset
    print("\n📊 Step 1: Loading Wine Dataset...")
    wine = load_wine()
    X_full = wine.data
    y = wine.target
    feature_names = wine.feature_names
    
    print(f"   Total samples: {len(y)}")
    print(f"   Total features: {len(feature_names)}")
    print(f"   Classes: {wine.target_names}")
    
    # Step 2: Feature Selection (Select 6 features as per assignment)
    print("\n🎯 Step 2: Feature Selection...")
    selected_feature_indices = [0, 1, 6, 9, 10, 12]  # alcohol, malic_acid, flavanoids, color_intensity, hue, proline
    selected_features = [feature_names[i] for i in selected_feature_indices]
    
    print("   Selected Features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"   {i}. {feat}")
    
    X = X_full[:, selected_feature_indices]
    
    # Step 3: Data Preprocessing - Check for missing values
    print("\n🔧 Step 3: Data Preprocessing...")
    missing_values = np.isnan(X).sum()
    print(f"   Missing values: {missing_values}")
    
    # Step 4: Feature Scaling (Mandatory as per assignment)
    print("\n📏 Step 4: Feature Scaling (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("   ✅ Features scaled successfully")
    
    # Step 5: Train-Test Split
    print("\n📂 Step 5: Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Step 6: Model Training
    print("\n🤖 Step 6: Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   ✅ Model trained successfully")
    
    # Step 7: Model Evaluation
    print("\n📈 Step 7: Model Evaluation...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n   Performance Metrics:")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    print("\n   Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=['Cultivar 1', 'Cultivar 2', 'Cultivar 3']))
    
    # Step 8: Save Model to Disk
    print("💾 Step 8: Saving Model to Disk...")
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': selected_features,
        'accuracy': accuracy,
        'algorithm': 'Random Forest Classifier',
        'author': 'SOMADE TOLUWANI (22CH032062)'
    }
    
    joblib.dump(model_data, 'wine_cultivar_model.pkl')
    print("   ✅ Model saved as 'wine_cultivar_model.pkl' using Joblib")
    
    print("\n" + "=" * 60)
    print("✅ Model Building Complete!")
    print("=" * 60)
    
    return model, scaler, accuracy

if __name__ == '__main__':
    build_and_train_model()
