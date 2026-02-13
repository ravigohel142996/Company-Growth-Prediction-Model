"""
AI Enterprise Growth Prediction System - Model Training Module

This module handles:
- Loading company business data from CSV
- Training a RandomForestRegressor model
- Evaluating model performance
- Saving the trained model using pickle

Author: AI Growth System
Date: 2026
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os


def load_data(file_path='data.csv'):
    """
    Load company business data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(file_path)
        print(f"✓ Data loaded successfully from {file_path}")
        print(f"  Dataset shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"✗ Error: File '{file_path}' not found.")
        raise
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        raise


def prepare_data(data):
    """
    Prepare features and target variables for model training.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Define features and target
    features = ['Revenue', 'Expenses', 'Customers', 'Employees', 'Growth_Rate']
    target = 'Next_Month_Revenue'
    
    # Check if all required columns exist
    missing_cols = set(features + [target]) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    
    X = data[features]
    y = data[target]
    
    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Data prepared for training")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a RandomForestRegressor model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators (int): Number of trees in the forest
        random_state (int): Random state for reproducibility
        
    Returns:
        RandomForestRegressor: Trained model
    """
    print("\n" + "="*50)
    print("Training RandomForest Regression Model...")
    print("="*50)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all available CPU cores
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    model.fit(X_train, y_train)
    
    print(f"✓ Model trained successfully")
    print(f"  Number of trees: {n_estimators}")
    print(f"  Max depth: {model.max_depth}")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on training and testing sets.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("\n" + "="*50)
    print("Model Evaluation Results")
    print("="*50)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training set
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Calculate metrics for testing set
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Display results
    print("\nTraining Set Metrics:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  RMSE: ${train_rmse:,.2f}")
    print(f"  MAE: ${train_mae:,.2f}")
    
    print("\nTesting Set Metrics:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  RMSE: ${test_rmse:,.2f}")
    print(f"  MAE: ${test_mae:,.2f}")
    
    # Feature importance
    print("\nFeature Importance:")
    features = ['Revenue', 'Expenses', 'Customers', 'Employees', 'Growth_Rate']
    importance = model.feature_importances_
    for feature, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {imp:.4f}")
    
    metrics = {
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'feature_importance': dict(zip(features, importance))
    }
    
    return metrics


def save_model(model, metrics, model_path='model.pkl', metrics_path='metrics.pkl'):
    """
    Save trained model and metrics to disk using pickle.
    
    Args:
        model: Trained model
        metrics (dict): Model evaluation metrics
        model_path (str): Path to save the model
        metrics_path (str): Path to save the metrics
    """
    try:
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n✓ Model saved to '{model_path}'")
        
        # Save metrics
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"✓ Metrics saved to '{metrics_path}'")
        
        # Display file sizes
        model_size = os.path.getsize(model_path) / 1024
        print(f"  Model file size: {model_size:.2f} KB")
        
    except Exception as e:
        print(f"✗ Error saving model: {str(e)}")
        raise


def main():
    """
    Main function to orchestrate the training pipeline.
    """
    try:
        print("\n" + "="*50)
        print("AI ENTERPRISE GROWTH PREDICTION SYSTEM")
        print("Model Training Pipeline")
        print("="*50 + "\n")
        
        # Step 1: Load data
        data = load_data('data.csv')
        
        # Step 2: Prepare data
        X_train, X_test, y_train, y_test = prepare_data(data)
        
        # Step 3: Train model
        model = train_model(X_train, y_train, n_estimators=100)
        
        # Step 4: Evaluate model
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Step 5: Save model
        save_model(model, metrics)
        
        print("\n" + "="*50)
        print("Training Pipeline Completed Successfully!")
        print("="*50 + "\n")
        print("Next step: Run the Streamlit app using 'streamlit run app.py'")
        
    except Exception as e:
        print(f"\n✗ Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
