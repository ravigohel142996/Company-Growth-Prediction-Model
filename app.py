"""
AI Enterprise Growth Prediction System - Streamlit Web Application

This is the main web application that provides an interactive UI for:
- Loading the trained ML model
- Accepting user inputs for company metrics
- Making revenue predictions
- Displaying results with visualizations
- Allowing model retraining

Author: AI Growth System
Date: 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Fix matplotlib backend for Streamlit Cloud
# Keep matplotlib optional so the app can still run when the package
# is unavailable in constrained deployment environments.
HAS_MATPLOTLIB = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False
    matplotlib = None
    plt = None

# Import scikit-learn components for training
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
except ImportError as e:
    st.error(f"Error importing scikit-learn: {e}")
    st.stop()

# Page configuration
try:
    st.set_page_config(
        page_title="AI Enterprise Growth Prediction",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    # If page config already set, ignore
    pass

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_training_data(file_path='data.csv'):
    """
    Load and cache training data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset or None if not found
    """
    try:
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            return data
        else:
            st.warning(f"⚠️ Data file '{file_path}' not found.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def train_model_inline(data):
    """
    Train the model inline (not using subprocess) for cloud compatibility.
    
    Args:
        data (pd.DataFrame): Training data
        
    Returns:
        tuple: (model, metrics) or (None, None) if training fails
    """
    try:
        # Define features and target
        features = ['Revenue', 'Expenses', 'Customers', 'Employees', 'Growth_Rate']
        target = 'Next_Month_Revenue'
        
        # Check if all required columns exist
        missing_cols = set(features + [target]) - set(data.columns)
        if missing_cols:
            st.error(f"Missing columns in dataset: {missing_cols}")
            return None, None
        
        X = data[features]
        y = data[target]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Feature importance
        importance = model.feature_importances_
        
        metrics = {
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'feature_importance': dict(zip(features, importance))
        }
        
        # Save model and metrics
        try:
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('metrics.pkl', 'wb') as f:
                pickle.dump(metrics, f)
        except Exception as e:
            st.warning(f"Could not save model files: {e}")
        
        return model, metrics
        
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        return None, None


@st.cache_resource
def load_model():
    """
    Load the trained model and metrics from pickle files.
    If model doesn't exist, train it automatically.
    Uses Streamlit caching for performance optimization.
    
    Returns:
        tuple: (model, metrics) or (None, None) if loading fails
    """
    model_path = 'model.pkl'
    metrics_path = 'metrics.pkl'
    
    # Try to load existing model
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            metrics = None
            if os.path.exists(metrics_path):
                with open(metrics_path, 'rb') as f:
                    metrics = pickle.load(f)
            
            return model, metrics
        except Exception as e:
            st.warning(f"Error loading existing model: {str(e)}")
    
    # If model doesn't exist, try to train it
    st.info("🔄 No trained model found. Training model automatically...")
    data = load_training_data()
    
    if data is not None:
        model, metrics = train_model_inline(data)
        if model is not None:
            st.success("✓ Model trained successfully!")
            return model, metrics
    
    return None, None


def validate_inputs(revenue, expenses, customers, employees, growth_rate):
    """
    Validate user inputs for logical consistency.
    
    Args:
        revenue: Monthly revenue
        expenses: Monthly expenses
        customers: Number of customers
        employees: Number of employees
        growth_rate: Growth rate percentage
        
    Returns:
        tuple: (is_valid, error_message)
    """
    errors = []
    
    if revenue < 0:
        errors.append("Revenue cannot be negative")
    if expenses < 0:
        errors.append("Expenses cannot be negative")
    if customers < 0:
        errors.append("Number of customers cannot be negative")
    if employees < 0:
        errors.append("Number of employees cannot be negative")
    if growth_rate < -100:
        errors.append("Growth rate cannot be less than -100%")
    
    if expenses > revenue * 1.5:
        errors.append("Warning: Expenses are significantly higher than revenue")
    
    if len(errors) > 0:
        return False, " | ".join(errors)
    
    return True, None


@st.cache_data
def create_visualization(inputs, prediction):
    """
    Create visualization comparing current and predicted metrics.
    Uses caching for performance.
    
    Args:
        inputs (dict): User input values
        prediction (float): Predicted next month revenue
        
    Returns:
        matplotlib.figure.Figure: Created figure
    """
    if not HAS_MATPLOTLIB:
        return None

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Chart 1: Current Metrics Bar Chart
        metrics = ['Revenue', 'Expenses', 'Customers\n(x100)', 'Employees\n(x1000)']
        values = [
            inputs['revenue'],
            inputs['expenses'],
            inputs['customers'] * 100,
            inputs['employees'] * 1000
        ]
        
        colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b']
        axes[0].bar(metrics, values, color=colors, alpha=0.7)
        axes[0].set_title('Current Business Metrics', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Value ($)', fontsize=10)
        axes[0].tick_params(axis='x', rotation=0)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Chart 2: Revenue Comparison
        revenue_data = ['Current\nRevenue', 'Predicted\nNext Month']
        revenue_values = [inputs['revenue'], prediction]
        
        colors2 = ['#667eea', '#43e97b']
        bars = axes[1].bar(revenue_data, revenue_values, color=colors2, alpha=0.7)
        axes[1].set_title('Revenue Comparison', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Revenue ($)', fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.0f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None


def display_model_metrics(metrics):
    """
    Display model performance metrics in an organized layout.
    
    Args:
        metrics (dict): Model evaluation metrics
    """
    try:
        st.subheader("📊 Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="R² Score (Test)",
                value=f"{metrics['test_r2']:.4f}",
                help="Coefficient of determination (1.0 is perfect)"
            )
        
        with col2:
            st.metric(
                label="RMSE (Test)",
                value=f"${metrics['test_rmse']:,.2f}",
                help="Root Mean Squared Error"
            )
        
        with col3:
            st.metric(
                label="MAE (Test)",
                value=f"${metrics['test_mae']:,.2f}",
                help="Mean Absolute Error"
            )
        
        # Feature importance
        with st.expander("🔍 Feature Importance Details"):
            importance_df = pd.DataFrame(
                list(metrics['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            st.dataframe(importance_df, use_container_width=True)
            
            # Bar chart for feature importance
            if HAS_MATPLOTLIB:
                try:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
                    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors_gradient)
                    ax.set_xlabel('Importance Score')
                    ax.set_title('Feature Importance in Prediction Model')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Could not display feature importance chart: {e}")
            else:
                st.info("Install matplotlib to enable the feature-importance chart.")
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")


def main():
    """
    Main application function with comprehensive error handling.
    """
    try:
        # Header
        st.markdown('<h1 class="main-header">📈 AI Enterprise Growth Prediction System</h1>', 
                    unsafe_allow_html=True)
        st.markdown("### Predict Your Company's Next Month Revenue Using AI")
        st.markdown("---")

        if not HAS_MATPLOTLIB:
            st.warning(
                "Matplotlib is not installed in this environment. "
                "Prediction features still work, but chart visualizations are disabled."
            )
        
        # Sidebar
        with st.sidebar:
            # Remove external image dependency - use emoji instead
            st.markdown("# 📊 AI Growth System")
            st.markdown("---")
            
            st.markdown("## 🎯 About")
            st.info(
                "This AI system uses **RandomForestRegressor** to predict next month's revenue "
                "based on current business metrics. Train the model with your data and get "
                "instant predictions!"
            )
            
            st.markdown("## 🔧 Model Management")
            
            # Load model
            model, metrics = load_model()
            
            if model is None:
                st.warning("⚠️ Model training failed. Please check data file.")
                st.markdown("---")
                st.markdown("## 📚 Troubleshooting")
                st.markdown("""
                - Ensure `data.csv` exists
                - Check data format matches requirements
                - Try refreshing the page
                """)
            else:
                st.success("✓ Model loaded successfully!")
                
                # Manual retrain button
                if st.button("🔄 Retrain Model", key="retrain_sidebar"):
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    try:
                        # Remove old model files
                        if os.path.exists('model.pkl'):
                            os.remove('model.pkl')
                        if os.path.exists('metrics.pkl'):
                            os.remove('metrics.pkl')
                    except:
                        pass
                    st.rerun()
            
            st.markdown("## 📚 How to Use")
            st.markdown("""
            1. **Model loads automatically** on startup
            2. **Enter your company metrics** in the form
            3. **Click Predict** to get next month's revenue
            4. **View results** and visualizations
            """)
            
            st.markdown("---")
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Main content area
        if model is None:
            st.error("🚫 Model could not be loaded or trained.")
            st.info("Please ensure data.csv exists with the correct format.")
            
            # Show expected data format
            st.subheader("📋 Expected Data Format")
            st.markdown("""
            The `data.csv` file should contain these columns:
            - Revenue
            - Expenses
            - Customers
            - Employees
            - Growth_Rate
            - Next_Month_Revenue
            """)
            
            # Try to show sample data if available
            try:
                sample_data = load_training_data()
                if sample_data is not None:
                    st.subheader("📊 Current Data Sample")
                    st.dataframe(sample_data.head(10), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load sample data: {e}")
            
            return
        
        # Display model metrics if available
        if metrics:
            display_model_metrics(metrics)
            st.markdown("---")
        
        # Prediction section
        st.subheader("🎯 Make a Prediction")
        st.markdown("Enter your company's current metrics below:")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                revenue = st.number_input(
                    "💰 Current Monthly Revenue ($)",
                    min_value=0.0,
                    value=150000.0,
                    step=1000.0,
                    help="Your company's current monthly revenue"
                )
                
                expenses = st.number_input(
                    "💸 Monthly Expenses ($)",
                    min_value=0.0,
                    value=80000.0,
                    step=1000.0,
                    help="Your company's monthly operating expenses"
                )
                
                customers = st.number_input(
                    "👥 Number of Customers",
                    min_value=0,
                    value=300,
                    step=10,
                    help="Total number of active customers"
                )
            
            with col2:
                employees = st.number_input(
                    "👔 Number of Employees",
                    min_value=0,
                    value=30,
                    step=1,
                    help="Total number of employees"
                )
                
                growth_rate = st.number_input(
                    "📈 Growth Rate (%)",
                    min_value=-100.0,
                    max_value=1000.0,
                    value=22.5,
                    step=0.1,
                    help="Your company's month-over-month growth rate"
                )
            
            st.markdown("###")
            submit_button = st.form_submit_button("🔮 Predict Next Month Revenue", 
                                                 use_container_width=True)
        
        # Make prediction
        if submit_button:
            # Validate inputs
            is_valid, error_msg = validate_inputs(revenue, expenses, customers, employees, growth_rate)
            
            if not is_valid:
                st.error(f"❌ Input validation failed: {error_msg}")
                return
            
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'Revenue': [revenue],
                    'Expenses': [expenses],
                    'Customers': [customers],
                    'Employees': [employees],
                    'Growth_Rate': [growth_rate]
                })
                
                # Make prediction
                with st.spinner("Calculating prediction..."):
                    prediction = model.predict(input_data)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Main prediction display
                st.markdown(
                    f'<div class="prediction-box">Next Month Revenue: ${prediction:,.2f}</div>',
                    unsafe_allow_html=True
                )
                
                # Additional insights
                col1, col2, col3, col4 = st.columns(4)
                
                revenue_change = prediction - revenue
                revenue_change_pct = (revenue_change / revenue) * 100 if revenue > 0 else 0
                profit_margin = ((prediction - expenses) / prediction) * 100 if prediction > 0 else 0
                
                with col1:
                    st.metric(
                        "Revenue Change",
                        f"${revenue_change:,.2f}",
                        f"{revenue_change_pct:+.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Current Revenue",
                        f"${revenue:,.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Projected Profit Margin",
                        f"{profit_margin:.2f}%"
                    )
                
                with col4:
                    customer_value = prediction / customers if customers > 0 else 0
                    st.metric(
                        "Revenue per Customer",
                        f"${customer_value:,.2f}"
                    )
                
                # Visualizations
                st.markdown("###")
                inputs_dict = {
                    'revenue': revenue,
                    'expenses': expenses,
                    'customers': customers,
                    'employees': employees,
                    'growth_rate': growth_rate
                }
                
                fig = create_visualization(inputs_dict, prediction)
                if fig is not None:
                    st.pyplot(fig)
                    if HAS_MATPLOTLIB:
                        plt.close(fig)
                
                # Business insights
                st.markdown("###")
                with st.expander("💡 Business Insights & Recommendations"):
                    st.markdown("**Key Observations:**")
                    
                    if revenue_change > 0:
                        st.success(f"✓ Projected revenue growth of ${revenue_change:,.2f} ({revenue_change_pct:.2f}%)")
                    else:
                        st.warning(f"⚠ Projected revenue decline of ${abs(revenue_change):,.2f} ({abs(revenue_change_pct):.2f}%)")
                    
                    if profit_margin > 20:
                        st.success(f"✓ Healthy profit margin of {profit_margin:.2f}%")
                    elif profit_margin > 10:
                        st.info(f"ℹ Moderate profit margin of {profit_margin:.2f}%")
                    else:
                        st.warning(f"⚠ Low profit margin of {profit_margin:.2f}%")
                    
                    expense_ratio = (expenses / revenue) * 100 if revenue > 0 else 0
                    if expense_ratio < 50:
                        st.success(f"✓ Good expense management at {expense_ratio:.2f}% of revenue")
                    else:
                        st.warning(f"⚠ High expenses at {expense_ratio:.2f}% of revenue")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.exception(e)
    
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Fatal application error: {str(e)}")
        st.exception(e)
