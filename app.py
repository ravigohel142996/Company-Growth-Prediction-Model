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
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="AI Enterprise Growth Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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


@st.cache_resource
def load_model():
    """
    Load the trained model and metrics from pickle files.
    Uses Streamlit caching for performance optimization.
    
    Returns:
        tuple: (model, metrics) or (None, None) if files don't exist
    """
    model_path = 'model.pkl'
    metrics_path = 'metrics.pkl'
    
    if not os.path.exists(model_path):
        return None, None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                metrics = pickle.load(f)
        
        return model, metrics
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def train_model_from_app():
    """
    Train the model by running train_model.py script.
    
    Returns:
        bool: True if training succeeded, False otherwise
    """
    try:
        with st.spinner("Training model... This may take a few moments."):
            result = subprocess.run(
                ['python', 'train_model.py'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                st.success("✓ Model trained successfully!")
                st.text(result.stdout)
                return True
            else:
                st.error("✗ Model training failed!")
                st.text(result.stderr)
                return False
    except subprocess.TimeoutExpired:
        st.error("✗ Model training timed out!")
        return False
    except Exception as e:
        st.error(f"✗ Error during training: {str(e)}")
        return False


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


def create_visualization(inputs, prediction):
    """
    Create visualization comparing current and predicted metrics.
    
    Args:
        inputs (dict): User input values
        prediction (float): Predicted next month revenue
    """
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


def display_model_metrics(metrics):
    """
    Display model performance metrics in an organized layout.
    
    Args:
        metrics (dict): Model evaluation metrics
    """
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
        fig, ax = plt.subplots(figsize=(10, 4))
        colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors_gradient)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance in Prediction Model')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)


def main():
    """
    Main application function.
    """
    # Header
    st.markdown('<h1 class="main-header">📈 AI Enterprise Growth Prediction System</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Predict Your Company's Next Month Revenue Using AI")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=AI+Growth+System", 
                use_column_width=True)
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
            st.warning("⚠️ No trained model found!")
            st.markdown("Click below to train the model:")
            if st.button("🚀 Train Model", key="train_sidebar"):
                if train_model_from_app():
                    st.rerun()
        else:
            st.success("✓ Model loaded successfully!")
            if st.button("🔄 Retrain Model", key="retrain_sidebar"):
                if train_model_from_app():
                    st.rerun()
        
        st.markdown("## 📚 How to Use")
        st.markdown("""
        1. **Train the model** (if not already trained)
        2. **Enter your company metrics** in the form
        3. **Click Predict** to get next month's revenue
        4. **View results** and visualizations
        """)
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main content area
    if model is None:
        st.error("🚫 Please train the model first using the sidebar button.")
        st.info("The model needs to be trained before you can make predictions.")
        
        # Show sample data
        st.subheader("📋 Sample Training Data")
        try:
            sample_data = pd.read_csv('data.csv').head(10)
            st.dataframe(sample_data, use_container_width=True)
        except:
            st.warning("Could not load sample data.")
        
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
            st.pyplot(fig)
            
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


if __name__ == "__main__":
    main()
