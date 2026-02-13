# 📈 AI Enterprise Growth Prediction System

A complete machine learning system that predicts next month's company revenue using RandomForest regression, built with Python, Streamlit, and scikit-learn.

## ✨ Features

- **Machine Learning Model**: RandomForestRegressor for accurate revenue prediction
- **Interactive Web UI**: Beautiful Streamlit interface for easy interaction
- **Real-time Predictions**: Input company metrics and get instant predictions
- **Model Performance Metrics**: View R² score, RMSE, MAE, and feature importance
- **Data Visualization**: Interactive charts showing current vs. predicted revenue
- **Model Retraining**: Train or retrain the model directly from the UI
- **Business Insights**: Get actionable recommendations based on predictions
- **Error Handling**: Comprehensive input validation and error management
- **Performance Optimized**: Model caching for fast predictions
- **Production Ready**: Deployable to Streamlit Cloud

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ravigohel142996/Company-Growth-Prediction-Model.git
cd Company-Growth-Prediction-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Step 1: Train the Model

```bash
python train_model.py
```

This will:
- Load company data from `data.csv`
- Train a RandomForestRegressor model
- Evaluate and display model performance
- Save the trained model to `model.pkl`

#### Step 2: Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

#### Step 3: Make Predictions

1. Enter your company's current metrics:
   - Monthly Revenue
   - Monthly Expenses
   - Number of Customers
   - Number of Employees
   - Growth Rate (%)

2. Click "Predict Next Month Revenue"

3. View the prediction results, visualizations, and business insights

## 📁 Project Structure

```
Company-Growth-Prediction-Model/
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── data.csv               # Company business data
├── requirements.txt       # Python dependencies
├── model.pkl             # Trained model (generated)
├── metrics.pkl           # Model metrics (generated)
└── README.md             # This file
```

## 📊 Data Format

The `data.csv` file contains company business metrics with the following columns:

| Column | Description |
|--------|-------------|
| Revenue | Current monthly revenue ($) |
| Expenses | Monthly operating expenses ($) |
| Customers | Number of active customers |
| Employees | Number of employees |
| Growth_Rate | Month-over-month growth rate (%) |
| Next_Month_Revenue | Target variable - next month's revenue ($) |

## 🎯 Model Details

- **Algorithm**: RandomForestRegressor
- **Features**: Revenue, Expenses, Customers, Employees, Growth_Rate
- **Target**: Next_Month_Revenue
- **Train/Test Split**: 80/20
- **Number of Trees**: 100
- **Max Depth**: 10

### Model Performance

Typical performance on the included dataset:
- **R² Score**: ~0.996 (test set)
- **RMSE**: ~$6,830
- **MAE**: ~$6,009

## 🔧 Customization

### Using Your Own Data

Replace `data.csv` with your own data following the same column structure, then retrain the model.

### Adjusting Model Parameters

Edit `train_model.py` and modify the RandomForestRegressor parameters:

```python
model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all CPU cores
)
```

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Deploy your app by selecting your repository

The app is ready for deployment with no additional configuration needed!

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

## 📈 Future Enhancements

- [ ] Add more ML models (XGBoost, Neural Networks)
- [ ] Support for multiple prediction periods
- [ ] Historical trend analysis
- [ ] Export predictions to CSV/PDF
- [ ] User authentication and data persistence
- [ ] Advanced feature engineering
- [ ] Automated model retraining scheduler

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Ravi Gohel**

## 🙏 Acknowledgments

- Built as part of the AI Enterprise Growth Prediction System project
- Thanks to the open-source community for the amazing tools and libraries
