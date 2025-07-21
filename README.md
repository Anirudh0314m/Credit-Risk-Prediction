# Loan Default Prediction - Business Analytics Project

## Project Overview

This project implements a comprehensive machine learning solution for predicting loan defaults using advanced feature engineering and multiple modeling techniques. The system analyzes loan applications, bank transaction patterns, and credit bureau data to assess the likelihood of borrower default.

## Business Problem

Financial institutions need to accurately assess credit risk to:
- Minimize loan defaults and financial losses
- Make informed lending decisions
- Optimize interest rates based on risk profiles
- Comply with regulatory requirements

## Key Features

- **Multi-source Data Integration**: Combines loan applications, bank transactions, and credit bureau reports
- **Advanced Feature Engineering**: Creates 60+ engineered features from raw data
- **Machine Learning Models**: Implements Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning
- **Class Imbalance Handling**: Uses SMOTE, class weights, and threshold optimization
- **Model Evaluation**: Comprehensive metrics including ROC-AUC, Precision-Recall curves, and confusion matrices

## Project Structure

```
BA/
├── data/
│   ├── raw/
│   │   ├── applications/
│   │   ├── bank_transactions/
│   │   └── credit_bureau_reports/
│   └── processed/
│       └── features/
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── test.ipynb
│   └── models/
│       └── tuned_xgboost_model/
├── scripts/
│   ├── db_connecter.py
│   ├── feature_engineering.py
│   └── generate_synthetic_data.py
├── models/
├── .gitignore
├── .gitattributes
└── README.md
```

## Data Sources

### 1. Loan Applications (`loan_applications_5k.csv`)
- Customer demographics and loan details
- Loan amount, term, purpose, collateral information
- Income and expense declarations

### 2. Bank Transactions (`bank_transactions_5k.csv`)
- Transaction history and banking behavior
- Income patterns and spending analysis
- Financial stability indicators

### 3. Credit Bureau Reports (`credit_bureau_data_5k.csv`)
- Credit history and existing obligations
- Credit scores and account information
- Default history and payment patterns

## Installation and Setup

### Prerequisites
- Python 3.8+
- MySQL Server (for database operations)
- Git LFS (for large file storage)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Anirudh0314m/BA.git
   cd BA
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost
   pip install mysql-connector-python jupyter matplotlib seaborn
   pip install joblib imbalanced-learn
   ```

4. **Configure database connection**
   Update the database configuration in `scripts/db_connecter.py`:
   ```python
   DB_CONFIG = {
       'host': 'localhost',
       'user': 'your_username',
       'password': 'your_password',
       'database': 'BA'
   }
   ```

## Usage

### 1. Data Exploration and Analysis
Open and run the main analysis notebook:
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

### 2. Feature Engineering
The notebook includes comprehensive feature engineering:
- Temporal features from application dates
- Financial ratios and derived metrics
- Credit utilization and payment behavior indicators
- Income stability and expense pattern analysis

### 3. Model Training
The project implements three main models:
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble method with class balancing
- **XGBoost**: Gradient boosting with hyperparameter tuning

### 4. Model Evaluation
Models are evaluated using:
- ROC-AUC scores
- Precision-Recall curves
- Confusion matrices
- Classification reports

## Model Performance

### XGBoost (Best Performing Model)
- **ROC-AUC**: 0.8542
- **Precision**: Optimized for default detection
- **Recall**: Balanced to minimize false negatives
- **F1-Score**: Comprehensive performance metric

### Key Model Features
- Handles class imbalance with `scale_pos_weight`
- Hyperparameter optimization using RandomizedSearchCV
- Cross-validation for robust performance estimates

## Feature Engineering Highlights

### Financial Health Indicators
- Debt-to-income ratios
- Monthly savings patterns
- Income stability metrics
- Expense categorization

### Credit Behavior Analysis
- Credit utilization ratios
- Payment history patterns
- Account age and diversity
- Default history indicators

### Banking Behavior Features
- Transaction frequency patterns
- Large withdrawal detection
- Overdraft frequency
- Recurring payment analysis

## Model Artifacts

Trained models and preprocessing objects are saved in:
- `notebooks/models/tuned_xgboost_model/tuned_xgboost_model_final.joblib`
- `notebooks/models/tuned_xgboost_model/scaler_final.joblib`

### Loading Saved Models
```python
import joblib

# Load the trained model
model = joblib.load('notebooks/models/tuned_xgboost_model/tuned_xgboost_model_final.joblib')

# Load the scaler
scaler = joblib.load('notebooks/models/tuned_xgboost_model/scaler_final.joblib')
```

## Key Insights

1. **Feature Importance**: Bank transaction patterns and credit history are the strongest predictors
2. **Class Imbalance**: Approximately 15% default rate in the dataset
3. **Threshold Optimization**: Custom thresholds improve precision-recall balance
4. **Temporal Patterns**: Application timing shows seasonal trends

## Requirements

### Core Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
mysql-connector-python>=8.0.0
jupyter>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
imbalanced-learn>=0.8.0
```




## Future Enhancements

- [ ] Real-time prediction API
- [ ] Model drift monitoring
- [ ] Additional data sources integration
- [ ] Deep learning models exploration
- [ ] Explainable AI dashboard
- [ ] A/B testing framework for model deployment

---

