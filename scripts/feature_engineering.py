import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# --- Configuration ---
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed/features'
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) # Ensure the output directory exists

# Define current date for age/history calculations (assume latest application date in data)
# Use a fixed date for reproducibility, slightly after the end of our synthetic data generation
CURRENT_DATE = datetime(2024, 12, 31)

# --- 1. Load Raw Data ---
print("Loading raw data...")
try:
    df_applications = pd.read_csv(os.path.join(RAW_DATA_DIR, 'applications', 'loan_applications_5k.csv'))
    df_credit_bureau = pd.read_csv(os.path.join(RAW_DATA_DIR, 'credit_bureau_reports', 'credit_bureau_data_5k.csv'))
    df_bank_transactions = pd.read_csv(os.path.join(RAW_DATA_DIR, 'bank_transactions', 'bank_transactions_5k.csv'))
except FileNotFoundError as e:
    print(f"Error: Raw data file not found. Please ensure you have run generate_synthetic_data.py first.")
    print(f"Missing file: {e.filename}")
    exit() # Exit if raw data is not found

# --- Initial Type Conversions ---
# FIX: Use specific format for each DataFrame based on observed data from errors
df_applications['application_date'] = pd.to_datetime(df_applications['application_date'], format="%d-%m-%Y")
df_credit_bureau['account_open_date'] = pd.to_datetime(df_credit_bureau['account_open_date'], format="%Y-%m-%d")
df_bank_transactions['transaction_date'] = pd.to_datetime(df_bank_transactions['transaction_date'], format="%Y-%m-%d")

print("Raw data loaded and initial type conversions done.")
print(f"Applications: {len(df_applications)} rows")
print(f"Credit Bureau: {len(df_credit_bureau)} rows")
print(f"Bank Transactions: {len(df_bank_transactions)} rows")
print("\n" + "="*50 + "\n")

# --- 2. Process Credit Bureau Data (for Established Borrowers) ---
print("Processing Credit Bureau Data...")
credit_features = []
# Filter only customers who have a credit history for this aggregation
customers_with_credit_history_ids = df_applications[df_applications['has_credit_history'] == 1]['customer_id'].unique()
df_credit_bureau_filtered = df_credit_bureau[df_credit_bureau['customer_id'].isin(customers_with_credit_history_ids)].copy()

if not df_credit_bureau_filtered.empty:
    # Basic Credit Score & Components
    # For simplicity, let's take the first credit score reported for the customer.
    # In reality, you'd aggregate or take latest from multiple bureau reports.
    df_credit_scores = df_credit_bureau_filtered.groupby('customer_id')['credit_score'].first().reset_index()

    # Aggregations per customer_id
    credit_agg = df_credit_bureau_filtered.groupby('customer_id').agg(
        num_credit_accounts_total=('account_id', 'nunique'),
        num_credit_accounts_active=('account_status', lambda x: (x == 'Active').sum()),
        oldest_account_age_months=('account_open_date', lambda x: ((CURRENT_DATE - x.min()).days // 30) if not pd.isna(x.min()) else np.nan),
        newest_account_age_months=('account_open_date', lambda x: ((CURRENT_DATE - x.max()).days // 30) if not pd.isna(x.max()) else np.nan),
        total_sanctioned_limit=('sanctioned_limit', 'sum'),
        total_outstanding_balance=('current_balance', 'sum'),
        num_defaults_reported_total=('num_defaults_total', 'sum'),
        num_settled_accounts_total=('num_settled_total', 'sum'),
        max_dpd_past_24m=('max_dpd_past_24m', 'max')
    ).reset_index()

    # Special handling for num_revolving_accounts_active to correctly use filtered data
    revolving_active_counts = df_credit_bureau_filtered[
        (df_credit_bureau_filtered['account_type'] == 'Credit Card') &
        (df_credit_bureau_filtered['account_status'] == 'Active')
    ].groupby('customer_id')['account_id'].nunique().reset_index(name='num_revolving_accounts_active')
    credit_agg = pd.merge(credit_agg, revolving_active_counts, on='customer_id', how='left').fillna({'num_revolving_accounts_active': 0})


    # Derived credit features
    credit_agg['overall_credit_utilization_ratio'] = credit_agg['total_outstanding_balance'] / credit_agg['total_sanctioned_limit']
    credit_agg['overall_credit_utilization_ratio'] = credit_agg['overall_credit_utilization_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0) # Handle division by zero/inf

    credit_agg['avg_account_age_months'] = (credit_agg['oldest_account_age_months'] + credit_agg['newest_account_age_months']) / 2
    credit_agg['ever_defaulted_flag'] = (credit_agg['num_defaults_reported_total'] > 0).astype(int)

    # Sum DPDs from individual accounts for the customer
    dpd_summary = df_credit_bureau_filtered.groupby('customer_id').agg(
        num_30_dpd_last_6m=('num_30_dpd_last_6m', 'sum'),
        num_90_dpd_last_12m=('num_90_dpd_last_12m', 'sum')
    ).reset_index()

    credit_features_df = df_credit_scores.merge(credit_agg, on='customer_id', how='left')
    credit_features_df = credit_features_df.merge(dpd_summary, on='customer_id', how='left')

    print(f"Generated {len(credit_features_df)} credit bureau feature sets.")
    print(credit_features_df.head())
else:
    # Create an empty DataFrame with expected columns if no customers have credit history
    # This ensures a consistent structure for merging later
    credit_features_df = pd.DataFrame(columns=['customer_id', 'credit_score', 'num_credit_accounts_total', 'num_credit_accounts_active',
                                              'oldest_account_age_months', 'newest_account_age_months', 'total_sanctioned_limit',
                                              'total_outstanding_balance', 'overall_credit_utilization_ratio', 'avg_account_age_months',
                                              'num_revolving_accounts_active', 'num_defaults_reported_total', 'num_settled_accounts_total',
                                              'max_dpd_past_24m', 'ever_defaulted_flag', 'num_30_dpd_last_6m', 'num_90_dpd_last_12m'])
print("\n" + "="*50 + "\n")


# --- 3. Process Bank Transactions Data (for All Borrowers) ---
print("Processing Bank Transactions Data...")
transaction_features = []

# Ensure transaction amounts are positive for credit, negative for debit
df_bank_transactions['signed_amount'] = df_bank_transactions.apply(lambda row: row['amount'] if row['type'] == 'credit' else -row['amount'], axis=1)

# Sort transactions by customer and date for correct running balance calculation
df_bank_transactions = df_bank_transactions.sort_values(by=['customer_id', 'transaction_date'])

for customer_id in df_applications['customer_id'].unique():
    customer_txns_all = df_bank_transactions[df_bank_transactions['customer_id'] == customer_id].copy()

    app_date = df_applications[df_applications['customer_id'] == customer_id]['application_date'].iloc[0]

    # Filter transactions for the relevant history period up to application date
    txns_in_period = customer_txns_all[customer_txns_all['transaction_date'] <= app_date].copy()
    txns_last_6m = txns_in_period[txns_in_period['transaction_date'] >= (app_date - pd.DateOffset(months=6))]
    txns_last_12m = txns_in_period[txns_in_period['transaction_date'] >= (app_date - pd.DateOffset(months=12))]

    # Initialize features for current customer, will be updated
    current_customer_features = {'customer_id': customer_id}

    if txns_in_period.empty:
        # If no transactions, fill with zeros or NaNs for all transaction-derived features
        # We'll fill with zeros for numerical, and specific values for flags as default
        current_customer_features.update({
            'avg_monthly_net_income_6m': 0, 'std_dev_monthly_net_income_6m': 0,
            'num_distinct_income_sources_6m': 0, 'longest_income_gap_days_12m': 0,
            'has_recurring_salary_flag': 0, 'avg_monthly_essential_expenses_6m': 0,
            'avg_monthly_discretionary_expenses_6m': 0, 'essential_to_discretionary_spend_ratio': 0,
            'num_recurring_bill_payments_detected': 0, 'num_overdrafts_6m': 0,
            'num_bounced_transactions_6m': 0, 'large_cash_withdrawal_frequency_6m': 0,
            'avg_monthly_balance_min_6m': 0, 'avg_monthly_savings_transfers_6m': 0,
            'has_investment_transactions_flag': 0,
            'detected_other_emis_sum': 0 # Initialize for later use in composite features
        })
        transaction_features.append(current_customer_features)
        continue

    # Income & Cash Flow Stability
    monthly_incomes_6m = txns_last_6m[txns_last_6m['type'] == 'credit'].groupby(pd.Grouper(key='transaction_date', freq='ME'))['amount'].sum()
    # Reindex to ensure all 6 months are represented, fill missing with 0 for average/std dev calculation
    monthly_incomes_6m = monthly_incomes_6m.reindex(pd.date_range(end=app_date.replace(day=1)+pd.DateOffset(months=1)-timedelta(days=1), periods=6, freq='ME'), fill_value=0)

    current_customer_features['avg_monthly_net_income_6m'] = monthly_incomes_6m.mean()
    current_customer_features['std_dev_monthly_net_income_6m'] = monthly_incomes_6m.std() if len(monthly_incomes_6m) > 1 else 0

    income_transactions = txns_last_12m[txns_last_12m['category'] == 'Income'].sort_values('transaction_date')
    income_gaps = income_transactions['transaction_date'].diff().dt.days
    current_customer_features['longest_income_gap_days_12m'] = income_gaps.max() if not income_gaps.empty else 0

    current_customer_features['num_distinct_income_sources_6m'] = txns_last_6m[txns_last_6m['category'] == 'Income']['description'].nunique()

    has_recurring_salary_flag = 0
    if not income_transactions.empty and ('Salary' in income_transactions['description'].values or 'Freelance Income' in income_transactions['description'].values):
        if (monthly_incomes_6m > 0).sum() >= 4: # At least 4 months with some income
            has_recurring_salary_flag = 1
    current_customer_features['has_recurring_salary_flag'] = has_recurring_salary_flag

    # Expense & Spending Behavior
    monthly_expenses_essential_6m = txns_last_6m[txns_last_6m['category'] == 'Essential'].groupby(pd.Grouper(key='transaction_date', freq='ME'))['amount'].sum()
    monthly_expenses_essential_6m = monthly_expenses_essential_6m.reindex(pd.date_range(end=app_date.replace(day=1)+pd.DateOffset(months=1)-timedelta(days=1), periods=6, freq='ME'), fill_value=0)
    current_customer_features['avg_monthly_essential_expenses_6m'] = monthly_expenses_essential_6m.mean()

    monthly_expenses_discretionary_6m = txns_last_6m[txns_last_6m['category'] == 'Discretionary'].groupby(pd.Grouper(key='transaction_date', freq='ME'))['amount'].sum()
    monthly_expenses_discretionary_6m = monthly_expenses_discretionary_6m.reindex(pd.date_range(end=app_date.replace(day=1)+pd.DateOffset(months=1)-timedelta(days=1), periods=6, freq='ME'), fill_value=0)
    current_customer_features['avg_monthly_discretionary_expenses_6m'] = monthly_expenses_discretionary_6m.mean()

    current_customer_features['essential_to_discretionary_spend_ratio'] = \
        current_customer_features['avg_monthly_essential_expenses_6m'] / \
        (current_customer_features['avg_monthly_discretionary_expenses_6m'] + 1e-6) # Add epsilon to avoid div by zero

    current_customer_features['num_recurring_bill_payments_detected'] = \
        txns_last_6m[txns_last_6m['description'].isin(['Rent Payment', 'Loan EMI Deduction'])].shape[0] # Count distinct instances, not types

    current_customer_features['num_overdrafts_6m'] = txns_last_6m[txns_last_6m['description'].str.contains('Overdraft', na=False)].shape[0]
    current_customer_features['num_bounced_transactions_6m'] = txns_last_6m[txns_last_6m['description'].str.contains('Bounce', na=False)].shape[0]

    current_customer_features['large_cash_withdrawal_frequency_6m'] = \
        txns_last_6m[(txns_last_6m['description'] == 'ATM Withdrawal') & (txns_last_6m['amount'] > 10000)].shape[0]

    # Calculate daily balances and then average of monthly minimum balances
    txns_for_balance = txns_in_period.sort_values('transaction_date').copy()
    if not txns_for_balance.empty:
        txns_for_balance['running_balance'] = txns_for_balance['signed_amount'].cumsum()

        # Get daily balances for the last 6 months
        daily_balances = txns_for_balance.set_index('transaction_date')['running_balance'].resample('D').last().ffill().dropna()
        daily_balances_last_6m = daily_balances[daily_balances.index >= (app_date - pd.DateOffset(months=6))]

        if not daily_balances_last_6m.empty:
            # Group by month and find min daily balance for each month, then average those minimums
            monthly_min_balances = daily_balances_last_6m.groupby(pd.Grouper(freq='ME')).min()
            current_customer_features['avg_monthly_balance_min_6m'] = monthly_min_balances.mean()
        else:
            current_customer_features['avg_monthly_balance_min_6m'] = 0 # No daily balances in period
    else:
        current_customer_features['avg_monthly_balance_min_6m'] = 0 # No transactions at all


    # Savings & Financial Prudence
    current_customer_features['avg_monthly_savings_transfers_6m'] = \
        txns_last_6m[txns_last_6m['description'] == 'Transfer to Savings']['amount'].sum() / 6 # Average over 6 months

    current_customer_features['has_investment_transactions_flag'] = \
        (txns_last_6m['description'] == 'Investment').any().astype(int)

    # Detect EMIs from bank transactions for the purpose of DTI calculation
    detected_emis_from_txns = txns_last_12m[txns_last_12m['description'] == 'Loan EMI Deduction']['amount'].sum()
    current_customer_features['detected_other_emis_sum'] = detected_emis_from_txns


    transaction_features.append(current_customer_features)

df_transaction_features = pd.DataFrame(transaction_features)
# Fill NaNs for customers with no transactions with zeros for numerical features
df_transaction_features = df_transaction_features.fillna(0) # For numerical values for customers with no transactions
print(f"Generated {len(df_transaction_features)} transaction feature sets.")
print(df_transaction_features.head())
print("\n" + "="*50 + "\n")


# --- 4. Merge All Features ---
print("Merging all features...")
# Start with applications data
df_final_features = df_applications.copy()

# Merge Credit Bureau features
# Use left merge to keep all applications; credit features will be NaN for those without history
df_final_features = pd.merge(df_final_features, credit_features_df, on='customer_id', how='left')

# Merge Transaction features
df_final_features = pd.merge(df_final_features, df_transaction_features, on='customer_id', how='left')

# Drop raw date column
df_final_features = df_final_features.drop(columns=['application_date'])

print("All features merged.")
print(df_final_features.head())
print(f"Final feature set shape: {df_final_features.shape}")
print("\n" + "="*50 + "\n")


# --- 5. Derive Composite Features ---
print("Deriving Composite Features...")

# Use the more reliable income source: if avg_monthly_net_income_6m is present and positive, use it, else use declared.
df_final_features['effective_monthly_income'] = df_final_features.apply(
    lambda row: row['avg_monthly_net_income_6m'] if row['avg_monthly_net_income_6m'] > 0 else row['monthly_gross_income'],
    axis=1
)

# Total monthly EMI obligations (emi_requested + declared existing EMIs + detected EMIs from bank transactions)
df_final_features['total_monthly_emi_obligations'] = df_final_features['emi_requested'] + df_final_features['other_emi_declared'] + df_final_features['detected_other_emis_sum']

# Re-calculate Debt-to-Income Ratio (DTI) with more robust income and expense data
df_final_features['debt_to_income_ratio_computed'] = df_final_features['total_monthly_emi_obligations'] / df_final_features['effective_monthly_income']
# Handle division by zero/infinity and assign a very high value (e.g., 999) if income is zero, indicating high risk
df_final_features['debt_to_income_ratio_computed'] = df_final_features['debt_to_income_ratio_computed'].replace([np.inf, -np.inf], np.nan).fillna(999)


# Residual Income
df_final_features['residual_income_after_emi'] = \
    df_final_features['effective_monthly_income'] - \
    df_final_features['avg_monthly_essential_expenses_6m'] - \
    df_final_features['total_monthly_emi_obligations']
df_final_features['residual_income_after_emi'] = df_final_features['residual_income_after_emi'].fillna(0) # Fill NaN with 0 if no expense/EMI data

print("Composite features derived.")
print(df_final_features[['customer_id', 'effective_monthly_income', 'total_monthly_emi_obligations', 'debt_to_income_ratio_computed', 'residual_income_after_emi']].head())
print("\n" + "="*50 + "\n")

# --- 6. Handle Missing Values (Final Pass) ---
print("Handling final missing values...")
# Fill NaN for credit bureau features for customers with no credit history
# 'credit_score' for has_credit_history=0 customers -> set to a specific indicator (e.g., 0)
df_final_features['credit_score'] = df_final_features['credit_score'].fillna(0) # Using 0 as indicator for no credit score

# For other numerical credit bureau features, fill with 0, indicating absence of such activity
credit_bureau_num_cols_to_fill_zero = [
    'num_credit_accounts_total', 'num_credit_accounts_active', 'oldest_account_age_months',
    'newest_account_age_months', 'total_sanctioned_limit', 'total_outstanding_balance',
    'overall_credit_utilization_ratio', 'avg_account_age_months', 'num_revolving_accounts_active',
    'num_defaults_reported_total', 'num_settled_accounts_total', 'max_dpd_past_24m',
    'ever_defaulted_flag', 'num_30_dpd_last_6m', 'num_90_dpd_last_12m'
]

for col in credit_bureau_num_cols_to_fill_zero:
    df_final_features[col] = df_final_features[col].fillna(0)

# Ensure 'loan_to_value_ratio' is handled for NaNs/Infs (e.e.g., if no collateral)
df_final_features['loan_to_value_ratio'] = df_final_features['loan_to_value_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Fill any remaining general numerical NaNs with 0 (can be refined based on feature meaning)
numerical_cols_after_merge = df_final_features.select_dtypes(include=np.number).columns
df_final_features[numerical_cols_after_merge] = df_final_features[numerical_cols_after_merge].fillna(0)


# Check for any remaining NaNs (for debugging)
print("NaNs after final pass (should be none or very few, if intentional):")
nan_counts = df_final_features.isnull().sum()
print(nan_counts[nan_counts > 0])
if nan_counts.sum() == 0:
    print("No NaNs remaining in numerical columns.")
print("\n" + "="*50 + "\n")


# --- 7. Categorical Feature Encoding ---
print("Encoding categorical features...")

# Identify categorical columns (objects/strings)
categorical_cols = df_final_features.select_dtypes(include='object').columns.tolist()

# Exclude identifier columns and any columns that are not meant for encoding
categorical_cols_to_encode = [col for col in categorical_cols if col not in ['customer_id', 'loan_id', 'pincode', 'current_delinquency_status']] # current_delinquency_status is not present in our data, but kept as a placeholder if you add it later.

# One-Hot Encoding for nominal categories
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_cols_to_encode)
    ],
    remainder='passthrough' # Keep all other columns (numerical, target, IDs, unencoded categoricals)
)

# Apply the transformation
df_encoded_features = preprocessor.fit_transform(df_final_features)

# Get the new column names for the one-hot encoded features
new_cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols_to_encode)

# Combine with original non-encoded column names (including numerical, IDs, and excluded categoricals)
# Order matters here: first new encoded names, then original columns that were passed through
original_passthrough_cols = [col for col in df_final_features.columns if col not in categorical_cols_to_encode]
df_final_features_encoded = pd.DataFrame(df_encoded_features, columns=list(new_cat_feature_names) + original_passthrough_cols)

print("Categorical features encoded.")
print(f"Shape after encoding: {df_final_features_encoded.shape}")
print(df_final_features_encoded.head())
print("\n" + "="*50 + "\n")

# --- 8. Feature Scaling (Numerical Features) ---
print("Scaling numerical features...")

# Identify numerical columns for scaling
# Exclude customer_id, loan_id, target variable 'is_default', and specific binary/categorical-like flags
numerical_cols = df_final_features_encoded.select_dtypes(include=np.number).columns.tolist()
features_to_exclude_from_scaling = [
    'customer_id', 'loan_id', 'is_default', 'has_credit_history', # Identifiers and target
    'collateral_flag', 'is_urban_residence', 'ever_defaulted_flag', # Binary flags
    'has_recurring_salary_flag', 'has_investment_transactions_flag', # Binary flags
    'num_dependents', # Often treated as ordinal or integer counts, not typically scaled
    'num_credit_accounts_total', 'num_credit_accounts_active', 'num_revolving_accounts_active', # Counts
    'num_defaults_reported_total', 'num_settled_accounts_total',
    'num_30_dpd_last_6m', 'num_90_dpd_last_12m', 'max_dpd_past_24m',
    'num_recurring_bill_payments_detected', 'num_overdrafts_6m', 'num_bounced_transactions_6m',
    'large_cash_withdrawal_frequency_6m', 'num_distinct_income_sources_6m' # Counts
]

numerical_cols_to_scale = [col for col in numerical_cols if col not in features_to_exclude_from_scaling]

scaler = StandardScaler()

# Create a pipeline for scaling only numerical features, leaving others as is
scaler_preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_cols_to_scale)
    ],
    remainder='passthrough' # Keep all other columns as they are
)

df_scaled_features_array = scaler_preprocessor.fit_transform(df_final_features_encoded)

# Get the names of columns after transformation (scaled numerical + passthrough)
scaled_feature_names = numerical_cols_to_scale
passthrough_feature_names = [col for col in df_final_features_encoded.columns if col not in numerical_cols_to_scale]

df_final_features_scaled = pd.DataFrame(df_scaled_features_array, columns=scaled_feature_names + passthrough_feature_names)

# Re-order columns to put identifiers first and target last for convenience
final_columns_ordered = [col for col in df_final_features_scaled.columns if col not in ['customer_id', 'loan_id', 'is_default', 'pincode']] # Removed 'current_delinquency_status' as it's not generated
df_final_features_ready = df_final_features_scaled[['customer_id', 'loan_id'] + final_columns_ordered + ['is_default']]

print("Numerical features scaled.")
print(f"Final dataset shape ready for modeling: {df_final_features_ready.shape}")
print(df_final_features_ready.head())
print("\n" + "="*50 + "\n")


# --- 9. Save Final Feature Set to Parquet ---
output_parquet_path = os.path.join(PROCESSED_DATA_DIR, 'engineered_features.parquet')
df_final_features_ready.to_parquet(output_parquet_path, index=False)
print(f"Final engineered features saved to Parquet: {output_parquet_path}")

print("\nFeature Engineering completed successfully! The processed features are ready in a Parquet file.")