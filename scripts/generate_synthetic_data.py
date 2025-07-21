import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker with an Indian locale for more realistic names/addresses
fake = Faker('en_IN')

# --- Configuration ---
NUM_CUSTOMERS = 5000
NUM_CREDIT_BUREAU_ACCOUNTS_PER_CUSTOMER = 1 # Avg, some will have more, some less
NUM_BANK_TRANSACTIONS_PER_CUSTOMER_PER_MONTH = 20 # Avg, varies
DATA_START_DATE_APPLICATIONS = datetime(2023, 1, 1)
DATA_END_DATE_APPLICATIONS = datetime(2024, 12, 31) # Up to end of last year for historical data
BANK_TRANSACTION_HISTORY_MONTHS = 12 # Generate transactions for last 12 months

# --- Helper Functions ---

def generate_customer_id(index):
    return f"CUST{index:05d}"

def generate_loan_id(index):
    return f"LOAN{index:05d}"

def get_random_date(start_date, end_date):
    return start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

def calculate_emi(principal, annual_interest_rate, loan_term_months):
    """Calculates EMI using the reducing balance method."""
    if annual_interest_rate == 0:
        return principal / loan_term_months
    monthly_interest_rate = annual_interest_rate / 12 / 100
    if monthly_interest_rate == 0: # Handle cases where rate is extremely small but not zero
        return principal / loan_term_months
    emi = principal * monthly_interest_rate * (1 + monthly_interest_rate)**loan_term_months / ((1 + monthly_interest_rate)**loan_term_months - 1)
    return emi

# --- 1. Generate Loan Applications Data ---
print("Generating Loan Applications Data...")
applications_data = []
credit_history_flag_distribution = [0] * int(NUM_CUSTOMERS * 0.3) + [1] * int(NUM_CUSTOMERS * 0.7) # ~30% new-to-credit, 70% established
random.shuffle(credit_history_flag_distribution)

for i in range(NUM_CUSTOMERS):
    customer_id = generate_customer_id(i)
    loan_id = generate_loan_id(i)
    application_date = get_random_date(DATA_START_DATE_APPLICATIONS, DATA_END_DATE_APPLICATIONS)
    
    # Demographics
    dob = get_random_date(datetime(1960, 1, 1), datetime(2003, 1, 1)) # Age 22-65
    age = (application_date - dob).days // 365
    gender = random.choice(['Male', 'Female', 'Other'])
    marital_status = random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
    education_level = random.choice(['High School', 'Graduate', 'Post-Graduate', 'Doctorate'])
    num_dependents = random.randint(0, 5)
    residential_status = random.choice(['Owned', 'Rented', 'Parental', 'Mortgaged'])
    time_at_current_address_months = random.randint(6, 240) # 0.5 to 20 years
    pincode = fake.postcode() # Indian postcode simulation
    is_urban_residence = 1 if int(pincode[0]) % 2 == 0 else 0 # Simple heuristic for urban/rural

    # Employment & Income
    employment_type = random.choice(['Salaried', 'Self-employed', 'Unemployed', 'Retired'])
    time_at_current_job_months = random.randint(12, 360) if employment_type != 'Unemployed' else 0
    
    monthly_gross_income = 0
    if employment_type == 'Salaried':
        monthly_gross_income = round(random.uniform(25000, 150000), 2)
    elif employment_type == 'Self-employed':
        monthly_gross_income = round(random.uniform(30000, 200000), 2)
    elif employment_type == 'Retired':
        monthly_gross_income = round(random.uniform(10000, 50000), 2)
    
    declared_monthly_expenses = round(monthly_gross_income * random.uniform(0.3, 0.7), 2) # Assume some % of income

    # Loan Details
    loan_amount_requested = round(random.uniform(50000, 1000000), 0)
    loan_term_months = random.choice([12, 24, 36, 48, 60])
    loan_purpose = random.choice(['Personal Loan', 'Home Loan', 'Auto Loan', 'Education Loan', 'Business Loan', 'Credit Card'])
    
    # Simple interest rate assumption for EMI calculation, will vary by risk in model
    assumed_annual_interest_rate = 12 + random.uniform(-3, 3) 
    emi_requested = round(calculate_emi(loan_amount_requested, assumed_annual_interest_rate, loan_term_months), 2)

    collateral_flag = 0
    collateral_value = 0
    if loan_purpose in ['Home Loan', 'Auto Loan', 'Business Loan']:
        collateral_flag = random.choice([0, 1]) # Some loans might not have explicit collateral for simplicity
        if collateral_flag == 1:
            collateral_value = round(loan_amount_requested * random.uniform(1.0, 2.5), 0) # Collateral > Loan Amt
            if loan_purpose == 'Home Loan':
                collateral_value = round(random.uniform(1500000, 10000000), 0) # Higher value for home loan
    
    loan_to_value_ratio = loan_amount_requested / collateral_value if collateral_value > 0 else 0
    
    other_emi_declared = round(monthly_gross_income * random.uniform(0, 0.2), 2) # Other existing EMIs
    debt_service_ratio_declared = (emi_requested + other_emi_declared) / monthly_gross_income if monthly_gross_income > 0 else 999
    
    # Credit History Flag - Crucial for our segmented approach
    has_credit_history = credit_history_flag_distribution.pop() # Assign based on pre-shuffled list

    # Simulate default status (target variable)
    # This is a simplified logic. Real models would predict this.
    # Higher risk factors increase default probability.
    default_prob = 0.05 # Baseline default probability
    if monthly_gross_income < 30000: default_prob += 0.05
    if age < 25: default_prob += 0.03
    if employment_type == 'Unemployed': default_prob += 0.2
    if debt_service_ratio_declared > 0.6: default_prob += 0.07
    if has_credit_history == 0: default_prob += 0.05 # New to credit is inherently higher risk

    if has_credit_history == 1:
        # Simulate impact of credit score for established borrowers
        temp_credit_score = random.randint(300, 900)
        if temp_credit_score < 550: default_prob += 0.15 # Very low score
        elif temp_credit_score < 650: default_prob += 0.08
        elif temp_credit_score < 750: default_prob += 0.02
        else: default_prob -= 0.02 # Good score
    
    is_default = 1 if random.random() < default_prob else 0 # 1 for default, 0 for non-default

    applications_data.append({
        'customer_id': customer_id,
        'loan_id': loan_id,
        'application_date': application_date.strftime('%Y-%m-%d'),
        'age': age,
        'gender': gender,
        'marital_status': marital_status,
        'education_level': education_level,
        'num_dependents': num_dependents,
        'residential_status': residential_status,
        'time_at_current_address_months': time_at_current_address_months,
        'pincode': pincode,
        'is_urban_residence': is_urban_residence,
        'employment_type': employment_type,
        'time_at_current_job_months': time_at_current_job_months,
        'monthly_gross_income': monthly_gross_income,
        'declared_monthly_expenses': declared_monthly_expenses,
        'loan_amount_requested': loan_amount_requested,
        'loan_term_months': loan_term_months,
        'loan_purpose': loan_purpose,
        'emi_requested': emi_requested,
        'collateral_flag': collateral_flag,
        'collateral_value': collateral_value,
        'loan_to_value_ratio': loan_to_value_ratio,
        'other_emi_declared': other_emi_declared,
        'debt_service_ratio_declared': debt_service_ratio_declared,
        'has_credit_history': has_credit_history, # Crucial flag
        'is_default': is_default # Target variable
    })

df_applications = pd.DataFrame(applications_data)
df_applications.to_csv('data/raw/applications/loan_applications_5k.csv', index=False)
print(f"Generated {len(df_applications)} loan applications.")
print(df_applications.head())
print("\n" + "="*50 + "\n")

# --- 2. Generate Credit Bureau Data (for customers with has_credit_history = 1) ---
print("Generating Credit Bureau Data...")
credit_bureau_data = []
customers_with_credit_history = df_applications[df_applications['has_credit_history'] == 1]['customer_id'].tolist()

for customer_id in customers_with_credit_history:
    credit_score = random.randint(300, 900)
    num_accounts = random.randint(1, 10)
    
    # Simulate historical accounts for this customer
    for j in range(random.randint(1, num_accounts)):
        account_id = f"ACC{customer_id.replace('CUST','')}{j:03d}"
        account_open_date = get_random_date(datetime(2000, 1, 1), datetime(2023, 6, 1)) # Accounts opened before application date
        account_type = random.choice(['Credit Card', 'Personal Loan', 'Home Loan', 'Auto Loan'])
        sanctioned_limit = round(random.uniform(50000, 2000000), 2) if account_type != 'Credit Card' else round(random.uniform(10000, 500000), 2)
        current_balance = round(random.uniform(0, sanctioned_limit * random.uniform(0.1, 0.9)), 2) # Balance is a % of limit
        
        # Simulate DPD status
        dpd_status = 'Current'
        max_dpd_past_24m = 0
        num_30_dpd = 0
        num_90_dpd = 0
        num_defaults = 0
        num_settled = 0

        # Based on credit score, simulate payment history
        if credit_score < 550: # Bad score
            if random.random() < 0.6: dpd_status = random.choice(['30_DPD', '60_DPD', '90_DPD'])
            if random.random() < 0.3: num_30_dpd = random.randint(1, 3)
            if random.random() < 0.1: num_90_dpd = random.randint(1, 2)
            if random.random() < 0.05: num_defaults = 1
            if random.random() < 0.02: num_settled = 1
            max_dpd_past_24m = random.randint(30, 180) if dpd_status != 'Current' else 0
        elif credit_score < 700: # Average score
            if random.random() < 0.2: dpd_status = random.choice(['30_DPD', '60_DPD'])
            if random.random() < 0.05: num_30_dpd = 1
            max_dpd_past_24m = random.randint(0, 60) if dpd_status != 'Current' else 0
        else: # Good score
            dpd_status = 'Current'
            max_dpd_past_24m = 0
        
        account_status = 'Active' if dpd_status == 'Current' else 'Delinquent'
        if num_defaults > 0: account_status = 'Defaulted'
        if num_settled > 0: account_status = 'Settled'
        if random.random() < 0.1: account_status = 'Closed' # Simulate some closed accounts

        credit_bureau_data.append({
            'customer_id': customer_id,
            'credit_score': credit_score,
            'account_id': account_id,
            'account_open_date': account_open_date.strftime('%Y-%m-%d'),
            'account_type': account_type,
            'sanctioned_limit': sanctioned_limit,
            'current_balance': current_balance,
            'dpd_status': dpd_status,
            'max_dpd_past_24m': max_dpd_past_24m,
            'num_30_dpd_last_6m': num_30_dpd,
            'num_90_dpd_last_12m': num_90_dpd,
            'num_defaults_total': num_defaults,
            'num_settled_total': num_settled,
            'account_status': account_status,
            'inquiry_date': get_random_date(application_date - timedelta(days=365), application_date).strftime('%Y-%m-%d') # Simulate inquiries around application date
        })

df_credit_bureau = pd.DataFrame(credit_bureau_data)
df_credit_bureau.to_csv('data/raw/credit_bureau_reports/credit_bureau_data_5k.csv', index=False)
print(f"Generated {len(df_credit_bureau)} credit bureau records.")
print(df_credit_bureau.head())
print("\n" + "="*50 + "\n")


# --- 3. Generate Bank Transactions Data (for all customers, especially new-to-credit) ---
print("Generating Bank Transactions Data...")
bank_transactions_data = []

# Define categories for transactions
transaction_categories = {
    'income': ['Salary', 'Freelance Income', 'Bonus', 'Rent Income'],
    'essential_expenses': ['Rent Payment', 'Electricity Bill', 'Water Bill', 'Mobile Recharge', 'Groceries', 'Public Transport', 'EMI Payment'],
    'discretionary_expenses': ['Dining Out', 'Shopping', 'Entertainment', 'Travel', 'Online Subscription', 'Fuel'],
    'transfers': ['Transfer to Savings', 'UPI Transfer to Friend', 'ATM Withdrawal', 'Investment'],
    'loan_disbursement': ['Loan Disbursement'] # For loan payouts
}

# Define typical patterns for income and bill payments
salary_days = [1, 5, 10, 15, 20, 25] # Days of month for salary
rent_days = [1, 5]
emi_days = [5, 10, 15]

for i, row in df_applications.iterrows():
    customer_id = row['customer_id']
    application_date = pd.to_datetime(row['application_date'])

    # Generate transactions for the last X months leading up to the application date
    for month_offset in range(BANK_TRANSACTION_HISTORY_MONTHS, 0, -1):
        current_month_start = application_date - pd.DateOffset(months=month_offset)
        current_month_end = application_date - pd.DateOffset(months=month_offset - 1) - timedelta(days=1)
        
        if current_month_start.month == application_date.month and current_month_start.year == application_date.year:
            # For the current application month, only generate up to the application date
            current_month_end = application_date 

        # Simulate salary/income
        if row['employment_type'] in ['Salaried', 'Retired'] and random.random() < 0.9: # 90% chance of consistent salary
            salary_day = random.choice(salary_days)
            if current_month_start.day <= salary_day <= current_month_end.day:
                transaction_date = current_month_start.replace(day=salary_day)
                amount = round(row['monthly_gross_income'] * random.uniform(0.95, 1.05), 2) # Slight variation
                bank_transactions_data.append({
                    'customer_id': customer_id,
                    'transaction_id': fake.uuid4(),
                    'transaction_date': transaction_date.strftime('%Y-%m-%d'),
                    'amount': amount,
                    'type': 'credit',
                    'description': random.choice(transaction_categories['income']),
                    'category': 'Income'
                })
        elif row['employment_type'] == 'Self-employed' and random.random() < 0.7: # Self-employed less consistent
            num_income_txns = random.randint(1, 3)
            for _ in range(num_income_txns):
                transaction_date = get_random_date(current_month_start, current_month_end)
                amount = round(row['monthly_gross_income'] * random.uniform(0.3, 0.7), 2) # Smaller, multiple incomes
                bank_transactions_data.append({
                    'customer_id': customer_id,
                    'transaction_id': fake.uuid4(),
                    'transaction_date': transaction_date.strftime('%Y-%m-%d'),
                    'amount': amount,
                    'type': 'credit',
                    'description': random.choice(transaction_categories['income']),
                    'category': 'Income'
                })
        
        # Simulate recurring bill payments (Rent, EMIs)
        if random.random() < 0.7: # Chance of having rent
            rent_day = random.choice(rent_days)
            if current_month_start.day <= rent_day <= current_month_end.day:
                transaction_date = current_month_start.replace(day=rent_day)
                amount = round(row['monthly_gross_income'] * random.uniform(0.2, 0.4), 2) # Rent is a significant chunk
                bank_transactions_data.append({
                    'customer_id': customer_id,
                    'transaction_id': fake.uuid4(),
                    'transaction_date': transaction_date.strftime('%Y-%m-%d'),
                    'amount': -amount, # Debit
                    'type': 'debit',
                    'description': 'Rent Payment',
                    'category': 'Essential'
                })
        
        if random.random() < 0.6 and row['other_emi_declared'] > 0: # Chance of having other EMIs
            emi_day = random.choice(emi_days)
            if current_month_start.day <= emi_day <= current_month_end.day:
                transaction_date = current_month_start.replace(day=emi_day)
                amount = round(row['other_emi_declared'] * random.uniform(0.9, 1.1), 2) # Close to declared EMI
                bank_transactions_data.append({
                    'customer_id': customer_id,
                    'transaction_id': fake.uuid4(),
                    'transaction_date': transaction_date.strftime('%Y-%m-%d'),
                    'amount': -amount, # Debit
                    'type': 'debit',
                    'description': 'Loan EMI Deduction',
                    'category': 'Essential'
                })

        # Simulate other expenses
        num_transactions = random.randint(NUM_BANK_TRANSACTIONS_PER_CUSTOMER_PER_MONTH - 5, NUM_BANK_TRANSACTIONS_PER_CUSTOMER_PER_MONTH + 5)
        for _ in range(num_transactions):
            transaction_date = get_random_date(current_month_start, current_month_end)
            
            # Ensure transactions don't go past application date
            if transaction_date > application_date:
                continue

            category_type = random.choices(['essential_expenses', 'discretionary_expenses', 'transfers'], weights=[0.4, 0.4, 0.2], k=1)[0]
            description = random.choice(transaction_categories[category_type])
            
            amount = round(random.uniform(50, 5000), 2) # Varying amounts
            if description == 'ATM Withdrawal': amount = round(random.uniform(1000, 10000), 2)
            if description == 'Investment': amount = round(random.uniform(500, 20000), 2)
            if description == 'UPI Transfer to Friend': amount = round(random.uniform(10, 5000), 2)

            # Simulate overdrafts/bounced transactions for riskier profiles
            is_risky = (row['is_default'] == 1) or (row['has_credit_history'] == 0 and random.random() < 0.3)
            if is_risky and random.random() < 0.02: # 2% chance of overdraft/bounce for risky
                 description = random.choice(['Overdraft Fee', 'Cheque Bounce Charge'])
                 amount = random.choice([250, 500, 750]) # Fixed charges
                 bank_transactions_data.append({
                    'customer_id': customer_id,
                    'transaction_id': fake.uuid4(),
                    'transaction_date': transaction_date.strftime('%Y-%m-%d'),
                    'amount': -amount, # Debit
                    'type': 'debit',
                    'description': description,
                    'category': 'Fee'
                })
                 continue # Skip normal expense for this iteration to make room for fee

            bank_transactions_data.append({
                'customer_id': customer_id,
                'transaction_id': fake.uuid4(),
                'transaction_date': transaction_date.strftime('%Y-%m-%d'),
                'amount': -amount, # Debit
                'type': 'debit',
                'description': description,
                'category': category_type.replace('_expenses', '').capitalize() if 'expenses' in category_type else category_type.capitalize()
            })

df_bank_transactions = pd.DataFrame(bank_transactions_data)
# Sort by customer and date for easier processing later
df_bank_transactions = df_bank_transactions.sort_values(by=['customer_id', 'transaction_date'])
df_bank_transactions.to_csv('data/raw/bank_transactions/bank_transactions_5k.csv', index=False)
print(f"Generated {len(df_bank_transactions)} bank transactions.")
print(df_bank_transactions.head())
print("\n" + "="*50 + "\n")

print("All synthetic datasets generated successfully!")