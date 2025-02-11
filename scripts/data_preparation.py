import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(df, target_col):
    """
    Prepares the dataset by separating features and target,
    handling missing values, and performing a stratified train-test split.
    """
    # Verify that the target column exists in the dataset
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle missing values in X and y
    X = X.fillna(X.median(), axis=0)  # Impute missing values in features with median
    y = y.fillna(y.mode()[0])         # Impute missing values in target with mode
    
    # Ensure no rows are dropped without aligning X and y
    X = X.reset_index(drop=True)      # Reset index to align with y
    y = y.reset_index(drop=True)      # Reset index to align with X
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Debugging: Print the shapes of the resulting datasets
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


# Load datasets
fraud_data = pd.read_csv('../data/engineered_Fraud_Data.csv')
creditcard_data = pd.read_csv('../data/cleaned_creditcard.csv')

# Prepare Fraud_Data
try:
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = prepare_data(fraud_data, 'class')
except ValueError as e:
    print(e)

# Prepare CreditCard Data
try:
    X_train_credit, X_test_credit, y_train_credit, y_test_credit = prepare_data(creditcard_data, 'Class')
except ValueError as e:
    print(e)

# Save prepared datasets for Fraud_Data
X_train_fraud.to_csv('../data/X_train_fraud.csv', index=False)
X_test_fraud.to_csv('../data/X_test_fraud.csv', index=False)
y_train_fraud.to_csv('../data/y_train_fraud.csv', index=False, header=False)
y_test_fraud.to_csv('../data/y_test_fraud.csv', index=False, header=False)

# Save prepared datasets for CreditCard Data
X_train_credit.to_csv('../data/X_train_credit.csv', index=False)
X_test_credit.to_csv('../data/X_test_credit.csv', index=False)
y_train_credit.to_csv('../data/y_train_credit.csv', index=False, header=False)
y_test_credit.to_csv('../data/y_test_credit.csv', index=False, header=False)