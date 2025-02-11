import pandas as pd

def handle_missing_values(df):
    print("Missing values before handling:\n", df.isnull().sum())
    
    # Impute numerical features with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Impute categorical features with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("Missing values after handling:\n", df.isnull().sum())
    return df


def clean_data(df):
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Correct data types for datetime columns
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
    
    # Convert IP address to integer format (if present)
    if 'ip_address' in df.columns:
        df['ip_address'] = pd.to_numeric(df['ip_address'], errors='coerce').astype('Int64', errors='ignore')
        
    return df


def preprocess_fraud_data(df):
    """
    Preprocesses the Fraud_Data.csv dataset by extracting time-based features
    and dropping unnecessary columns.
    """
    # Extract hour_of_day and day_of_week from purchase_time
    if 'purchase_time' in df.columns:
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Calculate time_since_signup in seconds
    if 'signup_time' in df.columns and 'purchase_time' in df.columns:
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    
    # Drop original datetime columns if they are no longer needed
    df.drop(columns=['signup_time', 'purchase_time'], errors='ignore', inplace=True)
    
    return df


# Load datasets
fraud_data = pd.read_csv('../data/Fraud_Data.csv')
ip_address_data = pd.read_csv('../data/IpAddress_to_Country.csv')
creditcard_data = pd.read_csv('../data/creditcard.csv')

# Handle missing values
fraud_data = handle_missing_values(fraud_data)
ip_address_data = handle_missing_values(ip_address_data)
creditcard_data = handle_missing_values(creditcard_data)

# Clean datasets
fraud_data = clean_data(fraud_data)
ip_address_data = clean_data(ip_address_data)
creditcard_data = clean_data(creditcard_data)

# Preprocess Fraud_Data.csv
fraud_data = preprocess_fraud_data(fraud_data)

# Verify the new features
print("Preprocessed Fraud Data:")
print(fraud_data[['hour_of_day', 'day_of_week', 'time_since_signup']].head())

# Save cleaned datasets
fraud_data.to_csv('../data/cleaned_Fraud_Data.csv', index=False)
ip_address_data.to_csv('../data/cleaned_IpAddress_to_Country.csv', index=False)
creditcard_data.to_csv('../data/cleaned_creditcard.csv', index=False)