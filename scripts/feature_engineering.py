import pandas as pd

def engineer_features(df):
    """
    Engineers new features for fraud detection from the given DataFrame.
    """
    # Handle missing values
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    df.fillna(method='bfill', inplace=True)  # Backward fill remaining missing values
    
    # Ensure required columns are present
    if 'purchase_time' in df.columns:
        # Convert purchase_time to datetime format (if not already done)
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
        
        # Extract hour_of_day and day_of_week from purchase_time
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Calculate transaction frequency (if user_id exists)
    if 'user_id' in df.columns:
        df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
    
    # Drop original datetime columns if they are no longer needed
    df.drop(columns=['purchase_time'], errors='ignore', inplace=True)
    
    # Drop non-numeric columns (e.g., device_id, source, browser, etc.)
    df = df.select_dtypes(include=['number'])  # Keep only numeric columns
    
    return df


# Load merged dataset
merged_data = pd.read_csv('../data/merged_Fraud_Data_with_Geolocation.csv')

# Engineer features
engineered_data = engineer_features(merged_data)

# Debugging: Print the shape and first few rows of the engineered dataset
print("Engineered Data Shape:", engineered_data.shape)
print(engineered_data.head())

# Save engineered dataset
engineered_data.to_csv('../data/engineered_Fraud_Data.csv', index=False)