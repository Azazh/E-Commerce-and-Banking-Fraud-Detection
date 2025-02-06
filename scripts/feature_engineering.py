import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def feature_engineering(df):
    # Time-Based Features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

    # Normalization
    scaler = StandardScaler()
    df['purchase_value'] = scaler.fit_transform(df[['purchase_value']])

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['source', 'browser', 'sex'], drop_first=True)

    return df

# Example usage
if __name__ == "__main__":
    fraud_data = pd.read_csv("../data/Fraud_Data_with_country.csv")
    fraud_data = feature_engineering(fraud_data)
    fraud_data.to_csv("../data/Fraud_Data_engineered.csv", index=False)