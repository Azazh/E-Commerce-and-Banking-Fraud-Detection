import pandas as pd

def handle_missing_values(df):
    # Impute numerical features with median
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())

    # Impute categorical features with mode
    categorical_features = df.select_dtypes(include=['object']).columns
    df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

    return df

# Example usage
if __name__ == "__main__":
    fraud_data = pd.read_csv("../data/Fraud_Data.csv")
    fraud_data = handle_missing_values(fraud_data)
    fraud_data.to_csv("../data/Fraud_Data_cleaned.csv", index=False)