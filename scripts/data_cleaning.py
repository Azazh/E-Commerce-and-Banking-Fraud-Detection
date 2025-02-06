import pandas as pd

def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Convert timestamps to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Convert IP address to integer
    df['ip_address'] = df['ip_address'].astype('int64')

    return df

# Example usage
if __name__ == "__main__":
    fraud_data = pd.read_csv("../data/Fraud_Data.csv")
    fraud_data = clean_data(fraud_data)
    fraud_data.to_csv("../data/Fraud_Data_cleaned.csv", index=False)