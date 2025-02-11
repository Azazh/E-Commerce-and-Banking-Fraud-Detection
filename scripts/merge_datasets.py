import pandas as pd

def merge_with_geolocation(fraud_data, ip_address_data):
    """
    Merges fraud_data with geolocation data by mapping IP addresses to countries.
    """
    # Function to map IP address to country
    def get_country(ip_address):
        # Find matching rows in ip_address_data
        matching_rows = ip_address_data[
            (ip_address_data['lower_bound_ip_address'] <= ip_address) &
            (ip_address_data['upper_bound_ip_address'] >= ip_address)
        ]
        return matching_rows['country'].values[0] if not matching_rows.empty else 'Unknown'
    
    # Debugging: Print unique values in ip_address before conversion
    print("Unique values in ip_address before conversion:")
    print(fraud_data['ip_address'].unique()[:10])  # Show first 10 unique values
    
    # Convert ip_address in fraud_data to numeric format (if not already done)
    fraud_data['ip_address'] = pd.to_numeric(fraud_data['ip_address'], errors='coerce')
    
    # Drop rows with invalid or missing IP addresses
    fraud_data = fraud_data.dropna(subset=['ip_address'])
    
    # Ensure ip_address is of type Int64 (nullable integer)
    fraud_data['ip_address'] = fraud_data['ip_address'].astype('Int64', errors='ignore')
    
    # Debugging: Print unique values in ip_address after conversion
    print("Unique values in ip_address after conversion:")
    print(fraud_data['ip_address'].unique()[:10])  # Show first 10 unique values
    
    # Add country column by applying the get_country function
    fraud_data['country'] = fraud_data['ip_address'].apply(get_country)
    
    return fraud_data


# Load datasets
fraud_data = pd.read_csv('../data/cleaned_Fraud_Data.csv')
ip_address_data = pd.read_csv('../data/cleaned_IpAddress_to_Country.csv')

# Ensure ip_address_data columns are numeric
ip_address_data['lower_bound_ip_address'] = pd.to_numeric(ip_address_data['lower_bound_ip_address'], errors='coerce').astype('Int64', errors='ignore')
ip_address_data['upper_bound_ip_address'] = pd.to_numeric(ip_address_data['upper_bound_ip_address'], errors='coerce').astype('Int64', errors='ignore')

# Merge datasets
merged_data = merge_with_geolocation(fraud_data, ip_address_data)

# Debugging: Print the first few rows of the merged dataset
print("Merged Data:")
print(merged_data[['ip_address', 'country']].head())

# Save merged dataset
merged_data.to_csv('../data/merged_Fraud_Data_with_Geolocation.csv', index=False)