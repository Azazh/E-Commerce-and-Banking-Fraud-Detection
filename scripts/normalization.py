import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define preprocessing pipeline
numeric_features = ['purchase_value', 'age']
categorical_features = ['source', 'browser', 'sex', 'country']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

engineered_data=pd.read_csv('../data/engineered_Fraud_Data.csv')


# Apply preprocessing
X = engineered_data.drop(columns=['class'])
y = engineered_data['class']
X_preprocessed = preprocessor.fit_transform(X)

# Save preprocessed data
pd.DataFrame(X_preprocessed).to_csv('../data/preprocessed_Fraud_Data.csv', index=False)