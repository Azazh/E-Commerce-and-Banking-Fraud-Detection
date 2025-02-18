import joblib
import pandas as pd
import shap
import lime
from lime import lime_tabular
import os
import matplotlib.pyplot as plt

# ------------------------------
# Load the Trained Model
# ------------------------------

# Specify the model path
model_path = '../models/RandomForest_Fraud_Data.joblib'  # Ensure no spaces in the file name

# Check if the file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the trained model
try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# ------------------------------
# Load Test Data
# ------------------------------

# Load test data
data_path = '../data/X_test_fraud.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Test data file not found at {data_path}")

X_test_fraud = pd.read_csv(data_path)

# Handle missing values in X_test_fraud
if X_test_fraud.isnull().any().any():
    print("Warning: Missing values detected in test data. Imputing with median.")
    X_test_fraud = X_test_fraud.fillna(X_test_fraud.median(), axis=0)

# Ensure feature names match the model's expected input
if not hasattr(model, "feature_names_in_") or not set(model.feature_names_in_).issubset(set(X_test_fraud.columns)):
    raise ValueError("Test data features do not match the model's expected input.")

# ------------------------------
# SHAP Explainability
# ------------------------------

# Initialize SHAP explainer
try:
    explainer = shap.TreeExplainer(model)  # Works well for tree-based models like Random Forest, XGBoost, etc.
    shap_values = explainer.shap_values(X_test_fraud)
except Exception as e:
    raise RuntimeError(f"Error initializing SHAP explainer: {e}")

# SHAP values for the Fraud class (class 1)
shap_values_fraud = shap_values[1] if isinstance(shap_values, list) else shap_values

# Summary Plot
try:
    shap.summary_plot(shap_values_fraud, X_test_fraud, plot_type="bar", show=False)
    plt.title("Feature Importance (SHAP Summary Plot)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error generating SHAP summary plot: {e}")

# Force Plot for a single instance
instance_index = 0
if instance_index >= len(X_test_fraud):
    raise IndexError(f"Instance index {instance_index} out of range. Test data has {len(X_test_fraud)} rows.")

try:
    shap.initjs()
    shap.force_plot(
        explainer.expected_value[1], 
        shap_values_fraud[instance_index, :], 
        X_test_fraud.iloc[instance_index, :], 
        matplotlib=True
    )
except Exception as e:
    print(f"Error generating SHAP force plot: {e}")

# Dependence Plot for a specific feature
feature_name = 'purchase_value'
if feature_name not in X_test_fraud.columns:
    raise KeyError(f"Feature '{feature_name}' not found in test data.")

try:
    shap.dependence_plot(
        feature_name, 
        shap_values_fraud, 
        X_test_fraud, 
        interaction_index=None, 
        show=False
    )
    plt.title(f"SHAP Dependence Plot for {feature_name}")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error generating SHAP dependence plot: {e}")

# ------------------------------
# LIME Explainability
# ------------------------------

# Create a LIME explainer for tabular data
try:
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_test_fraud.values,
        feature_names=X_test_fraud.columns,
        class_names=['Non-Fraud', 'Fraud'],
        mode='classification'
    )
except Exception as e:
    raise RuntimeError(f"Error initializing LIME explainer: {e}")

# Select a specific instance to explain
instance_index = 0
if instance_index >= len(X_test_fraud):
    raise IndexError(f"Instance index {instance_index} out of range. Test data has {len(X_test_fraud)} rows.")

instance = X_test_fraud.iloc[instance_index].values

# Generate LIME explanation
try:
    exp = explainer.explain_instance(instance, model.predict_proba, num_features=5)
except Exception as e:
    raise RuntimeError(f"Error generating LIME explanation: {e}")

# Show the explanation as a table
print("\nLIME Explanation (Top Features):")
print(exp.as_list())

# Show the explanation as a plot
try:
    exp.show_in_notebook(show_table=True, show_all=False)
except Exception as e:
    print(f"Error displaying LIME plot in notebook: {e}")

# Save the LIME explanation as an HTML file
try:
    exp.save_to_file('lime_explanation.html')
    print("LIME explanation saved to 'lime_explanation.html'.")
except Exception as e:
    print(f"Error saving LIME explanation: {e}")