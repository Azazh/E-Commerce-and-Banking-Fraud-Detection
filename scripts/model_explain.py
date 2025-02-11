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
model = joblib.load(model_path)

# ------------------------------
# Load Test Data
# ------------------------------

# Load test data
X_test_fraud = pd.read_csv('../data/X_test_fraud.csv')

# Ensure no missing values in X_test_fraud
X_test_fraud = X_test_fraud.fillna(X_test_fraud.median(), axis=0)

# ------------------------------
# SHAP Explainability
# ------------------------------

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)  # Works well for tree-based models like Random Forest, XGBoost, etc.
shap_values = explainer.shap_values(X_test_fraud)

# SHAP values for the Fraud class
shap_values_fraud = shap_values[1]

# Summary Plot
shap.summary_plot(shap_values_fraud, X_test_fraud, plot_type="bar", show=False)
plt.title("Feature Importance (SHAP Summary Plot)")
plt.show()

# Force Plot for a single instance
instance_index = 0
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values_fraud[instance_index, :], X_test_fraud.iloc[instance_index, :])

# Dependence Plot for a specific feature
feature_name = 'purchase_value'
shap.dependence_plot(feature_name, shap_values_fraud, X_test_fraud, interaction_index=None, show=False)
plt.title(f"SHAP Dependence Plot for {feature_name}")
plt.show()

# ------------------------------
# LIME Explainability
# ------------------------------

# Create a LIME explainer for tabular data
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_test_fraud.values,
    feature_names=X_test_fraud.columns,
    class_names=['Non-Fraud', 'Fraud'],
    mode='classification'
)

# Select a specific instance to explain
instance_index = 0
instance = X_test_fraud.iloc[instance_index].values

# Generate LIME explanation
exp = explainer.explain_instance(instance, model.predict_proba, num_features=5)

# Show the explanation as a table
print(exp.as_list())

# Show the explanation as a plot
exp.show_in_notebook(show_table=True, show_all=False)

# Save the LIME explanation as an HTML file
exp.save_to_file('lime_explanation.html')