import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
import pandas as pd

# Set the tracking URI programmatically
mlflow.set_tracking_uri('http://localhost:5000')

# Load prepared datasets for Fraud_Data
X_train_fraud = pd.read_csv('../data/X_train_fraud.csv')
X_test_fraud = pd.read_csv('../data/X_test_fraud.csv')
y_train_fraud = pd.read_csv('../data/y_train_fraud.csv', header=None).squeeze()  # Squeeze to convert to Series
y_test_fraud = pd.read_csv('../data/y_test_fraud.csv', header=None).squeeze()   # Squeeze to convert to Series

# Debugging: Print shapes of datasets
print("Shapes of datasets after loading:")
print(f"X_train_fraud shape: {X_train_fraud.shape}, y_train_fraud shape: {y_train_fraud.shape}")
print(f"X_test_fraud shape: {X_test_fraud.shape}, y_test_fraud shape: {y_test_fraud.shape}")

# Ensure consistent shapes between features and labels
if X_train_fraud.shape[0] != y_train_fraud.shape[0]:
    raise ValueError("Mismatch in number of samples between X_train_fraud and y_train_fraud.")
if X_test_fraud.shape[0] != y_test_fraud.shape[0]:
    raise ValueError("Mismatch in number of samples between X_test_fraud and y_test_fraud.")

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "MLP": MLPClassifier(max_iter=500)
}

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, dataset_name):
    for model_name, model in models.items():
        print(f"Training {model_name} on {dataset_name}...")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Evaluate metrics
                report = classification_report(y_test, y_pred, output_dict=True)
                auc = roc_auc_score(y_test, y_prob)
                
                # Log metrics to MLflow
                mlflow.log_param("model", model_name)
                mlflow.log_metric("accuracy", report['accuracy'])
                mlflow.log_metric("precision", report['1']['precision'])
                mlflow.log_metric("recall", report['1']['recall'])
                mlflow.log_metric("f1_score", report['1']['f1-score'])
                mlflow.log_metric("roc_auc", auc)
                
                # Log model
                mlflow.sklearn.log_model(model, f"{model_name}_model")
                
                # Save model locally
                joblib.dump(model, f'../models/{model_name}_{dataset_name}.joblib')
            
            except Exception as e:
                print(f"Error during training {model_name}: {e}")

# Train models for Fraud_Data
train_and_evaluate(X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud, 'Fraud_Data')

# Train models for CreditCard Data (if available)
train_and_evaluate(X_train_credit, X_test_credit, y_train_credit, y_test_credit, 'CreditCard_Data')