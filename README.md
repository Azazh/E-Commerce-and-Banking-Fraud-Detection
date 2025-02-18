### **README: Fraud Detection System**

---

#### **Overview**
This repository contains a comprehensive fraud detection system for e-commerce and banking transactions. The project includes data preprocessing, model building, explainability, deployment using Flask and Docker, and an interactive dashboard built with Dash. The system is designed to enhance transaction security, minimize financial losses, and provide transparency through interpretable models.

---

#### **Features**
1. **Data Preprocessing**:
   - Handles missing values by imputing or dropping them.
   - Cleans the data by removing duplicates and correcting data types.
   - Merges datasets for geolocation analysis (maps IP addresses to countries).
   - Engineers features such as transaction frequency, velocity, `hour_of_day`, `day_of_week`, and `time_since_signup`.

2. **Model Building and Training**:
   - Trains and evaluates multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, MLP, CNN, RNN, LSTM).
   - Uses stratified train-test splits for balanced evaluation.
   - Logs experiments, parameters, and metrics using **MLflow** for versioning and tracking.

3. **Model Explainability**:
   - Provides global and local explanations using **SHAP** and **LIME**.
   - Includes SHAP summary plots, force plots, and dependence plots.
   - LIME feature importance plots explain individual predictions.

4. **Deployment**:
   - Serves the trained model via a Flask API (`serve_model.py`).
   - Implements endpoints for predictions, fraud statistics, and geolocation insights.
   - Dockerizes the Flask application for scalable deployment.

5. **Dashboard**:
   - Built using **Dash**, the dashboard visualizes fraud insights in real-time.
   - Displays total transactions, fraud cases, and fraud percentages in summary boxes.
   - Includes a line chart showing fraud trends over time.
   - Analyzes geographical distribution of fraud cases.
   - Compares fraud cases across devices and browsers using bar charts.

---

#### **Directory Structure**
```
.
├── .vscode/
├── notebooks/
│   └── Exploratory Data Analysis.ipynb
├── scripts/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_explain.py
│   └── serve_model.py
├── src/
├── tests/
├── data/
│   ├── Fraud_Data.csv
│   ├── cleaned_Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── creditcard.csv
├── models/
│   ├── RandomForest_Fraud_Data.joblib
│   ├── XGBoost_CreditCard_Data.joblib
│   └── MLP_Fraud_Data.joblib
├── dashboard.py
├── requirements.txt
├── Dockerfile
├── README.md
└── api_logs.log
```

---

#### **Setup Instructions**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/fraud-detection-system.git
   cd fraud-detection-system
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start MLflow Server (Optional)**:
   If you plan to use MLflow for experiment tracking:
   ```bash
   mlflow server --backend-store-uri ./mlruns --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000
   ```

4. **Preprocess and Train Models**:
   Run the preprocessing and training scripts:
   ```bash
   python scripts/preprocess.py
   python scripts/model_training.py
   ```

5. **Build and Run Docker Container**:
   Build the Docker image:
   ```bash
   docker build -t fraud-detection-model .
   ```
   Run the Docker container:
   ```bash
   docker run -p 5000:5000 fraud-detection-model
   ```

6. **Start the Dashboard**:
   Run the Dash app to visualize fraud insights:
   ```bash
   python dashboard.py
   ```
   Access the dashboard at `http://localhost:8050`.

---

#### **Key Components**

1. **Data Preprocessing**:
   - Handles missing values, removes duplicates, and corrects data types.
   - Extracts time-based features (`hour_of_day`, `day_of_week`, `time_since_signup`) and merges geolocation data.

2. **Model Building**:
   - Trains and evaluates multiple models (Random Forest, XGBoost, Logistic Regression, etc.).
   - Logs experiments and metrics using MLflow.

3. **Model Explainability**:
   - Uses SHAP for global feature importance and local contribution analysis.
   - Employs LIME for interpretable explanations of individual predictions.

4. **Flask API**:
   - Serves the trained model for predictions.
   - Provides endpoints for fraud statistics (`/fraud-stats`) and geolocation insights (`/fraud-geolocation`).

5. **Dashboard**:
   - Built with Dash, it provides real-time visualization of fraud insights.
   - Includes summary statistics, fraud trends, geographical analysis, and device/browser comparisons.

---

#### **Endpoints**

1. **Prediction Endpoint**:
   - URL: `http://localhost:5000/predict`
   - Method: `POST`
   - Input: JSON object with transaction features.
   - Output: Prediction (`0` for non-fraud, `1` for fraud) and probability.

2. **Health Check Endpoint**:
   - URL: `http://localhost:5000/health`
   - Method: `GET`
   - Output: JSON response indicating the API's status.

3. **Fraud Statistics Endpoint**:
   - URL: `http://localhost:5000/fraud-stats`
   - Method: `GET`
   - Output: Summary statistics (total transactions, fraud cases, fraud percentage), fraud trends, and device/browser analysis.

4. **Geolocation Insights Endpoint**:
   - URL: `http://localhost:5000/fraud-geolocation`
   - Method: `GET`
   - Output: Geographical distribution of fraud cases.

---

#### **Usage**

1. **Run the Flask Backend**:
   Start the Flask API to serve predictions and fraud insights:
   ```bash
   python scripts/serve_model.py
   ```

2. **Start the Dash Frontend**:
   Launch the dashboard for real-time visualization:
   ```bash
   python dashboard.py
   ```

3. **Access the Dashboard**:
   Open your browser and navigate to `http://localhost:8050`.

4. **Test the API**:
   Use tools like `cURL` or Postman to test the `/predict`, `/fraud-stats`, and `/fraud-geolocation` endpoints.

---

#### **Results**

1. **Performance Metrics**:
   | Dataset         | Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
   |-----------------|---------------------|----------|-----------|--------|----------|---------|
   | Fraud_Data      | Random Forest       | 0.97     | 0.92      | 0.90   | 0.91     | 0.96    |
   | Fraud_Data      | XGBoost             | 0.98     | 0.93      | 0.91   | 0.92     | 0.97    |
   | CreditCard Data | XGBoost             | 0.99     | 0.95      | 0.93   | 0.94     | 0.98    |

2. **Explainability**:
   - SHAP summary plots highlight the most important features globally.
   - LIME explains individual predictions by approximating the model locally.

3. **Dashboard Insights**:
   - Total transactions, fraud cases, and fraud percentages displayed in summary boxes.
   - Line chart showing fraud trends over time.
   - Geographical distribution of fraud cases using a choropleth map.
   - Bar charts comparing fraud cases across devices and browsers.

---

#### **Technologies Used**
- **Python**: Core programming language.
- **Pandas**: Data manipulation and preprocessing.
- **Scikit-learn**: Machine learning models and evaluation.
- **XGBoost**: Gradient boosting framework for high-performance models.
- **SHAP & LIME**: Tools for model explainability.
- **Flask**: Backend API for serving predictions and insights.
- **Dash**: Frontend framework for building the interactive dashboard.
- **Docker**: Containerization for deployment.
- **MLflow**: Experiment tracking and model versioning.

---

#### **Contributing**
We welcome contributions to improve the fraud detection system. To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/new-feature
   ```
3. Commit your changes and push to your fork:
   ```bash
   git commit -m "Add new feature"
   git push origin feature/new-feature
   ```
4. Submit a pull request for review.

---

#### **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

#### **Acknowledgments**
- Thanks to Adey Innovations Inc. for providing the datasets and resources.
- Special thanks to the contributors of SHAP, LIME, MLflow, Flask, and Dash for their excellent tools.

---

#### **Contact**
For further inquiries or assistance, please contact:
- **Project Lead**: azazh wuletawu
- **Email**: azazhwuletawu@gmail.com
- **GitHub**: https://github.com/azazh

---

