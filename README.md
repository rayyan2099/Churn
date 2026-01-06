# Churn
📊 Customer Churn Prediction & Analysis
This project is an end-to-end Machine Learning solution designed to identify customers at high risk of churning (leaving the service). It includes a comprehensive data analysis pipeline and a live Streamlit web application for real-time predictions.

🚀 Live Demo
[https://customer-churn-prediction123.streamlit.app/]

🛠️ Tech Stack
Language: Python 3.10+

ML Frameworks: Scikit-learn, XGBoost, AdaBoost

Data Handling: Pandas, NumPy

Visualization: Seaborn, Matplotlib

Deployment: Streamlit

Sampling: IMBLearn (SMOTE)

📈 Key Features
Class Imbalance Handling: Implemented SMOTE to handle the minority churn class, ensuring the model doesn't just predict "No Churn" for everyone.

Model Comparison: Evaluated Random Forest, AdaBoost, and XGBoost to find the most robust classifier.

Full-Stack App: A clean Streamlit UI that takes customer details (tenure, contract type, monthly charges, etc.) and outputs a churn probability.

Automated Pipeline: Serialized encoders and models using pickle for consistent data transformation.

📊 Model Performance
The final model achieved the following metrics on the test set:

Accuracy: 79%

F1-Score (Churn Class): 0.58

Recall (Churn Class): 0.57

Key Insight: Contract type and Monthly Charges were identified as the strongest predictors of customer attrition.
