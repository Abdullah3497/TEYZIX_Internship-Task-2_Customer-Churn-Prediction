Customer Churn Prediction System
Machine Learning Internship Project

Author:
Abdullah Fayyaz

Task ID:
ML-2

------------------------------------------------------------
PROJECT OVERVIEW
------------------------------------------------------------

This project predicts whether a customer is likely to churn
(stop using the company's service) using Machine Learning.

The project includes:

• Self-created customer dataset
• Data preprocessing
• Feature engineering
• Exploratory Data Analysis (EDA)
• Machine Learning model training
• Model evaluation
• Streamlit prediction interface
• SHAP Explainable AI

The final prediction model used is XGBoost.

------------------------------------------------------------
PROJECT FILES
------------------------------------------------------------

Customer_Churn_Prediction.ipynb
    Complete notebook containing:
    - Dataset loading
    - Data cleaning
    - Feature engineering
    - Exploratory Data Analysis
    - Model training
    - Model evaluation

customer_churn_dataset.csv
    Synthetic customer dataset.

xgboost_model.pkl
    Trained XGBoost model.

app.py
    Streamlit application used to predict customer churn.

requirements.txt
    List of required Python packages.

Report.pdf
    Project report.

README.md / README.txt
    Project documentation.

------------------------------------------------------------
REQUIREMENTS
------------------------------------------------------------

Python 3.10 or later is recommended.

Install the required libraries by running:

pip install -r requirements.txt

------------------------------------------------------------
HOW TO RUN THE PROJECT
------------------------------------------------------------

Step 1

Open a terminal or command prompt.

Step 2

Navigate to the project folder.

Example:

cd Customer-Churn-Prediction

Step 3

Install dependencies:

pip install -r requirements.txt

Step 4

Run the Streamlit application:

streamlit run app.py

Step 5

Your web browser will automatically open the application.

------------------------------------------------------------
HOW TO USE THE APPLICATION
------------------------------------------------------------

1. Enter customer information.

2. Click the "Predict Churn" button.

3. The application will display:

• Churn Prediction
• Churn Probability
• Risk Level
• Feature Importance
• SHAP Explanation
• Business Recommendation

------------------------------------------------------------
MACHINE LEARNING MODELS
------------------------------------------------------------

Models trained:

• Random Forest
• XGBoost (Selected Model)

Final Model Accuracy:

99.25%

------------------------------------------------------------
FEATURES USED
------------------------------------------------------------

Age
Gender
City
Subscription Type
Monthly Spending
Tenure
Number of Purchases
Support Requests
Login Frequency
Last Activity Days
Satisfaction Score
Average Purchase
Engagement Score
Support Per Month

------------------------------------------------------------
PROJECT OBJECTIVE
------------------------------------------------------------

To help businesses identify customers who are likely to churn
so that appropriate retention strategies can be applied before
the customer leaves.

------------------------------------------------------------
THANK YOU
------------------------------------------------------------

Developed as part of the Machine Learning Internship Task (ML-2).