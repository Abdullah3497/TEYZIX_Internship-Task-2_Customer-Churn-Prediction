# Customer Churn Prediction System

## Machine Learning Internship Project

### Overview

The **Customer Churn Prediction System** is a machine learning application developed to predict whether a customer is likely to stop using a company's services. The project follows a complete machine learning workflow including data generation, preprocessing, feature engineering, model training, evaluation, and deployment through a Streamlit web application.

---

## Features

* Synthetic dataset containing 2,000 realistic customer records
* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model comparison using Random Forest and XGBoost
* Customer churn prediction
* Churn probability estimation
* Feature Importance Analysis
* SHAP Explainable AI
* Interactive Streamlit web application

---

## Dataset Features

The dataset contains the following attributes:

* CustomerID
* Age
* Gender
* City
* SubscriptionType
* MonthlySpending
* Tenure
* NumberOfPurchases
* SupportRequests
* LoginFrequency
* LastActivityDays
* SatisfactionScore
* AveragePurchase (Engineered)
* EngagementScore (Engineered)
* SupportPerMonth (Engineered)
* Churn (Target Variable)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* SHAP
* Matplotlib
* Joblib

---

## Machine Learning Models

Two classification models were trained and evaluated:

### Random Forest

* Accuracy: **98.75%**
* Precision: **98.43%**
* Recall: **97.66%**
* F1 Score: **98.04%**
* ROC-AUC: **99.82%**

### XGBoost (Selected Model)

* Accuracy: **99.25%**
* Precision: **99.21%**
* Recall: **98.44%**
* F1 Score: **98.82%**
* ROC-AUC: **99.89%**

XGBoost achieved the highest performance and was selected as the final prediction model.

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── app.py
├── customer_churn_dataset.csv
├── xgboost_model.pkl
├── requirements.txt
├── README.md
├── Report.pdf
└── screenshots/
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd Customer-Churn-Prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## Streamlit Application Features

* Customer information input form
* Churn prediction
* Churn probability display
* Risk level indicator
* Feature Importance visualization
* SHAP Explainable AI
* Business recommendations

---

## Key Findings

* Customer satisfaction strongly influences churn.
* High support requests increase churn probability.
* Longer inactivity leads to greater churn risk.
* Customer engagement reduces churn.
* Longer tenure generally improves customer retention.

---

## Business Recommendations

* Identify high-risk customers early.
* Improve customer support quality.
* Increase customer engagement through personalized campaigns.
* Offer loyalty rewards to long-term customers.
* Provide targeted retention offers to customers with high churn probability.

---

## Future Improvements

* Hyperparameter optimization
* Customer segmentation
* Real-time prediction API
* Cloud deployment
* Dashboard analytics
* Integration with live business databases

---

## Author

**Abdullah Fayyaz**

Machine Learning Internship Project

Task ID: **ML-2**

---

## License

This project was developed for educational and internship purposes.
