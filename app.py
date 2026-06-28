import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


model = joblib.load("xgboost_model.pkl")


explainer = shap.TreeExplainer(model)


st.title("📊 Customer Churn Prediction System")

st.markdown("""
### Machine Learning Internship Project

This application predicts whether a customer is likely to churn using an **XGBoost Machine Learning Model**.

Fill in the customer details below and click **Predict Churn**.
""")

st.divider()

left, right = st.columns(2)



with left:

    age = st.slider(
        "Age",
        18,
        70,
        30
    )

    gender = st.selectbox(
        "Gender",
        ["Female","Male"]
    )

    city = st.selectbox(
        "City",
        [
            "Faisalabad",
            "Islamabad",
            "Karachi",
            "Lahore",
            "Multan",
            "Peshawar",
            "Quetta",
            "Rawalpindi"
        ]
    )

    subscription = st.selectbox(
        "Subscription Type",
        [
            "Basic",
            "Premium",
            "Standard"
        ]
    )

    monthly_spending = st.number_input(
        "Monthly Spending",
        min_value=500,
        max_value=8000,
        value=2500
    )

    tenure = st.slider(
        "Customer Tenure (Months)",
        1,
        60,
        12
    )

with right:

    purchases = st.slider(
        "Number of Purchases",
        1,
        100,
        20
    )

    support = st.slider(
        "Customer Support Requests",
        0,
        10,
        2
    )

    login = st.slider(
        "Login Frequency",
        1,
        30,
        15
    )

    inactivity = st.slider(
        "Last Activity (Days Ago)",
        0,
        90,
        5
    )

    satisfaction = st.slider(
        "Satisfaction Score",
        1,
        10,
        7
    )

st.divider()



gender_map = {
    "Female":0,
    "Male":1
}

city_map = {
    "Faisalabad":0,
    "Islamabad":1,
    "Karachi":2,
    "Lahore":3,
    "Multan":4,
    "Peshawar":5,
    "Quetta":6,
    "Rawalpindi":7
}

subscription_map = {
    "Basic":0,
    "Premium":1,
    "Standard":2
}

gender = gender_map[gender]
city = city_map[city]
subscription = subscription_map[subscription]


average_purchase = monthly_spending / (purchases + 1)

engagement_score = login * satisfaction

support_per_month = support / (tenure + 1)

# -----------------------------------
# Create Input DataFrame
# -----------------------------------

customer = pd.DataFrame({

    "Age":[age],
    "Gender":[gender],
    "City":[city],
    "SubscriptionType":[subscription],
    "MonthlySpending":[monthly_spending],
    "Tenure":[tenure],
    "NumberOfPurchases":[purchases],
    "SupportRequests":[support],
    "LoginFrequency":[login],
    "LastActivityDays":[inactivity],
    "SatisfactionScore":[satisfaction],
    "AveragePurchase":[average_purchase],
    "EngagementScore":[engagement_score],
    "SupportPerMonth":[support_per_month]

})

predict = st.button(
    "🚀 Predict Churn",
    use_container_width=True
)

if predict:

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]



    st.divider()

    st.header("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")

    with result_col2:

        st.metric(
            label="Churn Probability",
            value=f"{probability*100:.2f}%"
        )

   
    st.subheader("Risk Level")

    if probability < 0.30:

        st.success("🟢 Low Risk")

    elif probability < 0.70:

        st.warning("🟡 Medium Risk")

    else:

        st.error("🔴 High Risk")

    st.progress(float(probability))

   

    st.divider()

    st.header("Customer Summary")

    summary = pd.DataFrame({

        "Feature":[

            "Age",
            "Monthly Spending",
            "Tenure",
            "Purchases",
            "Support Requests",
            "Login Frequency",
            "Last Activity Days",
            "Satisfaction Score",
            "Average Purchase",
            "Engagement Score",
            "Support Per Month"

        ],

        "Value":[

            age,
            monthly_spending,
            tenure,
            purchases,
            support,
            login,
            inactivity,
            satisfaction,
            round(average_purchase,2),
            engagement_score,
            round(support_per_month,2)

        ]

    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

   

    st.divider()

    st.header("Most Important Features (Model)")

    importance = pd.DataFrame({

        "Feature":customer.columns,

        "Importance":model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.write("Top 5 Most Important Features")

    st.dataframe(
        importance.head(5),
        use_container_width=True,
        hide_index=True
    )
      

    st.divider()

    st.header("Explainable AI (SHAP)")

    st.write(
        """
        SHAP (SHapley Additive Explanations) explains why the model
        predicted churn by showing how each feature influenced the prediction.
        Positive values increase churn risk, while negative values decrease it.
        """
    )

    try:

        # Calculate SHAP values
        shap_values = explainer(customer)

        # Waterfall Plot
        fig, ax = plt.subplots(figsize=(10,6))

        shap.plots.waterfall(
            shap_values[0],
            show=False
        )

        st.pyplot(fig)

    except Exception as e:

        st.warning("SHAP waterfall plot could not be displayed.")
        st.text(str(e))

    # -----------------------------------
    # SHAP Feature Contributions Table
    # -----------------------------------

    st.subheader("Feature Contributions")

    try:

        contributions = pd.DataFrame({

            "Feature": customer.columns,

            "Contribution":
                shap_values.values[0]

        })

        contributions["Absolute"] = contributions["Contribution"].abs()

        contributions = contributions.sort_values(
            by="Absolute",
            ascending=False
        )

        st.dataframe(
            contributions[["Feature","Contribution"]],
            use_container_width=True,
            hide_index=True
        )

    except:

        pass

   

    st.divider()

    st.header("Business Recommendation")

    if prediction == 1:

        st.error(
            """
### Recommended Actions

• Contact the customer with a retention offer.

• Provide personalized discounts.

• Improve customer satisfaction.

• Reduce response time to support tickets.

• Increase engagement through emails and notifications.

• Monitor this customer closely.
"""
        )

    else:

        st.success(
            """
### Recommended Actions

• Continue regular engagement.

• Offer loyalty rewards.

• Promote premium subscriptions.

• Maintain high customer satisfaction.

• Encourage referrals.
"""
        )

   