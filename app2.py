import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Churn Prediction Dashboard")
st.sidebar.markdown("Welcome! Navigate through the sections below:")

section = st.sidebar.radio(
    "Go to:",
    ["Home", "Data Preview", "Visualizations", "Model & Predictions", "Top 10% Risk", "ROI Calculator"]
)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("Churn_Modelling.csv")
X = df[["CreditScore", "Age", "Balance", "EstimatedSalary"]]
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:,1]
threshold = np.percentile(probabilities, 90)
top_10_preds = (probabilities >= threshold).astype(int)
precision_top10 = precision_score(y_test, top_10_preds)
recall_top10 = recall_score(y_test, top_10_preds)

X_test_with_prob = X_test.copy()
X_test_with_prob['Churn_Probability'] = probabilities
top_10_percent = X_test_with_prob.nlargest(int(0.1 * len(X_test_with_prob)), 'Churn_Probability')

# -----------------------------
# Home
# -----------------------------
if section == "Home":
    st.title("Welcome to the Customer Churn Dashboard")
    name = st.text_input("Enter your name", "Alireza Sadeghi")
    st.write(f"Hello, {name}! This dashboard helps you explore your customers, predict churn, and plan retention campaigns.")
    st.info("Use the sidebar to navigate through different sections: Data, Visualizations, Model, Top 10% Risk, ROI.")

# -----------------------------
# Data Preview
# -----------------------------
elif section == "Data Preview":
    st.title("Dataset Preview")
    st.write("Here is a preview of your customer data:")
    st.dataframe(df.head())
    st.write(f"Total records: {df.shape[0]}, Total columns: {df.shape[1]}")

# -----------------------------
# Visualizations
# -----------------------------
elif section == "Visualizations":
    st.title("Customer Data Visualizations")

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Exited', data=df, ax=ax)
    ax.set_xticklabels(['Stayed', 'Churned'])
    st.pyplot(fig)

    st.subheader("Age Distribution by Churn Status")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='Age', hue='Exited', bins=30, kde=True, multiple="stack", palette="Set2", ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Correlation Heatmap")
    numeric_df = df[["CreditScore", "Age", "Balance", "EstimatedSalary", "Exited"]]
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# -----------------------------
# Model & Predictions
# -----------------------------
elif section == "Model & Predictions":
    st.title("Churn Predictions")
    st.success(f"Model Accuracy: {model.score(X_test, y_test):.2f}")

    st.subheader("Predict for a New Customer")

    # Two-column layout for user-friendly order
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30)
        credit_score = st.slider("Credit Score", 400, 900, 650, step=10)
    with col2:
        balance = st.slider("Balance ($)", 0, 200000, 50000, step=1000)
        salary = st.slider("Estimated Salary ($)", 0, 200000, 50000, step=1000)

    st.write("")  # small spacing

    if st.button("Predict"):
        user_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Age": [age],
            "Balance": [balance],
            "EstimatedSalary": [salary]
        })
        prediction = model.predict(user_data)[0]
        probability = model.predict_proba(user_data)[0][1]

        st.markdown("### Prediction Result")
        st.write(f"**Prediction:** {'Will Leave ' if prediction else 'Will Stay'}")
        st.write(f"**Churn Probability:** {probability:.2f}")

# -----------------------------
# Top 10% Risk
# -----------------------------
elif section == "Top 10% Risk":
    st.title("Top 10% High-Risk Customers")
    st.write("These customers have the highest predicted probability to churn:")
    st.dataframe(top_10_percent)
    st.metric("Precision @ Top 10%", f"{precision_top10:.2f}")
    st.metric("Recall @ Top 10%", f"{recall_top10:.2f}")

# -----------------------------
# ROI Calculator
# -----------------------------
elif section == "ROI Calculator":
    st.title("Business ROI from Retention Campaign")
    st.write("Estimate ROI by targeting the top 10% high-risk customers.")

    campaign_cost = st.number_input("Retention Campaign Cost per Customer ($)", 0.0, 1000.0, 50.0)
    revenue_saved = st.number_input("Revenue Saved per Retained Customer ($)", 0.0, 5000.0, 500.0)

    num_targeted = len(top_10_percent)
    expected_savings = revenue_saved * num_targeted
    total_cost = campaign_cost * num_targeted
    roi = expected_savings - total_cost

    st.write(f"Number of customers targeted: {num_targeted}")
    st.write(f"Total Campaign Cost: ${total_cost:.2f}")
    st.write(f"Expected Revenue Saved: ${expected_savings:.2f}")

    if roi >= 0:
        st.success(f"Estimated ROI: ${roi:.2f}")
    else:
        st.error(f"Estimated ROI: ${roi:.2f}")
