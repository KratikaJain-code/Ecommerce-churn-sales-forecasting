# 🛒 E-Commerce Customer Churn & Sales Forecasting

An end-to-end data science and machine learning project that analyzes e-commerce transaction data, predicts customer churn, and forecasts future sales.

## 📌 Project Overview

E-commerce businesses generate large amounts of transaction data, but raw transaction records alone do not provide actionable insights.

This project transforms transaction-level data into meaningful business insights by combining:

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Revenue analysis
- Customer RFM analysis
- Time-based customer churn prediction
- Machine learning
- Sales forecasting
- Interactive visualization

## 🎯 Objectives

- Analyze historical e-commerce sales.
- Identify important sales trends and patterns.
- Understand customer purchasing behavior.
- Predict future customer inactivity using machine learning.
- Compare Logistic Regression and Random Forest models.
- Forecast future sales.
- Present results through an interactive application.

## 🧠 Machine Learning

### Customer Churn Prediction

Customer behavior is summarized using:

- Recency
- Frequency
- Monetary
- Average Order Value

The project uses a time-based approach where historical customer behavior is used to predict future inactivity.

Models:

- Logistic Regression
- Random Forest

### Sales Forecasting

Historical transaction revenue is aggregated over time and used to identify sales trends and generate future sales forecasts.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Google Colab
- GitHub

## 📊 Dataset

The project uses the Online Retail II transaction dataset.

The analysis uses the `Year 2010-2011` worksheet containing transaction-level e-commerce data.

## 👩‍💻 Author

**Kratika Jain**

## 📂 Project Structure

```text
ecommerce-churn-sales-forecasting/
│
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
├── data/
├── notebooks/
├── src/
├── models/
└── outputs/
