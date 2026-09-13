import streamlit as st
import joblib as joblib 
import numpy as np 
import pandas as pd 

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

cluster_names = {
    0: "Low Value Customers",
    1: "Young High Value Customers",
    2: "Budget Customers",
    3: "Frequent Buyers",
    4: "Old High Value Customers"
}

st.title("Customer Segment Prediction")
st.write("Enter Customer Details to predict the Segment")

age= st.number_input("Age",min_value=18,max_value=100, value =30)
Income = st.number_input("Income", min_value=0, max_value=2000000, value=50000)
Total_Spending = st.number_input("Total Number of Spendings",min_value=0, max_value=100000000,value=1000)
num_web_purchases = st.number_input("Number of Web Purchases",min_value=0,value=5)
num_store_purchases = st.number_input("Number of Store Purchases",min_value=0,value=5)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0,value=5)
recency = st.number_input("Recency (days since last purchase)",min_value=0,value=30)

input_data = pd.DataFrame({
    "Age": [age],
    "Income": [Income],
    "Total_Spendings": [Total_Spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segmentation"):
    cluster = kmeans.predict(input_scaled)

    segment = cluster_names[cluster[0]]

    st.success(f"Predicted Customer Segment: {segment}")