import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("tip_prediction_model.pkl")

st.set_page_config(page_title="Tip Prediction", page_icon="💰")

st.title("💰 Restaurant Tip Prediction")
st.write("Enter the customer information below to predict the expected tip.")

# Inputs
total_bill = st.number_input("Total Bill ($)", min_value=0.0, value=20.0)
sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])
size = st.slider("Party Size", 1, 10, 2)

is_weekend = 1 if day in ["Sat", "Sun"] else 0

if st.button("Predict Tip"):
    sample = pd.DataFrame({
        "total_bill": [total_bill],
        "sex": [sex],
        "smoker": [smoker],
        "day": [day],
        "time": [time],
        "size": [size],
        "is_weekend": [is_weekend]
    })

    prediction = model.predict(sample)[0]

    st.success(f"💵 Expected Tip: ${prediction:.2f}")