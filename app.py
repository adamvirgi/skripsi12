import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('model.joblib') # Pastikan 'model.joblib' berada di direktori yang sama

# Create a function to predict the status of a child
def predict_status(age, weight, height, gender, lingkar_kepala): # Mengganti 'other_feature' dengan 'lingkar_kepala'
    input_data = np.asarray([age, weight, height, gender, lingkar_kepala])
    input_data_reshaped = input_data.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

# Streamlit app
st.title("Child Status Prediction")

# Get user input
age = st.number_input("Enter the child's age in months:", min_value=0)
weight = st.number_input("Enter the child's weight in kilograms:", min_value=0.0)
height = st.number_input("Enter the child's height in centimeters:", min_value=0.0)
gender = st.selectbox("Enter the child's gender:", ["Female", "Male"])
gender = 0 if gender == "Female" else 1
lingkar_kepala = st.number_input("Enter the child's head circumference in centimeters:") # Ganti '[other feature name]' dengan 'lingkar_kepala'

# Make prediction
if st.button("Predict"):
    predicted_status = predict_status(age, weight, height, gender, lingkar_kepala)
    st.success(f"The predicted status of the child is: {predicted_status}")
