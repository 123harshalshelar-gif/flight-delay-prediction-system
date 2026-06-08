import streamlit as st
import numpy as np
import joblib

model = joblib.load("flight_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("✈ Flight Delay Prediction System")

month = st.number_input("Month", 1, 12, 1)
day = st.number_input("Day", 1, 31, 1)
day_of_week = st.number_input("Day Of Week", 1, 7, 1)
distance = st.number_input("Distance")
departure_delay = st.number_input("Departure Delay")

if st.button("Predict"):

    data = np.array([[
        month,
        day,
        day_of_week,
        distance,
        departure_delay
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Flight Likely To Be Delayed")
    else:
        st.success("Flight Likely To Arrive On Time")
