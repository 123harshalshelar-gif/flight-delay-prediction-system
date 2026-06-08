import streamlit as st
import numpy as np
import joblib

model = joblib.load("flight_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("✈ Flight Delay Prediction System")

month = st.number_input("Month", 1, 12, 1)
day = st.number_input("Day", 1, 31, 1)
day_of_week = st.number_input("Day Of Week", 1, 7, 1)
airline = st.number_input("Airline")
origin = st.number_input("Origin Airport")
destination = st.number_input("Destination Airport")
distance = st.number_input("Distance")
departure_delay = st.number_input("Departure Delay")
scheduled_time = st.number_input("Scheduled Time")
air_time = st.number_input("Air Time")

if st.button("Predict"):

    data = np.array([[
        month,
        day,
        day_of_week,
        airline,
        origin,
        destination,
        distance,
        departure_delay,
        scheduled_time,
        air_time
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Flight Likely To Be Delayed")
    else:
        st.success("Flight Likely To Arrive On Time")
