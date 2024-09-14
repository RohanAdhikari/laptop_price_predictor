import streamlit as st
import pickle
import numpy as np

# Load the model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Title of the app
st.title("💻 Laptop Price Predictor")
st.write("Predict the price of a laptop based on various configurations.")

# --- Sidebar for user input ---
st.sidebar.header("Select Laptop Specifications")

# Brand
company = st.sidebar.selectbox('Laptop Brand', df['Company'].unique())

# Laptop Type
laptop_type = st.sidebar.selectbox('Type of Laptop', df['TypeName'].unique())

# RAM (in GB)
ram = st.sidebar.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight (in kg)
weight = st.sidebar.number_input('Weight of the Laptop (in kg)', min_value=0.5, max_value=5.0, step=0.1)

# Touchscreen
touchscreen = st.sidebar.selectbox('Touchscreen', ['No', 'Yes'])

# IPS Display
ips = st.sidebar.selectbox('IPS Display', ['No', 'Yes'])

# Screen Size
screen_size = st.sidebar.slider('Screen Size (in inches)', 10.0, 18.0, 15.6)

# Screen Resolution
resolution = st.sidebar.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU
cpu = st.sidebar.selectbox('CPU Brand', df['Cpu brand'].unique())

# HDD (in GB)
hdd = st.sidebar.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD (in GB)
ssd = st.sidebar.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.sidebar.selectbox('GPU Brand', df['Gpu brand'].unique())

# Operating System
os = st.sidebar.selectbox('Operating System', df['os'].unique())

# --- Prediction Button ---
if st.sidebar.button('Predict Price 💰'):
    # Convert categorical values to numeric
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI (Pixels Per Inch)
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Prepare input query for prediction
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, -1)

    # Predict the price
    predicted_price = int(np.exp(pipe.predict(query)[0]) * 1.6)


    # Display the result
    st.subheader(f"The predicted price of this configuration is: Rs. {predicted_price}")

# --- Footer ---
st.write("--------------------------------------------")
st.write("First end-to-end project by Rohan Adhikari using Streamlit")
