import streamlit as st
import pickle
import numpy as np


pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))


st.title("💻 Laptop Price Predictor")
st.write("Predict the price of a laptop based on various configurations.")


st.sidebar.header("Select Laptop Specifications")


company = st.sidebar.selectbox('Laptop Brand', df['Company'].unique())


laptop_type = st.sidebar.selectbox('Type of Laptop', df['TypeName'].unique())


ram = st.sidebar.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])


weight = st.sidebar.number_input('Weight of the Laptop (in kg)', min_value=0.5, max_value=5.0, step=0.1)


touchscreen = st.sidebar.selectbox('Touchscreen', ['No', 'Yes'])


ips = st.sidebar.selectbox('IPS Display', ['No', 'Yes'])


screen_size = st.sidebar.slider('Screen Size (in inches)', 10.0, 18.0, 15.6)


resolution = st.sidebar.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])


cpu = st.sidebar.selectbox('CPU Brand', df['Cpu brand'].unique())


hdd = st.sidebar.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])


ssd = st.sidebar.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])


gpu = st.sidebar.selectbox('GPU Brand', df['Gpu brand'].unique())


os = st.sidebar.selectbox('Operating System', df['os'].unique())


if st.sidebar.button('Predict Price 💰'):
    
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

  
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size


    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, -1)


    predicted_price = int(np.exp(pipe.predict(query)[0]))


   
    st.subheader(f"The predicted price of this configuration is: Rs. {predicted_price}")


st.write("--------------------------------------------")
st.write("First end-to-end project by Rohan Adhikari using Streamlit")
