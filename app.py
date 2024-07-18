import streamlit as st
import pickle
import numpy as np

# Muat model dan scaler
with open('model/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_heart_disease(model, scaler, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    # Masukkan data ke dalam numpy array
    user_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
    
    # Standarisasi data
    user_data = scaler.transform(user_data)
    
    # Lakukan prediksi
    prediction = model.predict(user_data)
    
    # Konversi hasil prediksi ke dalam bentuk yang dapat dibaca
    if prediction[0] == 1:
        result = "Risiko penyakit jantung"
    else:
        result = "Tidak ada risiko penyakit jantung"
    
    return result

st.title('Heart Disease Prediction')

age = st.number_input('Age', min_value=0, max_value=120, value=0)
gender = st.selectbox('Gender', options=[1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
height = st.number_input('Height (cm)', min_value=50, max_value=250, value=0)
weight = st.number_input('Weight (kg)', min_value=10, max_value=300, value=0)
ap_hi = st.number_input('Systolic blood pressure', min_value=50, max_value=250, value=0)
ap_lo = st.number_input('Diastolic blood pressure', min_value=30, max_value=150, value=0)
cholesterol = st.selectbox('Cholesterol', options=[1, 2, 3], format_func=lambda x: ['normal', 'above normal', 'well above normal'][x-1])
gluc = st.selectbox('Glucose', options=[1, 2, 3], format_func=lambda x: ['normal', 'above normal', 'well above normal'][x-1])
smoke = st.selectbox('Do you smoke?', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
alco = st.selectbox('Do you drink alcohol?', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
active = st.selectbox('Are you physically active?', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

if st.button('Predict'):
    result = predict_heart_disease(knn_model, scaler, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
    st.write(result)
