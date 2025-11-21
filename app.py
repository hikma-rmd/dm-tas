import streamlit as st
import pickle
import numpy as np

# =======================
# LOAD MODEL & SCALER
# =======================
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Obesity Level Prediction App")
st.write("""
Aplikasi ini memprediksi tingkat obesitas seseorang berdasarkan data fisik dan gaya hidup.
Model yang digunakan adalah **model terbaik** hasil training di Google Colab.
""")

# =======================
# INPUT FORM USER
# =======================

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=1, max_value=120)
height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, step=0.01)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, step=0.1)

family_overweight = st.selectbox("Family history with overweight", ["no", "yes"])
FAVC = st.selectbox("Frequent high-caloric food consumption (FAVC)", ["no", "yes"])
FCVC = st.number_input("Vegetable consumption frequency (1–3)", min_value=1, max_value=3)
NCP = st.number_input("Number of main meals (1–4)", min_value=1, max_value=4)
CAEC = st.selectbox("Eating between meals", ["no", "Sometimes", "Frequently", "Always"])
SMOKE = st.selectbox("Do you smoke?", ["no", "yes"])
CH2O = st.number_input("Daily water consumption (1–3)", min_value=1, max_value=3)
SCC = st.selectbox("Calories monitoring?", ["no", "yes"])
FAF = st.number_input("Physical activity frequency (0–3)", min_value=0, max_value=3)
TUE = st.number_input("Time using technology daily (0–2)", min_value=0, max_value=2)
CALC = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
MTRANS = st.selectbox("Transportation", ["Walking", "Bike", "Motorbike", "Public", "Car"])

# =======================
# LABEL ENCODING MANUAL (harus sama dengan Colab)
# =======================
encode = {
    "Female": 0, "Male": 1,
    "no": 0, "yes": 1,
    "Sometimes": 1, "Frequently": 2, "Always": 3,
    "Walking": 0, "Bike": 1, "Motorbike": 2, "Public": 3, "Car": 4
}

input_data = np.array([[
    encode[gender],
    age,
    height,
    weight,
    encode[family_overweight],
    encode[FAVC],
    FCVC,
    NCP,
    encode[CAEC],
    encode[SMOKE],
    CH2O,
    encode[SCC],
    FAF,
    TUE,
    encode[CALC],
    encode[MTRANS]
]])

# =======================
# NORMALISASI
# =======================
scaled_input = scaler.transform(input_data)

# =======================
# PREDIKSI
# =======================
if st.button("Predict Obesity Level"):
    prediction = model.predict(scaled_input)
    st.success(f"Hasil Prediksi: **{prediction[0]}**")
