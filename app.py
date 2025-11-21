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

# =======================
# JUDUL APLIKASI
# =======================
st.title("Aplikasi Prediksi Tingkat Obesitas")
st.write("""
Selamat datang!  
Aplikasi ini memprediksi **tingkat obesitas seseorang** berdasarkan data fisik dan kebiasaan hidup.

---

### 🇮🇩 Bahasa Indonesia:
Masukkan data pada form di bawah untuk mendapatkan hasil prediksi.

### 🇺🇸 English:
Fill in the form below to get the obesity prediction.
""")

# =======================
# INPUT FORM USER
# =======================

st.subheader("📌 Input Data Pengguna / User Input")

gender = st.selectbox("Gender (Jenis Kelamin)", ["Female (Perempuan)", "Male (Laki-laki)"])
age = st.number_input("Age (Umur)", min_value=1, max_value=120)
height = st.number_input("Height / Tinggi Badan (meter)", min_value=1.0, max_value=2.5, step=0.01)
weight = st.number_input("Weight / Berat Badan (kg)", min_value=20.0, max_value=200.0, step=0.1)

family_overweight = st.selectbox("Family history of overweight (Riwayat keluarga obesitas)", ["no", "yes"])
FAVC = st.selectbox("High-calorie food consumption (Konsumsi makanan berkalori tinggi)", ["no", "yes"])
FCVC = st.number_input("Vegetable intake (Frekuensi makan sayur) (1–3)", min_value=1, max_value=3)
NCP = st.number_input("Number of main meals (Jumlah makan utama) (1–4)", min_value=1, max_value=4)
CAEC = st.selectbox("Eating between meals (Cemilan di antara makan)", ["no", "Sometimes", "Frequently", "Always"])
SMOKE = st.selectbox("Do you smoke? (Apakah merokok?)", ["no", "yes"])
CH2O = st.number_input("Daily water intake / Konsumsi air (1–3)", min_value=1, max_value=3)
SCC = st.selectbox("Calories monitoring (Memantau kalori?)", ["no", "yes"])
FAF = st.number_input("Physical activity (Aktivitas fisik) (0–3)", min_value=0, max_value=3)
TUE = st.number_input("Technology usage time (Waktu pakai gadget) (0–2)", min_value=0, max_value=2)
CALC = st.selectbox("Alcohol consumption (Konsumsi alkohol)", ["no", "Sometimes", "Frequently", "Always"])
MTRANS = st.selectbox("Transportation (Transportasi)", ["Walking", "Bike", "Motorbike", "Public", "Car"])

# =======================
# LABEL ENCODING (sama seperti training)
# =======================
encode = {
    "Female (Perempuan)": 0, "Male (Laki-laki)": 1,
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
# NORMALISASI DATA
# =======================
scaled_input = scaler.transform(input_data)

# =======================
# PREDIKSI
# =======================
if st.button("🔍 Prediksi / Predict"):
    prediction = model.predict(scaled_input)

    st.success(f"""
### 🎯 Hasil Prediksi / Prediction Result
**{prediction[0]}**

Artinya: Tingkat obesitas diperkirakan berada pada kategori tersebut.
""")
