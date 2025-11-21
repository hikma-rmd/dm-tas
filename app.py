import streamlit as st
import numpy as np
import pickle

# ============================================
# 1. LOAD MODEL
# ============================================
@st.cache_resource
def load_models():
    with open("svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)

    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

    with open("voting_model.pkl", "rb") as f:
        voting_model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return svm_model, rf_model, voting_model, scaler


svm_model, rf_model, voting_model, scaler = load_models()

label_mapping = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}

# ============================================
# 2. UI
# ============================================
st.title("🍏 Prediksi Obesitas Menggunakan Machine Learning")

st.sidebar.header("📊 Pilih Model Machine Learning")
model_choice = st.sidebar.selectbox(
    "Pilih Model Prediksi:",
    ("SVM", "Random Forest", "Voting Classifier")
)

# ============================================
# 3. FORM INPUT
# ============================================
st.header("📝 Input Data Pengguna")

with st.form("prediction_form"):
    gender = st.selectbox("Jenis Kelamin", ("Perempuan", "Laki-laki"))
    age = st.number_input("Usia", 10, 100, 25)
    height = st.number_input("Tinggi Badan (m)", 1.20, 2.20, 1.60)
    weight = st.number_input("Berat Badan (kg)", 30, 200, 60)
    fh = st.selectbox("Riwayat Obesitas Keluarga", ("Tidak", "Ya"))
    favc = st.selectbox("Konsumsi Makanan Tinggi Kalori", ("Tidak", "Ya"))
    fcvc = st.slider("Frekuensi Konsumsi Sayur", 1, 3, 2)
    ncp = st.slider("Jumlah Makan Utama Per Hari", 1, 4, 3)
    caec = st.selectbox("Kebiasaan Ngemil", ("Tidak Pernah", "Kadang", "Sering", "Selalu"))
    smoke = st.selectbox("Merokok", ("Tidak", "Ya"))
    ch2o = st.slider("Konsumsi Air (L)", 1, 4, 2)
    scc = st.selectbox("Pantau Kalori", ("Tidak", "Ya"))
    faf = st.slider("Frekuensi Aktivitas Fisik", 0, 3, 1)
    tue = st.slider("Waktu Layar / Gadget (jam)", 0, 3, 2)
    calc = st.selectbox("Konsumsi Alkohol", ("Tidak Pernah", "Kadang", "Sering"))
    mtrans = st.selectbox("Transportasi Utama", ("Kendaraan Umum", "Sepeda", "Mobil", "Berjalan Kaki"))

    submit = st.form_submit_button("Prediksi Sekarang")

# ============================================
# 4. KONVERSI INPUT
# ============================================
def convert_text(value):
    mapping = {
        "Perempuan": 1, "Laki-laki": 0,
        "Tidak": 0, "Ya": 1,
        "Tidak Pernah": 0, "Kadang": 1, "Sering": 2, "Selalu": 3,
        "Kendaraan Umum": 0, "Sepeda": 1, "Mobil": 2, "Berjalan Kaki": 3
    }
    return mapping.get(value, value)

if submit:

    input_data = np.array([[
        convert_text(gender),
        age,
        height,
        weight,
        convert_text(fh),
        convert_text(favc),
        fcvc,
        ncp,
        convert_text(caec),
        convert_text(smoke),
        ch2o,
        convert_text(scc),
        faf,
        tue,
        convert_text(calc),
        convert_text(mtrans)
    ]])

    input_scaled = scaler.transform(input_data)

    # Prediksi
    if model_choice == "SVM":
        prediction = svm_model.predict(input_scaled)[0]
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(input_scaled)[0]
    else:
        prediction = voting_model.predict(input_scaled)[0]

    result_text = label_mapping.get(prediction, "Tidak diketahui")

    st.success(f"📌 **Hasil Prediksi: {result_text}**")
