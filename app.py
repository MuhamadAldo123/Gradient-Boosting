import streamlit as st
import pandas as pd
import joblib

# Fungsi untuk memuat model dengan cache agar tidak di-load berulang kali
@st.cache_resource
def load_model():
    return joblib.load('gradient_boosting_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

st.title("Aplikasi Prediksi Gradient Boosting")

# Form input data untuk mencegah refresh otomatis setiap kali input berubah
with st.form("prediction_form"):
    st.subheader("Input Fitur Model")

    # Fitur Numerik
    bmxbmi = st.number_input("BMXBMI (Indeks Massa Tubuh)", value=25.0, step=0.1)
    lbxglu = st.number_input("LBXGLU (Glukosa Darah Puasa)", value=100.0, step=1.0)
    lbxglt = st.number_input("LBXGLT (Glukosa Toleransi Oral)", value=140.0, step=1.0)
    lbxin = st.number_input("LBXIN (Insulin)", value=15.0, step=0.1)

    # Fitur Kategorikal
    riagendr = st.selectbox("RIAGENDR (Jenis Kelamin)", options=[1, 2], format_func=lambda x: "Pria" if x == 1 else "Wanita")
    paq605 = st.selectbox("PAQ605 (Aktivitas Fisik)", options=[1, 2, 7, 9])
    diq010 = st.selectbox("DIQ010 (Status Diabetes)", options=[1, 2, 3, 7, 9])

    # Tombol Submit
    submitted = st.form_submit_button("Lakukan Prediksi")

if submitted:
    # Struktur data input disesuaikan dengan ColumnTransformer pada model
    input_data = pd.DataFrame({
        'BMXBMI': [bmxbmi],
        'LBXGLU': [lbxglu],
        'LBXGLT': [lbxglt],
        'LBXIN': [lbxin],
        'RIAGENDR': [riagendr],
        'PAQ605': [paq605],
        'DIQ010': [diq010]
    })

    try:
        # Eksekusi prediksi
        prediksi = model.predict(input_data)
        probabilitas = model.predict_proba(input_data)

        st.success("Proses Prediksi Berhasil")
        st.write(f"**Hasil Prediksi Kelas:** {prediksi[0]}")
        st.write(f"**Probabilitas Nilai:** {probabilitas[0]}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")