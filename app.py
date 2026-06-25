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

st.title("Prediksi Kelompok Usia Berdasarkan Kesehatan (NHANES 2013-2014)")
st.markdown("""
Aplikasi ini menggunakan dataset **NHANES Age Prediction Subset** dari UCI Machine Learning Repository. 
Tujuan dari model ini adalah memprediksi apakah profil kesehatan seseorang lebih mencerminkan kelompok **Senior (≥ 65 tahun)** atau **Non-Senior (< 65 tahun)** berdasarkan pengukuran fisiologis, tes biokimia, dan gaya hidup.
""")

# Form input data
with st.form("prediction_form"):
    st.subheader("📊 Pengukuran Fisiologis & Biokimia")

    # Fitur Numerik
    bmxbmi = st.number_input(
        "BMXBMI - Indeks Massa Tubuh (BMI)", 
        value=25.0, step=0.1,
        help="Continuous: Nilai Body Mass Index (BMI) responden."
    )
    lbxglu = st.number_input(
        "LBXGLU - Glukosa Darah Puasa (mg/dL)", 
        value=100.0, step=1.0,
        help="Continuous: Kadar glukosa darah responden setelah berpuasa."
    )
    lbxglt = st.number_input(
        "LBXGLT - Glukosa Toleransi Oral (mg/dL)", 
        value=140.0, step=1.0,
        help="Continuous: Kadar glukosa darah setelah Tes Toleransi Glukosa Oral (OGTT)."
    )
    lbxin = st.number_input(
        "LBXIN - Kadar Insulin (μU/mL)", 
        value=15.0, step=0.1,
        help="Continuous: Kadar insulin dalam darah responden."
    )

    st.markdown("---")
    st.subheader("🗣️ Demografi & Gaya Hidup")

    # Fitur Kategorikal 
    riagendr = st.selectbox(
        "RIAGENDR - Jenis Kelamin", 
        options=[1, 2], 
        format_func=lambda x: "1 - Pria" if x == 1 else "2 - Wanita",
        help="Feature: Jenis kelamin responden."
    )
    
    paq605 = st.selectbox(
        "PAQ605 - Aktivitas Fisik", 
        options=[1, 2, 7, 9],
        format_func=lambda x: {
            1: "1 - Ya (Melakukan olahraga/aktivitas fisik sedang-berat)", 
            2: "2 - Tidak (Tidak melakukan)", 
            7: "7 - Menolak menjawab", 
            9: "9 - Tidak tahu"
        }[x],
        help="Feature: Apakah responden terlibat dalam olahraga atau aktivitas fisik intensitas sedang/berat di minggu biasa."
    )
    
    diq010 = st.selectbox(
        "DIQ010 - Riwayat Diabetes", 
        options=[1, 2, 3, 7, 9],
        format_func=lambda x: {
            1: "1 - Ya (Pernah diberitahu dokter mengidap diabetes)", 
            2: "2 - Tidak", 
            3: "3 - Borderline (Garis batas/Prediabetes)", 
            7: "7 - Menolak menjawab", 
            9: "9 - Tidak tahu"
        }[x],
        help="Feature: Apakah responden pernah diberitahu oleh dokter bahwa mereka mengidap diabetes."
    )

    # Tombol Submit
    submitted = st.form_submit_button("Lakukan Prediksi", use_container_width=True)

if submitted:
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
        prediksi = model.predict(input_data)[0]
        probabilitas = model.predict_proba(input_data)[0]

        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi Model (Target: age_group)")
        
        # Menerjemahkan angka 0 dan 1 menjadi teks
        nama_kelas = "Non-Senior (< 65 tahun)" if prediksi == 0 else "Senior (≥ 65 tahun)"
        
        # Memberikan warna beda untuk hasil prediksi
        if prediksi == 0:
            st.success(f"**Model memprediksi pasien masuk dalam kelas:** `{prediksi}` - **{nama_kelas}**")
        else:
            st.warning(f"**Model memprediksi pasien masuk dalam kelas:** `{prediksi}` - **{nama_kelas}**")
        
        # Penjelasan kelas berdasarkan dokumentasi UCI NHANES
        st.markdown("""
        💡 **Keterangan Output Label:**
        Berdasarkan dataset asli, prediksi menargetkan dua grup:
        * **Kelas 0 (Non-Senior)** : Individu berusia di bawah 65 tahun.
        * **Kelas 1 (Senior)** : Individu berusia 65 tahun ke atas.
        """)

        st.write("### Detail Probabilitas (Keyakinan Model):")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Peluang Kelas '0' (Non-Senior)", value=f"{probabilitas[0] * 100:.1f}%")
        with col2:
            st.metric(label="Peluang Kelas '1' (Senior)", value=f"{probabilitas[1] * 100:.1f}%")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
