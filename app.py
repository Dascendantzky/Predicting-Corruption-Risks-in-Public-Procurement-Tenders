import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import os

st.title("🧾 Prediksi Risiko Korupsi Tender Pengadaan")

# Load dataset untuk encoder
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/Muhammad Zaky T A/Downloads/Machine Learning/Projek ML/data_bersih2.csv")

data = load_data()

# Daftar fitur yang digunakan model
feature_names = [
    'jenis_prosedur', 'jenis_pengadaan', 'jumlah_lot',
    'jumlah_penawaran_terekam', 'harga_estimasi', 'harga_penawaran',
    'status_lot', 'jumlah_penawar', 'negara_instansi', 'tipe_penyedia',
    'penyedia_menang', 'sumber_data', 'tahun_tender', 'harga_digiwhist',
    'filter_instansi_valid', 'filter_penyedia_valid', 'filter_dibatalkan',
    'filter_terbuka', 'filter_tahun_valid', 'filter_penawar_kalah',
    'data_valid', 'durasi_penawaran', 'durasi_keputusan',
    'ada_harga_penawaran'
]

# Label Encoding untuk fitur kategorikal
def build_label_encoders(df, feature_names):
    le_dict = {}
    for col in feature_names:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
    return le_dict

le_dict = build_label_encoders(data.copy(), feature_names)

# Load model dari file pickle
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

rf_model = load_model("C:/Users/Muhammad Zaky T A/Downloads/Machine Learning/Projek ML/rf_manual_model.pkl")
svm_model = load_model("C:/Users/Muhammad Zaky T A/Downloads/Machine Learning/Projek ML/svm_manual_model.pkl")

# Form input user
st.markdown("### Masukkan data fitur untuk melakukan prediksi:")

input_data = {}
for fitur in feature_names:
    if fitur in le_dict:
        options = list(le_dict[fitur].classes_)
        pilihan = st.selectbox(f"{fitur}", options)
        input_data[fitur] = le_dict[fitur].transform([pilihan])[0]
    else:
        val = st.number_input(f"{fitur}", value=0.0)
        input_data[fitur] = val

if st.button("🔍 Prediksi Risiko"):
    input_df = pd.DataFrame([input_data])

    pred_rf = rf_model.predict(input_df)[0]
    pred_svm_raw = svm_model.predict(input_df)[0]
    pred_svm = 0 if pred_svm_raw == -1 else 1

    st.subheader("📊 Hasil Prediksi")
    st.write(f"🟢 Random Forest Manual: **{pred_rf}**")
    st.write(f"🔵 SVM Manual: **{pred_svm}**")

    if pred_rf == 1 or pred_svm == 1:
        st.error("⚠️ Risiko korupsi TERDETEKSI!")
    else:
        st.success("✅ Tidak terdeteksi risiko korupsi.")
