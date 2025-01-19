import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

import os
import matplotlib.pyplot as plt
# Set page configuration
st.set_page_config(
    page_title="Aplikasi Prediksi Diabetes",
    page_icon="🩺",
    initial_sidebar_state="expanded"  # Mulai dengan sidebar terbuka
)

st.write("""
# APLIKASI PREDIKSI DIABETES
Aplikasi ini digunakan untuk memprediksi apakah seseorang memiliki penyakit diabetes atau tidak.""")

img = Image.open("gambar.png")
st.image(img, width=600)

st.sidebar.header('Masukkan Parameter Pasien')

#upload file cv
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        gender = st.sidebar.selectbox('Jenis Kelamin (1: Laki-laki, 0: Perempuan)', (1, 0))
        age = st.sidebar.slider('Umur', 21, 81, 21)
        hypertension = st.sidebar.selectbox(' Darah Tinggi (Hypertension) (1: Iya, 0: Tidak)', (1, 0))
        heart_disease = st.sidebar.selectbox('Penyakit Jantung (1: Iya, 0: No)', (1, 0))
        smoking_history = st.sidebar.selectbox('Riwayat Merokok (0: Tidak Pernah, 1: Pernah Merokok, 2: Masih Merokok)', (0, 1, 2))
        bmi = st.sidebar.slider('Body Mass Index (Berat Badan (kg) / (Tinggi Badan (m)) x 2)', 10.0, 70.0, 10.0)
        HbA1c_level = st.sidebar.slider('HbA1c Level (rata-rata kadar gula darah Anda dalam dua hingga tiga bulan terakhir)', 0.0, 20.0, 0.0)
        blood_glucose_level = st.sidebar.slider('Tingkat Glukosa Darah', 0.0, 300.0, 0.0)
        data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_history': smoking_history,
            'bmi': bmi,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level
            }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

#menggabungkan input user dengan data diabetes_prediction_dataset.csv
diabetes_raw = pd.read_csv('diabetes.csv')
diabetes = diabetes_raw.drop(columns=['diabetes'])
df = pd.concat([input_df,diabetes],axis=0)
df = df[:1]

#menampilkan parameter hasil inputan
if uploaded_file is not None:
    st.write(df)
else:
    st.write("Silahkan upload file CSV atau masukkan parameter pada sidebar")
    st.write(df)

if st.button("Mulai prediksi"):
    # load model random forest classifier
    with open('model_prediksi.pkl', 'rb') as file:
        load_model = pickle.load(file)
    # Check if the loaded model is a RandomForestClassifier
    if isinstance(load_model, RandomForestClassifier):
        # prediksi
        prediksi = load_model.predict(df)
        prediksi_proba = load_model.predict_proba(df)

        prediksi_int = prediksi.astype(int)
        st.subheader("Keterangan Label Kelas")
        kelas_diabetes = np.array(['Tidak Diabetes', 'Diabetes'])
        st.write(kelas_diabetes)

        st.subheader("Hasil Prediksi")
        st.write(kelas_diabetes[prediksi_int])

        st.subheader("Probabilitas (0: Tidak Diabetes, 1: Diabetes)")
        st.write(prediksi_proba)

        #create pie chart
        labels = ['Tidak Diabetes', 'Diabetes']
        sizes = prediksi_proba[0]
        colors = ['#ff9999','#66b3ff']
        explode = (0.1, 0)

        fig1, ax1 = plt.subplots()
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',startangle=90)
        ax1.axis('equal')

        st.pyplot(fig1)

        if prediksi_int == 1:
            st.write("Pasien memiliki kemungkinan diabetes, segera konsultasikan ke dokter.")
        else:
            st.write("Pasien tidak memiliki diabetes, tetap jaga pola hidup sehat.")
    else:
        st.write("Model yang diunggah tidak valid. Pastikan model adalah RandomForestClassifier.")

