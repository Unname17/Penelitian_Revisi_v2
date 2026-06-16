import streamlit as st
import joblib
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from deep_translator import GoogleTranslator
import numpy as np

# 1. LOAD MODEL
@st.cache_resource
def load_models():
    svm_model = joblib.load('svm_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    lda_model = LdaModel.load('lda_model.gensim')
    lda_dict = Dictionary.load('dictionary.gensim')
    return svm_model, tfidf_vectorizer, lda_model, lda_dict

svm_model, tfidf_vectorizer, lda_model, lda_dict = load_models()

aspect_names = {
    0: "Kesejahteraan & Status Kepegawaian",
    1: "Anggaran & Kualitas Pendidikan",
    2: "Komponen Pendapatan & Tunjangan",
    3: "Kondisi Sosial Ekonomi & Kelas Menengah",
    4: "Birokrasi & Jenjang Karier"
}

# 2. UI DASHBOARD
st.title("Dashboard Analisis Aspek & Sentimen")
st.markdown("Analisis mendalam mengenai keluhan Dosen Non-ASN")

user_input = st.text_area("Masukkan ulasan:", placeholder="Contoh: Gaji bulan ini telat lagi...")

if st.button("Analisis Sekarang"):
    # Translasi
    translated = GoogleTranslator(source='id', target='en').translate(user_input)
    
    # Deteksi Aspek (LDA)
    tokens = translated.lower().split()
    bow = lda_dict.doc2bow(tokens)
    topics = lda_model[bow][0]
    best_topic = sorted(topics, key=lambda x: x[1], reverse=True)[0][0]
    aspek = aspect_names[best_topic]
    
    # Deteksi Sentimen (SVM)
    tfidf = tfidf_vectorizer.transform([translated])
    sentimen = svm_model.predict(tfidf)[0]
    
    # 3. OUTPUT TERPADU
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Aspek Terdeteksi", aspek)
    with col2:
        st.metric("Sentimen", sentimen)
    
    st.write(f"Penjelasan: Teks ulasan Anda dikategorikan sebagai aspek **{aspek}** dengan sentimen **{sentimen}**.")
