import streamlit as st
import joblib
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from deep_translator import GoogleTranslator
import numpy as np
import pandas as pd
import requests
from io import BytesIO

# 1. LOAD MODEL
@st.cache_resource(ttl=600) # ttl=600 artinya cache akan disegarkan setiap 10 menit

def load_resources():
    # Load model
    svm_model = joblib.load('svm_model_4.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer_4.pkl')
    lda_model = LdaModel.load('lda_model_4.model') # Nama file diperbarui
    lda_dict = Dictionary.load('lda_dictionary_4.dict') # Nama file diperbarui
    
    url = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
    response = requests.get(url)
    kamus_data = pd.read_excel(BytesIO(response.content))
    
    # Membuat dict dari kolom 'tidak_baku' ke 'kata_baku'
    kamus_norm = dict(zip(kamus_data.iloc[:, 0].astype(str).str.lower(), kamus_data.iloc[:, 1].astype(str)))
    
    # Tambahkan kamus pribadi (contoh)
    kamus_pribadi = {
    'lecturers': 'lecturer', 'professor': 'lecturer', 'professors': 'lecturer',
    'prof': 'lecturer', 'tutor': 'lecturer', 'tutors': 'lecturer',
    'instructor': 'lecturer', 'instructors': 'lecturer', 'educator': 'lecturer',
    'educators': 'lecturer', 'academic': 'lecturer', 'academics': 'lecturer',
    'academician': 'lecturer', 'faculty': 'lecturer', 'faculties': 'lecturer',
    'teacher': 'lecturer', 'teachers': 'lecturer', 'lecturing': 'lecturer',

    # b. Kluster Finansial/Kesejahteraan -> Target Induk: salary & welfare
    'salaries': 'salary', 'wage': 'salary', 'wages': 'salary',
    'income': 'salary', 'incomes': 'salary', 'pay': 'salary',
    'paid': 'salary', 'paying': 'salary', 'paycheck': 'salary',
    'fee': 'salary', 'honorarium': 'salary', 'honorariums': 'salary',
    'honorary': 'salary', 'honor': 'salary', 'remuneration': 'salary',
    'compensation': 'salary', 'allowance': 'salary', 'allowances': 'salary',
    'stipend': 'salary', 'tukin': 'salary', 'bonus': 'salary',
    'bonuses': 'salary', 'reward': 'salary', 'rewards': 'salary',
    'pension': 'salary', 'pensions': 'salary',
    'prosperity': 'welfare', 'livelihood': 'welfare',

    # c. Kluster Institusi/Tempat Kerja -> Target Induk: institution
    'university': 'institution', 'universities': 'institution', 'college': 'institution',
    'colleges': 'institution', 'campus': 'institution', 'campuses': 'institution',
    'school': 'institution', 'schools': 'institution', 'institutions': 'institution',

    # d. Kluster Sistem & Status Kepegawaian -> Target Induk: system & status
    'application': 'system', 'app': 'system', 'apps': 'system', 'platform': 'system',
    'systems': 'system', 'sister': 'system', 'siakad': 'system', 'sinta': 'system',
    'bkd': 'system', 'dapodik': 'system', 'feeder': 'system',
    'pppk': 'status', 'pns': 'status', 'asn': 'status',
    'tenure': 'status', 'tenured': 'status',
    'gw': 'saya', 'ga': 'tidak', 'keknya': 'sepertinya', 
    'udh': 'sudah', 'gausa': 'tidak usah', 'jdi': 'jadi', 
    'nder': '' 
    } 
    kamus_norm.update(kamus_pribadi)
    
    return svm_model, tfidf_vectorizer, lda_model, lda_dict, kamus_norm

svm_model, tfidf_vectorizer, lda_model, lda_dict, kamus_norm = load_resources()

def normalize_text(text):
    words = text.lower().split()
    normalized_words = [kamus_norm.get(w, w) for w in words]
    return " ".join(normalized_words)

aspect_names = {
    0: "Kebijakan Nasional dan Regulasi Finansial",
    1: "Pernyataan Pejabat Pemerintah terkait Beban Fiskal",
    2: "Nominal Gaji Bulanan dalam Rupiah",
    3: "Pernyataan Kebijakan Anggaran Pendidikan dan Kesejahteraan",
    4: "Kualifikasi Akademik dan Persyaratan Gelar"
}

# 2. UI DASHBOARD
st.title("📊 Dashboard Analisis Aspek & Sentimen")
st.markdown("Analisis mendalam mengenai keluhan Dosen Non-ASN")

user_input = st.text_area("Masukkan ulasan:", placeholder="Contoh: Gaji bulan ini telat lagi...")
if st.button("Analisis Sekarang"):
    # Translasi
    text_normalized = normalize_text(user_input)
    
    # Tahap 2: Translasi
    translated = GoogleTranslator(source='id', target='en').translate(text_normalized)    
# Deteksi Aspek (LDA)
    tokens = translated.lower().split()
    bow = lda_dict.doc2bow(tokens)
    topics = lda_model.get_document_topics(bow) # Gunakan eksplisit getter
    
    # Ambil topik dengan probabilitas tertinggi
    if topics and len(topics) > 0:
        try:
            best_topic = max(topics, key=lambda x: x[1])[0]
            aspek = aspect_names.get(best_topic, "Lainnya")
        except (IndexError, TypeError, ValueError):
                aspek = "Tidak Terdeteksi"
    else:
            aspek = "Tidak Terdeteksi"
        
        # Tahap 4: Deteksi Sentimen (Diperbarui untuk model 3 kelas)
    tfidf = tfidf_vectorizer.transform([translated])
        
        # SVM langsung menebak kelasnya (Positif / Negatif / Netral)
    prediksi_label = svm_model.predict(tfidf)[0]
        
        # Mendapatkan nilai probabilitas / skor keyakinan dari tebakan
    probabilitas = svm_model.predict_proba(tfidf)[0]
    confidence = max(probabilitas) # Ambil probabilitas tertinggi

    # Logika Warna dan Ikon Sentimen
    if prediksi_label == "Positif":
        warna = "green"
        ikon = "😊"
    elif prediksi_label == "Negatif":
        warna = "red"
        ikon = "😡"
    else:
        warna = "gray" # Untuk Netral
        ikon = "😐"

    label_sentimen = prediksi_label

        # --- TAMPILAN BERSIH (Cards Layout) ---
    st.divider()
    col1, col2 = st.columns(2)
        
    with col1:
            st.subheader("📌 Aspek Terdeteksi")
            st.info(f"### {aspek}")
            
    with col2:
        st.subheader("⚖️ Sentimen")
        if warna == "red": st.error(f"### {ikon} {label_sentimen}")
        elif warna == "green": st.success(f"### {ikon} {label_sentimen}")
        else: st.warning(f"### {ikon} {label_sentimen}")

        # --- VISUALISASI CONFIDENCE (Progress Bar) ---
    st.subheader("💡 Tingkat Keyakinan Model")
        # Nilai confidence sudah berada direntang 0.0 - 1.0 sehingga langsung dimasukkan ke progress bar
    st.progress(float(confidence)) 
    st.write(f"Skor Keyakinan: **{confidence * 100:.2f}%**")
        
    st.divider()
    st.write(f"**Penjelasan Sistematis:** Teks ulasan Anda dikategorikan ke dalam aspek **{aspek}** dengan sentimen **{label_sentimen}** dan tingkat keyakinan empiris mesin sebesar **{confidence * 100:.2f}%**.")