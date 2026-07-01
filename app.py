import streamlit as st
import joblib
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from deep_translator import GoogleTranslator
import numpy as np
import pandas as pd
import requests
from io import BytesIO

# =========================================================================
# 1. LOAD MODEL & RESOURCES
# =========================================================================
@st.cache_resource(ttl=600)  # cache akan disegarkan setiap 10 menit
def load_resources():
    # Load model SVM & TF-IDF
    svm_model = joblib.load('svm_model_valid.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer_valid.pkl')

    # Load model LDA & Dictionary
    lda_model = LdaModel.load('lda_model.model')
    lda_dict = Dictionary.load('lda_dictionary.dict')

    # Load kamus normalisasi dari GitHub
    url = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
    response = requests.get(url)
    kamus_data = pd.read_excel(BytesIO(response.content))

    # Membuat dict dari kolom 'tidak_baku' ke 'kata_baku'
    kamus_norm = dict(zip(
        kamus_data.iloc[:, 0].astype(str).str.lower(),
        kamus_data.iloc[:, 1].astype(str)
    ))

    # Tambahkan kamus pribadi (khusus konteks dosen non-ASN)
    kamus_pribadi = {
        # Kluster Dosen
        'lecturers': 'lecturer', 'professor': 'lecturer', 'professors': 'lecturer',
        'prof': 'lecturer', 'tutor': 'lecturer', 'tutors': 'lecturer',
        'instructor': 'lecturer', 'instructors': 'lecturer', 'educator': 'lecturer',
        'educators': 'lecturer', 'academic': 'lecturer', 'academics': 'lecturer',
        'academician': 'lecturer', 'faculty': 'lecturer', 'faculties': 'lecturer',
        'teacher': 'lecturer', 'teachers': 'lecturer', 'lecturing': 'lecturer',

        # Kluster Finansial/Kesejahteraan
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

        # Kluster Institusi
        'university': 'institution', 'universities': 'institution', 'college': 'institution',
        'colleges': 'institution', 'campus': 'institution', 'campuses': 'institution',
        'school': 'institution', 'schools': 'institution', 'institutions': 'institution',

        # Kluster Sistem & Status Kepegawaian
        'application': 'system', 'app': 'system', 'apps': 'system', 'platform': 'system',
        'systems': 'system', 'sister': 'system', 'siakad': 'system', 'sinta': 'system',
        'bkd': 'system', 'dapodik': 'system', 'feeder': 'system',
        'pppk': 'status', 'pns': 'status', 'asn': 'status',
        'tenure': 'status', 'tenured': 'status',

        # Slang/gaul Indonesia
        'gw': 'saya', 'ga': 'tidak', 'keknya': 'sepertinya',
        'udh': 'sudah', 'gausa': 'tidak usah', 'jdi': 'jadi',
        'nder': ''
    }
    kamus_norm.update(kamus_pribadi)

    return svm_model, tfidf_vectorizer, lda_model, lda_dict, kamus_norm

# Muat semua resource
svm_model, tfidf_vectorizer, lda_model, lda_dict, kamus_norm = load_resources()

# =========================================================================
# 2. FUNGSI NORMALISASI TEKS
# =========================================================================
def normalize_text(text):
    words = text.lower().split()
    normalized_words = [kamus_norm.get(w, w) for w in words]
    return " ".join(normalized_words)

# =========================================================================
# 3. MAPPING ASPEK LDA
# =========================================================================
aspect_names = {
    0: "Kebijakan Nasional dan Regulasi Finansial",
    1: "Pernyataan Pejabat Pemerintah terkait Beban Fiskal",
    2: "Nominal Gaji Bulanan dalam Rupiah",
    3: "Pernyataan Kebijakan Anggaran Pendidikan dan Kesejahteraan",
    4: "Kualifikasi Akademik dan Persyaratan Gelar"
}

# =========================================================================
# 4. UI DASHBOARD STREAMLIT
# =========================================================================
st.set_page_config(page_title="Analisis Sentimen Dosen Non-ASN", layout="wide")
st.title("📊 Dashboard Analisis Aspek & Sentimen")
st.markdown("Analisis mendalam mengenai keluhan Dosen Non-ASN berdasarkan postingan media sosial X")

user_input = st.text_area("✍️ Masukkan ulasan / postingan:", placeholder="Contoh: Gaji bulan ini telat lagi, padahal sudah kerja keras...")

if st.button("🔍 Analisis Sekarang", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")
        st.stop()

    # =====================================================================
    # STEP 1: Normalisasi Teks (kata tidak baku → baku)
    # =====================================================================
    text_normalized = normalize_text(user_input)

    # =====================================================================
    # STEP 2: Translasi Indonesia → Inggris (karena model SVM dilatih dengan teks Inggris)
    # =====================================================================
    try:
        translated = GoogleTranslator(source='id', target='en').translate(text_normalized)
    except Exception as e:
        st.error(f"❌ Gagal menerjemahkan teks: {e}")
        st.stop()

    # =====================================================================
    # STEP 3: Deteksi Aspek dengan LDA
    # =====================================================================
    tokens = translated.lower().split()
    bow = lda_dict.doc2bow(tokens)
    topics = lda_model.get_document_topics(bow)

    if topics and len(topics) > 0:
        try:
            best_topic = max(topics, key=lambda x: x[1])[0]
            aspek = aspect_names.get(best_topic, "Lainnya")
        except (IndexError, TypeError, ValueError):
            aspek = "Tidak Terdeteksi"
    else:
        aspek = "Tidak Terdeteksi"

    # =====================================================================
    # STEP 4: Deteksi Sentimen dengan SVM (3 kelas: Positif / Negatif / Netral)
    # =====================================================================
    tfidf = tfidf_vectorizer.transform([translated])

    # Prediksi label
    prediksi_label = svm_model.predict(tfidf)[0]

    # Ambil probabilitas / tingkat keyakinan
    try:
        # Jika model memiliki predict_proba (misal SVC dengan probability=True)
        probabilitas = svm_model.predict_proba(tfidf)[0]
        confidence = max(probabilitas)
    except AttributeError:
        # Fallback untuk LinearSVC: gunakan decision_function + softmax
        decision = svm_model.decision_function(tfidf)[0]  # shape: (n_classes,)
        exp_scores = np.exp(decision - np.max(decision))  # stabilisasi numerik
        probabilitas = exp_scores / np.sum(exp_scores)
        confidence = max(probabilitas)

    # =====================================================================
    # STEP 5: Tampilan Hasil
    # =====================================================================

    # Tentukan warna dan ikon berdasarkan sentimen
    if prediksi_label == "Positif":
        warna = "green"
        ikon = "😊"
    elif prediksi_label == "Negatif":
        warna = "red"
        ikon = "😡"
    else:  # Netral
        warna = "gray"
        ikon = "😐"

    label_sentimen = prediksi_label

    # --- TAMPILKAN HASIL ---
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Aspek Terdeteksi")
        st.info(f"### {aspek}")
        st.caption(f"Berdasarkan pemodelan topik LDA dari 5 aspek yang tersedia")

    with col2:
        st.subheader("⚖️ Sentimen")
        if warna == "red":
            st.error(f"### {ikon} {label_sentimen}")
        elif warna == "green":
            st.success(f"### {ikon} {label_sentimen}")
        else:
            st.warning(f"### {ikon} {label_sentimen}")
        st.caption("Klasifikasi menggunakan Support Vector Machine (SVM)")

    # --- Tampilkan Tingkat Keyakinan ---
    st.subheader("💡 Tingkat Keyakinan Model")
    st.progress(float(confidence))
    st.write(f"Skor Keyakinan: **{confidence * 100:.2f}%**")

    # --- Tampilkan Detail Proses ---
    with st.expander("📖 Lihat Detail Proses Analisis"):
        st.markdown("**Teks Normalisasi:**")
        st.code(text_normalized, language="text")
        st.markdown("**Teks Terjemahan (Inggris):**")
        st.code(translated, language="text")
        st.markdown("**Distribusi Probabilitas Sentimen:**")
        # Tampilkan probabilitas per kelas
        try:
            kelas = svm_model.classes_
        except AttributeError:
            kelas = ['Negatif', 'Netral', 'Positif']
        df_proba = pd.DataFrame({
            'Kelas': kelas,
            'Probabilitas': probabilitas
        })
        st.dataframe(df_proba, use_container_width=True)

    # --- Penjelasan Sistematis ---
    st.divider()
    st.write(f"""
    **📝 Penjelasan Sistematis:**
    Teks ulasan Anda dikategorikan ke dalam aspek **{aspek}** dengan sentimen **{label_sentimen}**
    dan tingkat keyakinan empiris mesin sebesar **{confidence * 100:.2f}%**.
    """)

    st.success("✅ Analisis selesai!")

# =========================================================================
# FOOTER
# =========================================================================
st.divider()
st.caption("© 2026 - Analisis Sentimen Dosen Non-ASN | Dibangun dengan Streamlit, LDA, dan SVM")
