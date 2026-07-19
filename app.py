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
    svm_model = joblib.load('svm_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    lda_model = LdaModel.load('lda_model.model')
    lda_dict = Dictionary.load('lda_dictionary.dict')

    url = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
    response = requests.get(url)
    kamus_data = pd.read_excel(BytesIO(response.content))
    kamus_norm = dict(zip(
        kamus_data.iloc[:, 0].astype(str).str.lower(),
        kamus_data.iloc[:, 1].astype(str)
    ))

    kamus_pribadi = {
        'lecturers': 'lecturer', 'professor': 'lecturer', 'professors': 'lecturer',
        'prof': 'lecturer', 'tutor': 'lecturer', 'tutors': 'lecturer',
        'instructor': 'lecturer', 'instructors': 'lecturer', 'educator': 'lecturer',
        'educators': 'lecturer', 'academic': 'lecturer', 'academics': 'lecturer',
        'academician': 'lecturer', 'faculty': 'lecturer', 'faculties': 'lecturer',
        'teacher': 'lecturer', 'teachers': 'lecturer', 'lecturing': 'lecturer',
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
        'university': 'institution', 'universities': 'institution', 'college': 'institution',
        'colleges': 'institution', 'campus': 'institution', 'campuses': 'institution',
        'school': 'institution', 'schools': 'institution', 'institutions': 'institution',
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

# =========================================================================
# VALIDASI: pastikan model yang ter-load memang cuma 2 kelas
# =========================================================================
model_classes = list(svm_model.classes_)
if len(model_classes) != 2:
    st.error(
        f"⚠️ Model yang dimuat punya {len(model_classes)} kelas ({model_classes}), "
        f"bukan 2 kelas seperti yang diharapkan. Pastikan 'svm_model.pkl' sudah "
        f"ditimpa dengan model hasil training 2 label (Positif/Negatif), bukan "
        f"file lama yang masih 3 kelas (Positif/Negatif/Netral)."
    )
    st.stop()

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
    0: "Kelembagaan & Aktor Pendidikan",
    1: "Beban & Kualitas Pendidikan",
    2: "Status Kontrak & Non-ASN",
    3: "Anggaran & Kebijakan Kesejahteraan",
    4: "Status Kepegawaian & Legalitas",
    5: "Kualifikasi Akademik & Kebijakan Sosial",
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

    # STEP 1: Normalisasi Teks
    text_normalized = normalize_text(user_input)

    # STEP 2: Translasi Indonesia -> Inggris
    try:
        translated = GoogleTranslator(source='id', target='en').translate(text_normalized)
    except Exception as e:
        st.error(f"❌ Gagal menerjemahkan teks: {e}")
        st.stop()

    # STEP 3: Deteksi Aspek dengan LDA
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

    # STEP 4: Deteksi Sentimen dengan SVM (2 kelas: Positif / Negatif)
    combined_text = aspek + " " + translated
    tfidf = tfidf_vectorizer.transform([combined_text])

    feature_names = tfidf_vectorizer.get_feature_names_out()
    nonzero_idx = tfidf.nonzero()[1]
    matched_terms = [feature_names[i] for i in nonzero_idx]

    prediksi_label = svm_model.predict(tfidf)[0]
    try:
        probabilitas = svm_model.predict_proba(tfidf)[0]
        confidence = max(probabilitas)
    except AttributeError:
        decision = svm_model.decision_function(tfidf)[0]
        # decision_function untuk 2 kelas mengembalikan skalar, bukan array,
        # jadi perlu ditangani berbeda dari kasus multi-kelas
        if np.isscalar(decision) or np.ndim(decision) == 0:
            # skor positif -> kelas kedua (classes_[1]), skor negatif -> classes_[0]
            prob_positive_class = 1 / (1 + np.exp(-decision))
            probabilitas = np.array([1 - prob_positive_class, prob_positive_class])
        else:
            exp_scores = np.exp(decision - np.max(decision))
            probabilitas = exp_scores / np.sum(exp_scores)
        confidence = max(probabilitas)

    # STEP 5: Tampilan Hasil
    if prediksi_label == "Positif":
        warna = "green"
        ikon = "😊"
    elif prediksi_label == "Negatif":
        warna = "red"
        ikon = "😡"
    else:
        # fallback jaga-jaga, seharusnya tidak pernah tercapai karena sudah
        # divalidasi di atas bahwa model cuma 2 kelas
        warna = "gray"
        ikon = "❓"

    label_sentimen = prediksi_label

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Aspek Terdeteksi")
        st.info(f"### {aspek}")
        st.caption("Berdasarkan pemodelan topik LDA")
    with col2:
        st.subheader("⚖️ Sentimen")
        if warna == "red":
            st.error(f"### {ikon} {label_sentimen}")
        elif warna == "green":
            st.success(f"### {ikon} {label_sentimen}")
        else:
            st.warning(f"### {ikon} {label_sentimen}")
        st.caption("Klasifikasi menggunakan Support Vector Machine (SVM) — 2 kelas")

    st.subheader("💡 Tingkat Keyakinan Model")
    st.progress(float(confidence))
    st.write(f"Skor Keyakinan: **{confidence * 100:.2f}%**")

    with st.expander("📖 Lihat Detail Proses Analisis"):
        st.markdown("**Teks Normalisasi:**")
        st.code(text_normalized, language="text")
        st.markdown("**Teks Terjemahan (Inggris):**")
        st.code(translated, language="text")
        st.markdown("**Teks Gabungan (Aspek + Terjemahan) yang masuk ke TF-IDF:**")
        st.code(combined_text, language="text")
        st.markdown("**Term yang dikenali TF-IDF vectorizer:**")
        st.write(f"Jumlah term yang match: {len(matched_terms)}")
        st.write(matched_terms)
        st.markdown("**Distribusi Probabilitas Sentimen:**")
        df_proba = pd.DataFrame({'Kelas': model_classes, 'Probabilitas': probabilitas})
        st.dataframe(df_proba, use_container_width=True)

    st.divider()
    st.write(f"""
    **📝 Penjelasan Sistematis:**
    Teks ulasan Anda dikategorikan ke dalam aspek **{aspek}** dengan sentimen **{label_sentimen}**
    dan tingkat keyakinan empiris mesin sebesar **{confidence * 100:.2f}%**.
    """)
    st.success("✅ Analisis selesai!")

st.divider()
st.caption("© 2026 - Analisis Sentimen Dosen Non-ASN | Dibangun dengan Streamlit, LDA, dan SVM")