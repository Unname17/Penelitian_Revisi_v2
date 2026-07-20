import streamlit as st
import joblib
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from deep_translator import GoogleTranslator
import numpy as np
import pandas as pd
import requests
from io import BytesIO

# --- 1. KONFIGURASI HALAMAN (HARUS DI ATAS) ---
# layout="centered" memastikan tampilan proporsional di tengah untuk PC, dan full-width di HP
st.set_page_config(page_title="Analisis Dosen", layout="centered", initial_sidebar_state="collapsed")

# =========================================================================
# 2. LOAD MODEL & RESOURCES
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
# VALIDASI 
# =========================================================================
model_classes = list(svm_model.classes_)
if len(model_classes) != 2:
    st.error(
        f"⚠️ Model yang dimuat punya {len(model_classes)} kelas ({model_classes}), "
        f"bukan 2 kelas seperti yang diharapkan. Pastikan 'svm_model.pkl' sudah "
        f"ditimpa dengan model hasil training 2 label (Positif/Negatif)."
    )
    st.stop()

# =========================================================================
# 3. FUNGSI & MAPPING
# =========================================================================
def normalize_text(text):
    words = text.lower().split()
    normalized_words = [kamus_norm.get(w, w) for w in words]
    return " ".join(normalized_words)

aspect_names = {
    0: "Kelembagaan & Aktor Pendidikan",
    1: "Beban & Kualitas Pendidikan",
    2: "Status Kontrak & Non-ASN",
    3: "Anggaran & Kebijakan Kesejahteraan",
    4: "Status Kepegawaian & Legalitas",
    5: "Kualifikasi Akademik & Kebijakan Sosial",
}

# --- 4. CSS KUSTOM (RESPONSIVE) ---
st.markdown("""
    <style>
    /* Latar belakang aplikasi */
    .stApp {
        background-color: #f4f6f9;
    }
    
    /* Sembunyikan top padding bawaan Streamlit (untuk Desktop) */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 800px; /* Lebar maksimal agar tidak terlalu memanjang di monitor besar */
    }
    header {visibility: hidden;}

    /* Styling Container / Kartu */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #eaedf0;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.03);
        padding: 1rem;
    }

    /* Styling Text Area */
    .stTextArea textarea {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
        color: #1f2937 !important;
    }

    /* Styling Button */
    button[kind="primary"] {
        background-color: #0f766e !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: 0.2s ease;
    }
    button[kind="primary"]:hover {
        background-color: #115e59 !important;
        transform: translateY(-1px);
    }

    /* Typografi Umum (Desktop) */
    .header-app { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
    .title-app { font-size: 20px; font-weight: 700; color: #111827; margin-left: 10px; display: flex; align-items: center;}
    .lbl-kecil { font-size: 11px; font-weight: 700; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 6px; margin-bottom: 8px;}
    .txt-aspek { font-size: 16px; font-weight: 700; color: #111827; line-height: 1.3; margin-bottom: 12px;}
    .badge-utama { background-color: #f3f4f6; color: #374151; font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 12px; display: inline-flex; align-items: center; gap: 6px; border: 1px solid #e5e7eb;}
    .dot { width: 6px; height: 6px; background-color: #0f766e; border-radius: 50%; }
    .txt-penjelasan { font-size: 14px; color: #4b5563; line-height: 1.6; }
    .txt-sentimen-hasil { font-size: 22px; font-weight: 700; text-align: center; margin-top: 15px; margin-bottom: 5px; }

    /* ========================================================
       MEDIA QUERIES (RESPONSIVITAS UNTUK HP / LAYAR KECIL) 
       ======================================================== */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        .title-app { 
            font-size: 18px !important; /* Judul lebih kecil di HP */
        }
        
        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.8rem !important; /* Kartu tidak perlu padding terlalu besar di HP */
        }
        
        .txt-aspek { 
            font-size: 14px !important; /* Teks aspek disesuaikan ukurannya */
        }
        
        .txt-sentimen-hasil { 
            font-size: 18px !important; /* Font sentimen lebih kecil */
            margin-top: 10px;
        }
        
        .txt-penjelasan { 
            font-size: 13px !important; /* Penjelasan lebih kecil agar mudah dibaca di HP */
        }
        
        button[kind="primary"] {
            font-size: 15px !important; /* Tombol disesuaikan ukurannya */
            padding: 0.5rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- 5. HEADER APLIKASI ---
st.markdown("""
    <div class="header-app">
        <div class="title-app">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 10H8V20H4V10ZM10 4H14V20H10V4ZM16 14H20V20H16V14Z" fill="#111827"/></svg>
            <span style="margin-left: 8px;">Analisis Dosen</span>
        </div>
        <img src="https://avatars.githubusercontent.com/u/151605727?v=4&size=40" width="36" height="36" style="border-radius: 50%; object-fit: cover;">
    </div>
""", unsafe_allow_html=True)

# --- 6. INPUT KARTU ---
with st.container(border=True):
    st.markdown('<div style="font-size: 16px; font-weight: 700; color: #111827; margin-bottom: 12px;">Input Data Aspirasi</div>', unsafe_allow_html=True)
    user_input = st.text_area("input_label", placeholder="Masukkan ulasan / postingan di sini (Contoh: Gaji bulan ini telat lagi...)", height=120, label_visibility="collapsed")
    btn_analisis = st.button("🔍 Mulai Analisis", type="primary", use_container_width=True)

# --- 7. LOGIKA & OUTPUT KARTU ---
if btn_analisis:
    if user_input.strip() == "":
        st.warning("⚠️ Masukkan teks aspirasi terlebih dahulu.")
    else:
        # Normalisasi & Translasi
        text_normalized = normalize_text(user_input)
        try:
            translated = GoogleTranslator(source='id', target='en').translate(text_normalized)
        except Exception as e:
            st.error(f"❌ Gagal menerjemahkan teks: {e}")
            st.stop()
        
        # LDA Topik
        tokens = translated.lower().split()
        bow = lda_dict.doc2bow(tokens)
        topics = lda_model.get_document_topics(bow)
        
        if topics and len(topics) > 0:
            try:
                best_topic = max(topics, key=lambda x: x[1])[0]
                aspek = aspect_names.get(best_topic, "Lainnya")
            except:
                aspek = "Tidak Terdeteksi"
        else:
            aspek = "Tidak Terdeteksi"
        
        # SVM Klasifikasi (Teks Gabungan)
        combined_text = aspek + " " + translated
        tfidf = tfidf_vectorizer.transform([combined_text])
        
        feature_names = tfidf_vectorizer.get_feature_names_out()
        nonzero_idx = tfidf.nonzero()[1]
        matched_terms = [feature_names[i] for i in nonzero_idx]

        prediksi_label = svm_model.predict(tfidf)[0]
        
        # Kalkulasi Confidence
        try:
            probabilitas = svm_model.predict_proba(tfidf)[0]
            confidence = max(probabilitas)
        except AttributeError:
            decision = svm_model.decision_function(tfidf)[0]
            if np.isscalar(decision) or np.ndim(decision) == 0:
                prob_positive_class = 1 / (1 + np.exp(-decision))
                probabilitas = np.array([1 - prob_positive_class, prob_positive_class])
            else:
                exp_scores = np.exp(decision - np.max(decision))
                probabilitas = exp_scores / np.sum(exp_scores)
            confidence = max(probabilitas)

        # Mapping Warna Tampilan
        if prediksi_label == "Positif":
            label_sentimen = "Positif"
            warna_hex = "#0f766e" # Teal
        elif prediksi_label == "Negatif":
            label_sentimen = "Negatif"
            warna_hex = "#ef4444" # Merah
        else:
            label_sentimen = "Ambigu / Netral"
            warna_hex = "#6b7280"

        penjelasan = f"Aspirasi menunjukkan sentimen {label_sentimen.lower()} terhadap aspek {aspek.lower()}."
        accuracy = int(min(confidence, 1.0) * 100)

        # Layout Output
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown(f"""
                <div class="lbl-kecil">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>
                    ASPEK TERDETEKSI
                </div>
                <div class="txt-aspek">{aspek}</div>
                <div class="badge-utama"><div class="dot"></div> Utama</div>
                """, unsafe_allow_html=True)
                
        with col2:
            with st.container(border=True):
                st.markdown(f"""
                <div class="lbl-kecil" style="justify-content: center;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                    HASIL SENTIMEN
                </div>
                <div class="txt-sentimen-hasil" style="color: {warna_hex};">{label_sentimen}</div>
                """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f"""
            <div class="lbl-kecil" style="color: #111827; margin-bottom: 12px; font-size: 13px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#111827" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                PENJELASAN ANALISIS
            </div>
            <div class="txt-penjelasan">{penjelasan}</div>
            """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 13px; font-weight: 600; color: #4b5563; display: flex; align-items: center; gap: 8px;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#4b5563" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                    AI Confidence Score
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-weight: 700; color: #111827; font-size: 14px;">{accuracy}% Accuracy</div>
                    <div style="width: 40px; height: 3px; background-color: #111827; border-radius: 2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Fitur Expander Detail
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📖 Lihat Detail Proses Analisis Model AI"):
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

github_url = "https://github.com/Unname17/Penelitian_Revisi_v2"

st.markdown(f"""
    <div style="text-align: center; font-size: 12px; color: #6b7280; margin-top: 20px;">
        <a href="{github_url}" target="_blank" style="color: #6b7280; text-decoration: none; transition: 0.3s;">
            © 2026 - Analisis Sentimen Dosen Non-ASN | Dibangun dengan Streamlit, LDA, dan SVM By <b>Muhajir Kelana Saputra</b>
        </a>
    </div>
    
    <style>
        /* Efek hover agar teks berubah warna sedikit saat kursor berada di atasnya */
        a:hover {{
            color: #0f766e !important;
            text-decoration: underline !important;
        }}
    </style>
""", unsafe_allow_html=True)