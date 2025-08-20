import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from IndicTransToolkit.processor import IndicProcessor
from IndicTransToolkit import IndicProcessor
# from indictrans2 import IndicTrans2Translator
import pandas as pd
import io
import chardet

# Clear Streamlit cache
st.cache_resource.clear()

# Enhanced Custom CSS with more beautiful design elements
st.markdown("""
<style>
/* Modern gradient background with subtle pattern */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e3eeff 100%);
    background-image: url("https://www.transparenttextures.com/patterns/white-paper-fibers.png");
    font-size: 20px;
    font-family: 'Segoe UI', Roboto, sans-serif;
}
/* Fixed title bar with gradient */
.title-container {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background: linear-gradient(120deg, #1E3D59, #17A2B8);
    padding: 2rem;
    margin-top: 2rem;
    text-align: center;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.title-text {
    color: white !important;
    font-size: 36px;
    font-weight: bold;
    margin: 0;
    padding: 0;
}
/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(120deg, #1E3D59, #17A2B8) !important;
}
[data-testid="stSidebar"] > div {
    background: transparent !important;
}
/* Sidebar text and selectbox styling */
[data-testid="stSidebar"] .block-container {
    color: white !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: black !important;
}
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stMultiSelect > div {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
    color: black !important;
}
/* Main content padding to account for fixed header */
.main .block-container {
    padding-top: 5rem !important;
}
/* Main title styling with enhanced 3D effect and animated gradient */
.main-title {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(120deg, #0D47A1, #2196F3, #64B5F6, #1E88E5);
    background-size: 300% auto;
    color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    animation: gradient 8s ease infinite;
    text-align: center;
    padding: 30px 0 10px 0;
    margin-bottom: 10px;
    text-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    letter-spacing: -0.5px;
    position: relative;
}
/* Subtitle with elegant styling */
.subtitle {
    font-size: 20px;
    font-weight: 400;
    color: #546E7A;
    text-align: center;
    margin-bottom: 40px;
    font-style: italic;
    letter-spacing: 0.5px;
}
@keyframes gradient {
    0% {background-position: 0% 50%}
    50% {background-position: 100% 50%}
    100% {background-position: 0% 50%}
}
/* Glassmorphism card effect */
.glass-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    padding: 30px;
    margin-bottom: 30px;
}
/* Card-like containers with enhanced shadows */
.stSelectbox, .stMultiSelect {
    font-size: 22px !important;
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.05), 0 6px 6px rgba(0,0,0,0.07);
    margin-bottom: 25px;
    transition: all 0.3s ease;
}
.stSelectbox:hover, .stMultiSelect:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 28px rgba(0,0,0,0.08), 0 10px 10px rgba(0,0,0,0.07);
}
/* Enhanced buttons with gradient and animation */
.stButton > button {
    width: 100%;
    font-size: 20px;
    font-weight: 600;
    padding: 14px 30px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1976D2, #2196F3, #42A5F5);
    background-size: 200% auto;
    color: white;
    border: none;
    transition: all 0.4s ease;
    box-shadow: 0 6px 15px rgba(33,150,243,0.3);
    position: relative;
    overflow: hidden;
    z-index: 1;
}
.stButton > button:hover {
    background-position: right center;
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(33,150,243,0.4);
    font-size: 22px !important;
}
.stButton > button:active {
    transform: translateY(1px);
    box-shadow: 0 4px 12px rgba(33,150,243,0.3);
}
.stButton > button::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: 0.5s;
    z-index: -1;
}
.stButton > button:hover::after {
    left: 100%;
}
/* Text area styling with focus effects */
.stTextArea > div > div > textarea {
    font-size: 18px !important;
    padding: 18px !important;
    border-radius: 15px !important;
    border: 2px solid #E3F2FD;
    background: white;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}
.stTextArea > div > div > textarea:focus {
    border-color: #2196F3;
    box-shadow: 0 0 0 3px rgba(33,150,243,0.2);
}
/* Text area label styling */
.stTextArea label {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #1565C0 !important;
    margin-bottom: 8px !important;
    letter-spacing: 0.3px;
}
/* File uploader with enhanced styling */
.stFileUploader > div {
    font-size: 18px !important;
    background: white;
    padding: 30px;
    border-radius: 18px;
    border: 2px dashed #2196F3;
    transition: all 0.3s ease;
}
.stFileUploader > div:hover {
    border-color: #1565C0;
    background-color: #F5F9FF;
}
/* Translation result container with enhanced styling */
.translation-result {
    background: white;
    padding: 25px;
    border-radius: 18px;
    margin: 25px 0;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    border-left: 6px solid #2196F3;
    font-size: 18px !important;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.translation-result:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.12);
}
.translation-result::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg, rgba(33,150,243,0.05) 0%, transparent 100%);
    pointer-events: none;
}
/* Language tags with enhanced styling */
.language-tag {
    display: inline-block;
    padding: 6px 14px;
    margin: 5px;
    border-radius: 30px;
    background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
    color: #0D47A1;
    font-size: 16px;
    font-weight: 600;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.language-tag:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}
/* Progress bar with animation */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1E88E5, #64B5F6);
    height: 8px;
    border-radius: 4px;
    animation: shimmer 2s infinite linear;
    background-size: 200% 100%;
}
@keyframes shimmer {
    0% {background-position: -200% 0}
    100% {background-position: 200% 0}
}
/* Selectbox and multiselect styling */
.stSelectbox div[data-baseweb="select"] span,
.stMultiSelect div[data-baseweb="select"] span {
    font-size: 18px !important;
}
/* Tabs styling with enhanced effects */
.stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; padding: 0 10px; }
.stTabs [data-baseweb="tab"] {
    height: 60px;
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 12px;
    color: #1976D2;
    font-weight: 600;
    transition: all 0.3s ease;
    font-size: 18px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: white;
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.08);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #1976D2;
    color: white;
    box-shadow: 0 6px 15px rgba(25, 118, 210, 0.3);
}
/* Footer styling with enhanced design */
.footer {
    text-align: center;
    padding: 30px 20px;
    color: #546E7A;
    font-size: 16px;
    margin-top: 60px;
    border-top: 1px solid rgba(0,0,0,0.08);
    background: linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.8) 100%);
}
.footer-logo { font-size: 24px; font-weight: 700; color: #1976D2; margin-bottom: 10px; letter-spacing: 1px; }
/* Container width control */
.main-container { max-width: 75%; margin: 0 auto; padding: 1.5rem; }
/* Override Streamlit's default container */
.block-container {
    max-width: 65% !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    font-size: 22px !important;
}
/* Adjust padding for better spacing */
.stTabs [data-baseweb="tab-panel"] { padding: 1.5rem 0.5rem; }
/* Custom section headers */
.section-header { font-size: 28px; font-weight: 700; color: #1976D2; margin-bottom: 20px; }
/* Responsive design for smaller screens */
@media (max-width: 1200px) { .block-container { max-width: 90% !important; } }
@media (max-width: 768px) { .block-container { max-width: 100% !important; } }
</style>
""", unsafe_allow_html=True)


# Translator class
class IndicTranslator:
    def __init__(self, direction="en_to_indic"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.direction = direction
        if direction == "en_to_indic":
            self.model_name = "prajdabre/rotary-indictrans2-en-indic-dist-200M"
        else:
            self.model_name = "prajdabre/rotary-indictrans2-indic-en-dist-200M"

        self.processor = IndicProcessor(inference=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)

    def translate(self, text, src_lang, tgt_lang, max_length=2048, num_beams=10):
        if isinstance(text, str):
            text = [text]
        batch = self.processor.preprocess_batch(text, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.tokenizer(batch, padding="longest", truncation=True,
                                max_length=max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                num_beams=num_beams,
                length_penalty=1.5,
                repetition_penalty=2.0,
                num_return_sequences=1,
                max_new_tokens=max_length,
                early_stopping=True
            )
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations = self.processor.postprocess_batch(translations, lang=tgt_lang)
        return translations[0] if len(text) == 1 else translations

# Supported languages
INDIC_LANGUAGES = {
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym",
    "Kannada": "kan_Knda",
    "Gujarati": "guj_Gujr",
    "Marathi": "mar_Deva",
    "Punjabi": "pan_Guru",
    "Urdu": "urd_Arab",
}

# Initialize translators
if 'translator_en_to_indic' not in st.session_state:
    st.session_state.translator_en_to_indic = IndicTranslator(direction="en_to_indic")
if 'translator_indic_to_en' not in st.session_state:
    st.session_state.translator_indic_to_en = IndicTranslator(direction="indic_to_en")

# Title
st.markdown("<h1 style='text-align:center; color:#1976D2;'>Multilingual Text Translator 🌐</h1>", unsafe_allow_html=True)

# Sidebar language selection
st.sidebar.markdown('<h2 style="color:white;">Language Options</h2>', unsafe_allow_html=True)
source_lang = st.sidebar.selectbox("Source Language", ["English"] + list(INDIC_LANGUAGES.keys()))
target_langs = st.sidebar.multiselect("Target Languages", list(INDIC_LANGUAGES.keys()) if source_lang=="English" else ["English"])

# Tabs
tab1, tab2 = st.tabs(["📝 Text Translation", "📁 File Translation"])

# Text translation tab
with tab1:
    text_input = st.text_area("Enter text to translate:", height=270)
    if st.button("🔄 Translate", key="translate_text"):
        if not text_input.strip():
            st.error("Please enter some text.")
        else:
            for target_lang in target_langs:
                try:
                    if source_lang == "English":
                        translator = st.session_state.translator_en_to_indic
                        src = "eng_Latn"
                        tgt = INDIC_LANGUAGES[target_lang]
                    else:
                        translator = st.session_state.translator_indic_to_en
                        src = INDIC_LANGUAGES[source_lang]
                        tgt = "eng_Latn"
                    translated_text = translator.translate(text_input, src, tgt)
                    st.markdown(f"<div class='translation-result'><span class='language-tag'>{target_lang}</span><p>{translated_text}</p></div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error for {target_lang}: {str(e)}")

# File translation tab
with tab2:
    uploaded_file = st.file_uploader("Upload a TXT or XLSX file", type=["txt","xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.type == "text/plain":
                raw_data = uploaded_file.read()
                detected = chardet.detect(raw_data)
                text_content = raw_data.decode(detected['encoding'] or 'utf-8')
            else:
                df = pd.read_excel(uploaded_file)
                text_content = "\n".join(df.iloc[:,0].astype(str))
            st.text(text_content[:500] + ("..." if len(text_content)>500 else ""))

            if st.button("🔄 Translate File", key="translate_file"):
                source_lang_file = source_lang
                target_langs_file = target_langs
                all_translations = {}
                for target_lang in target_langs_file:
                    try:
                        if source_lang_file == "English":
                            translator = st.session_state.translator_en_to_indic
                            src = "eng_Latn"
                            tgt = INDIC_LANGUAGES[target_lang]
                        else:
                            translator = st.session_state.translator_indic_to_en
                            src = INDIC_LANGUAGES[source_lang_file]
                            tgt = "eng_Latn"
                        translated_text = translator.translate(text_content, src, tgt)
                        all_translations[target_lang] = translated_text
                        st.markdown(f"<div class='translation-result'><span class='language-tag'>{target_lang}</span><p>{translated_text}</p></div>", unsafe_allow_html=True)

                        # Download button
                        output = io.StringIO()
                        output.write(translated_text)
                        st.download_button(
                            label=f"Download {target_lang} Translation",
                            data=output.getvalue().encode("utf-8"),
                            file_name=f"translation_{target_lang}.txt",
                            mime="text/plain",
                            key=f"download_{target_lang}"
                        )
                    except Exception as e:
                        st.error(f"Error for {target_lang}: {str(e)}")

                # Combined download
                if len(all_translations) > 1:
                    combined_output = io.StringIO()
                    for lang, trans in all_translations.items():
                        combined_output.write(f"\n\n=== {lang} Translation ===\n\n{trans}")
                    st.download_button(
                        label="Download All Translations",
                        data=combined_output.getvalue().encode("utf-8"),
                        file_name="all_translations.txt",
                        mime="text/plain",
                        key="download_all"
                    )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
