<<<<<<< HEAD
# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# # from IndicTransToolkit.processor import IndicProcessor

# # import sys
# # import os
# # sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from IndicTransToolkit.processor import IndicProcessor

# class IndicTranslator:
#     def __init__(self, model_name="prajdabre/rotary-indictrans2-en-indic-dist-200M"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # Initialize processor, tokenizer, and model
#         self.processor = IndicProcessor(inference=True)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             trust_remote_code=True
#         ).to(self.device)

#     def translate(self, text, tgt_lang, max_length=2048, num_beams=10):
#         if isinstance(text, str):
#             text = [text]
        
#         # Preprocess the input text
#         batch = self.processor.preprocess_batch(text, src_lang="eng_Latn", tgt_lang=tgt_lang)
        
#         # Tokenize
#         inputs = self.tokenizer(
#             batch,
#             padding="longest",
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt"
#         ).to(self.device)
        
#         # Generate translation
#         with torch.inference_mode():
#             outputs = self.model.generate(
#                 **inputs,
#                 num_beams=num_beams,
#                 length_penalty=1.5,
#                 repetition_penalty=2.0,
#                 num_return_sequences=1,
#                 max_new_tokens=max_length,
#                 early_stopping=True
#             )
        
#         # Decode the outputs
#         translations = self.tokenizer.batch_decode(
#             outputs,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=True
#         )
        
#         # Postprocess the translations
#         translations = self.processor.postprocess_batch(translations, lang=tgt_lang)
        
#         return translations[0] if len(text) == 1 else translations

# # Supported language pairs
# LANGUAGE_PAIRS = {
#     "English to Hindi": "hin_Deva",
#     "English to Bengali": "ben_Beng",
#     "English to Tamil": "tam_Taml",
#     "English to Telugu": "tel_Telu",
#     "English to Malayalam": "mal_Mlym",
#     "English to Kannada": "kan_Knda",
#     "English to Gujarati": "guj_Gujr",
#     "English to Marathi": "mar_Deva",
#     "English to Punjabi": "pan_Guru",
#     "English to Urdu": "urd_Arab",
# }

# def main():
#     st.set_page_config(page_title="Indic Language Translator", page_icon="🌏")
    
#     st.title("🌏 Indic Language Translator")
#     st.markdown("Translate English text to various Indian languages")

#     # Initialize the translator
#     @st.cache_resource
#     def load_translator():
#         return IndicTranslator()

#     translator = load_translator()

#     # Language selection
#     language_pair = st.selectbox(
#         "Select target language:",
#         list(LANGUAGE_PAIRS.keys())
#     )

#     # Text input
#     input_text = st.text_area("Enter English text:", height=200)

#     # Translation button
#     if st.button("Translate"):
#         if input_text:
#             with st.spinner("Translating..."):
#                 try:
#                     tgt_lang = LANGUAGE_PAIRS[language_pair]
#                     translation = translator.translate(
#                         input_text,
#                         tgt_lang=tgt_lang
#                     )
                    
#                     st.success("Translation completed!")
#                     st.markdown("### Translation:")
#                     st.write(translation)
#                 except Exception as e:
#                     st.error(f"An error occurred during translation: {str(e)}")
#         else:
#             st.warning("Please enter some text to translate.")

#     # Additional information
#     with st.expander("About this translator"):
#         st.write("""
#         This translator uses the IndicTrans2 model from AI4Bharat to translate English text into various Indian languages.
#         - Supports long text translation up to 2048 tokens
#         - Uses beam search for better translation quality
#         - Handles multiple Indian languages and scripts
#         """)

# if __name__ == "__main__":
#     main()










#indic to eng and vice versa text

# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from IndicTransToolkit.processor import IndicProcessor

# # Clear Streamlit cache
# st.cache_resource.clear()

# class IndicTranslator:
#     def __init__(self, direction="en_to_indic"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.direction = direction
        
#         # Initialize only the required model based on direction
#         if direction == "en_to_indic":
#             self.model_name = "prajdabre/rotary-indictrans2-en-indic-dist-200M"
#         else:
#             self.model_name = "prajdabre/rotary-indictrans2-indic-en-dist-200M"
        
#         # Initialize processor
#         self.processor = IndicProcessor(inference=True)
        
#         # Initialize tokenizer and model for the specified direction
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(
#             self.model_name,
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             trust_remote_code=True
#         ).to(self.device)

#     def translate(self, text, src_lang, tgt_lang, max_length=2048, num_beams=10):
#         if isinstance(text, str):
#             text = [text]
        
#         # Preprocess the input text
#         batch = self.processor.preprocess_batch(text, src_lang=src_lang, tgt_lang=tgt_lang)
        
#         # Tokenize
#         inputs = self.tokenizer(
#             batch,
#             padding="longest",
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt"
#         ).to(self.device)
        
#         # Generate translation
#         with torch.inference_mode():
#             outputs = self.model.generate(
#                 **inputs,
#                 num_beams=num_beams,
#                 length_penalty=1.5,
#                 repetition_penalty=2.0,
#                 num_return_sequences=1,
#                 max_new_tokens=max_length,
#                 early_stopping=True
#             )
        
#         # Decode the outputs
#         translations = self.tokenizer.batch_decode(
#             outputs,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=True
#         )
        
#         # Postprocess the translations
#         translations = self.processor.postprocess_batch(translations, lang=tgt_lang)
        
#         return translations[0] if len(text) == 1 else translations

# # Supported languages
# INDIC_LANGUAGES = {
#     "Hindi": "hin_Deva",
#     "Bengali": "ben_Beng",
#     "Tamil": "tam_Taml",
#     "Telugu": "tel_Telu",
#     "Malayalam": "mal_Mlym",
#     "Kannada": "kan_Knda",
#     "Gujarati": "guj_Gujr",
#     "Marathi": "mar_Deva",
#     "Punjabi": "pan_Guru",
#     "Urdu": "urd_Arab",
# }

# def main():
#     st.set_page_config(page_title="Indic Language Translator", page_icon="🌏")
    
#     st.title("Indic Language Translator")
#     st.markdown("Translate between English and Indian languages")

#     # Translation direction
#     translation_direction = st.radio(
#         "Select translation direction:",
#         ["English to Indian Language", "Indian Language to English"],
#         key="translation_direction"
#     )

#     # Initialize the appropriate translator based on direction
#     @st.cache_resource
#     def load_translator(direction):
#         return IndicTranslator(direction="en_to_indic" if direction == "English to Indian Language" else "indic_to_en")

#     translator = load_translator(translation_direction)

#     if translation_direction == "English to Indian Language":
#         target_lang = st.selectbox(
#             "Select target language:",
#             list(INDIC_LANGUAGES.keys())
#         )
#         input_label = "Enter English text:"
#         target_lang_code = INDIC_LANGUAGES[target_lang]
#         source_lang_code = "eng_Latn"
#     else:
#         source_lang = st.selectbox(
#             "Select source language:",
#             list(INDIC_LANGUAGES.keys())
#         )
#         input_label = f"Enter text in {source_lang}:"
#         source_lang_code = INDIC_LANGUAGES[source_lang]
#         target_lang_code = "eng_Latn"

#     # Text input
#     input_text = st.text_area(input_label, height=200)

#     # Translation button
#     if st.button("Translate"):
#         if input_text:
#             with st.spinner("Translating..."):
#                 try:
#                     translation = translator.translate(
#                         input_text,
#                         src_lang=source_lang_code,
#                         tgt_lang=target_lang_code
#                     )
                    
#                     st.success("Translation completed!")
#                     st.markdown("### Translation:")
#                     st.write(translation)
#                 except Exception as e:
#                     st.error(f"An error occurred during translation: {str(e)}")
#         else:
#             st.warning("Please enter some text to translate.")

#     # Additional information
#     with st.expander("About this translator"):
#         st.write("""
#         This translator uses the IndicTrans2 model from AI4Bharat to translate between English and Indian languages.
#         - Supports long text translation up to 2048 tokens
#         - Uses beam search for better translation quality
#         - Handles multiple Indian languages and scripts
#         - Optimized to load only the required model based on translation direction
#         """)

# if __name__ == "__main__":
#     main()












# csv columns translation for single lang at one time
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import pandas as pd
import io
import chardet

# Clear Streamlit cache
st.cache_resource.clear()

class IndicTranslator:
    def __init__(self, direction="en_to_indic"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.direction = direction
        
        # Initialize only the required model based on direction
        if direction == "en_to_indic":
            self.model_name = "prajdabre/rotary-indictrans2-en-indic-dist-200M"
        else:
            self.model_name = "prajdabre/rotary-indictrans2-indic-en-dist-200M"
        
        # Initialize processor
        self.processor = IndicProcessor(inference=True)
        
        # Initialize tokenizer and model for the specified direction
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)

    def translate(self, text, src_lang, tgt_lang, max_length=2048, num_beams=10):
        if isinstance(text, str):
            text = [text]
        
        # Preprocess the input text
        batch = self.processor.preprocess_batch(text, src_lang=src_lang, tgt_lang=tgt_lang)
        
        # Tokenize
        inputs = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate translation
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
        
        # Decode the outputs
        translations = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Postprocess the translations
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

def detect_encoding(file_content):
    """Detect the encoding of the file"""
    result = chardet.detect(file_content)
    return result['encoding']

def read_file_with_encoding(uploaded_file):
    """Read file with proper encoding for Indic text"""
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            # For Excel files, use openpyxl engine for better Unicode support
            uploaded_file.seek(0)
            return pd.read_excel(
                uploaded_file,
                engine='openpyxl',
                dtype=str  # Read all columns as string to preserve Indic text
            )
        else:
            # For CSV files
            content = uploaded_file.read()
            encoding = detect_encoding(content)
            encodings_to_try = [encoding, 'utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252']
            
            for enc in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file,
                        encoding=enc,
                        dtype=str  # Read all columns as string
                    )
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    st.error(f"Error reading file with {enc} encoding: {str(e)}")
            
            raise Exception("Unable to read file with any known encoding")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def process_file_en_to_indic(df, text_column, translator, target_lang_codes):
    """Process the dataframe and translate from English to multiple Indic languages"""
    result_df = df.copy()
    source_lang_code = "eng_Latn"
    
    total_languages = len(target_lang_codes)
    progress_placeholder = st.empty()
    
    for i, target_lang_code in enumerate(target_lang_codes):
        target_lang_name = next((k for k, v in INDIC_LANGUAGES.items() if v == target_lang_code), target_lang_code)
        progress_placeholder.text(f"Translating to {target_lang_name} ({i+1}/{total_languages})")
        
        translations = []
        errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows = len(df)
        for idx, text in enumerate(df[text_column]):
            try:
                if pd.isna(text) or str(text).strip() == '':
                    translations.append("")
                    errors.append("")
                    continue
                
                # Clean the input text
                cleaned_text = str(text).strip()
                
                translation = translator.translate(
                    cleaned_text,
                    src_lang=source_lang_code,
                    tgt_lang=target_lang_code
                )
                translations.append(translation)
                errors.append("")
                
            except Exception as e:
                translations.append("")
                errors.append(f"Error: {str(e)}")
            
            # Update progress
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1} of {total_rows} for {target_lang_name}")
        
        # Add translations and errors to the dataframe
        result_df[f'{text_column}_translated_{target_lang_name}'] = translations
        result_df[f'{text_column}_translation_errors_{target_lang_name}'] = errors
        
        # Remove error column if no errors occurred
        if not any(errors):
            result_df = result_df.drop(f'{text_column}_translation_errors_{target_lang_name}', axis=1)
    
    progress_placeholder.text("Translation completed for all languages!")
    return result_df

def process_file_indic_to_en(df, text_column, translator, source_lang_codes):
    """Process the dataframe and translate from multiple Indic languages to English"""
    result_df = df.copy()
    target_lang_code = "eng_Latn"
    
    # Create a new column for each source language
    for source_lang_code in source_lang_codes:
        source_lang_name = next((k for k, v in INDIC_LANGUAGES.items() if v == source_lang_code), source_lang_code)
        
        translations = []
        errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Translating from {source_lang_name} to English")
        
        total_rows = len(df)
        for idx, text in enumerate(df[text_column]):
            try:
                if pd.isna(text) or str(text).strip() == '':
                    translations.append("")
                    errors.append("")
                    continue
                
                # Clean the input text
                cleaned_text = str(text).strip()
                
                translation = translator.translate(
                    cleaned_text,
                    src_lang=source_lang_code,
                    tgt_lang=target_lang_code
                )
                translations.append(translation)
                errors.append("")
                
            except Exception as e:
                translations.append("")
                errors.append(f"Error: {str(e)}")
            
            # Update progress
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1} of {total_rows} from {source_lang_name}")
        
        # Add translations and errors to the dataframe
        result_df[f'{text_column}_translated_from_{source_lang_name}'] = translations
        result_df[f'{text_column}_translation_errors_from_{source_lang_name}'] = errors
        
        # Remove error column if no errors occurred
        if not any(errors):
            result_df = result_df.drop(f'{text_column}_translation_errors_from_{source_lang_name}', axis=1)
    
    return result_df

def main():
    st.set_page_config(page_title="Indic Language Translator", page_icon="🌏", layout="wide")
    
    st.title("Indic Language Translator")
    st.markdown("Translate between English and Indian languages")

    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Text Input", "File Upload"],
        key="input_method"
    )

    # Translation direction
    translation_direction = st.radio(
        "Select translation direction:",
        ["English to Indian Language", "Indian Language to English"],
        key="translation_direction"
    )

    @st.cache_resource
    def load_translator(direction):
        return IndicTranslator(direction="en_to_indic" if direction == "English to Indian Language" else "indic_to_en")

    translator = load_translator(translation_direction)

    if translation_direction == "English to Indian Language":
        # Multiple target languages
        target_langs = st.multiselect(
            "Select target language(s):",
            list(INDIC_LANGUAGES.keys()),
            default=[list(INDIC_LANGUAGES.keys())[0]]  # Default to first language
        )
        target_lang_codes = [INDIC_LANGUAGES[lang] for lang in target_langs]
        source_lang_code = "eng_Latn"
    else:
        # Multiple source languages
        source_langs = st.multiselect(
            "Select source language(s):",
            list(INDIC_LANGUAGES.keys()),
            default=[list(INDIC_LANGUAGES.keys())[0]]  # Default to first language
        )
        source_lang_codes = [INDIC_LANGUAGES[lang] for lang in source_langs]
        target_lang_code = "eng_Latn"

    if input_method == "Text Input":
        input_label = "Enter English text:" if translation_direction == "English to Indian Language" else f"Enter text in selected language(s):"
        input_text = st.text_area(input_label, height=200)

        if st.button("Translate Text"):
            if input_text:
                with st.spinner("Translating..."):
                    try:
                        if translation_direction == "English to Indian Language":
                            st.markdown("### Translations:")
                            for target_lang, target_code in [(lang, INDIC_LANGUAGES[lang]) for lang in target_langs]:
                                translation = translator.translate(
                                    input_text,
                                    src_lang=source_lang_code,
                                    tgt_lang=target_code
                                )
                                
                                st.markdown(f"**{target_lang}:**")
                                st.write(translation)
                                st.markdown("---")
                        else:
                            st.markdown("### Translations to English:")
                            for source_lang, source_code in [(lang, INDIC_LANGUAGES[lang]) for lang in source_langs]:
                                translation = translator.translate(
                                    input_text,
                                    src_lang=source_code,
                                    tgt_lang=target_lang_code
                                )
                                
                                st.markdown(f"**From {source_lang}:**")
                                st.write(translation)
                                st.markdown("---")
                    except Exception as e:
                        st.error(f"An error occurred during translation: {str(e)}")
            else:
                st.warning("Please enter some text to translate.")

    else:  # File Upload
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                df = read_file_with_encoding(uploaded_file)
                
                st.write("Preview of uploaded file:")
                st.dataframe(df.head())
                
                text_column = st.selectbox(
                    "Select the column to translate:",
                    df.columns
                )
                
                if st.button("Translate File"):
                    if translation_direction == "English to Indian Language" and not target_langs:
                        st.warning("Please select at least one target language.")
                        return
                    
                    if translation_direction == "Indian Language to English" and not source_langs:
                        st.warning("Please select at least one source language.")
                        return
                        
                    with st.spinner("Processing file..."):
                        if translation_direction == "English to Indian Language":
                            result_df = process_file_en_to_indic(
                                df,
                                text_column,
                                translator,
                                target_lang_codes
                            )
                        else:
                            result_df = process_file_indic_to_en(
                                df,
                                text_column,
                                translator,
                                source_lang_codes
                            )
                        
                        st.write("Preview of translated data:")
                        st.dataframe(result_df.head())
                        
                        # Download buttons
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            result_df.to_excel(writer, sheet_name='Translated', index=False)
                        
                        st.download_button(
                            label="Download translated file (Excel)",
                            data=buffer.getvalue(),
                            file_name="translated_data.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                        
                        # Also offer CSV download
                        # csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                        # st.download_button(
                        #     label="Download translated file (CSV)",
                        #     data=csv,
                        #     file_name="translated_data.csv",
                        #     mime="text/csv"
                        # )
                        
            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
    
    # Additional information
    with st.expander("About this translator"):
        st.write("""
        This translator uses the IndicTrans2 model from AI4Bharat to translate between English and Indian languages.
        - Supports both text input and file upload (CSV/Excel)
        - Handles long text translation up to 2048 tokens
        - Supports multiple Indian languages and scripts
        - For file translation, creates new columns with translations
        - Provides download options in Excel and CSV formats
        - Supports translating to/from multiple""")

if __name__ == "__main__":
=======
# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# # from IndicTransToolkit.processor import IndicProcessor

# # import sys
# # import os
# # sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from IndicTransToolkit.processor import IndicProcessor

# class IndicTranslator:
#     def __init__(self, model_name="prajdabre/rotary-indictrans2-en-indic-dist-200M"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # Initialize processor, tokenizer, and model
#         self.processor = IndicProcessor(inference=True)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             trust_remote_code=True
#         ).to(self.device)

#     def translate(self, text, tgt_lang, max_length=2048, num_beams=10):
#         if isinstance(text, str):
#             text = [text]
        
#         # Preprocess the input text
#         batch = self.processor.preprocess_batch(text, src_lang="eng_Latn", tgt_lang=tgt_lang)
        
#         # Tokenize
#         inputs = self.tokenizer(
#             batch,
#             padding="longest",
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt"
#         ).to(self.device)
        
#         # Generate translation
#         with torch.inference_mode():
#             outputs = self.model.generate(
#                 **inputs,
#                 num_beams=num_beams,
#                 length_penalty=1.5,
#                 repetition_penalty=2.0,
#                 num_return_sequences=1,
#                 max_new_tokens=max_length,
#                 early_stopping=True
#             )
        
#         # Decode the outputs
#         translations = self.tokenizer.batch_decode(
#             outputs,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=True
#         )
        
#         # Postprocess the translations
#         translations = self.processor.postprocess_batch(translations, lang=tgt_lang)
        
#         return translations[0] if len(text) == 1 else translations

# # Supported language pairs
# LANGUAGE_PAIRS = {
#     "English to Hindi": "hin_Deva",
#     "English to Bengali": "ben_Beng",
#     "English to Tamil": "tam_Taml",
#     "English to Telugu": "tel_Telu",
#     "English to Malayalam": "mal_Mlym",
#     "English to Kannada": "kan_Knda",
#     "English to Gujarati": "guj_Gujr",
#     "English to Marathi": "mar_Deva",
#     "English to Punjabi": "pan_Guru",
#     "English to Urdu": "urd_Arab",
# }

# def main():
#     st.set_page_config(page_title="Indic Language Translator", page_icon="🌏")
    
#     st.title("🌏 Indic Language Translator")
#     st.markdown("Translate English text to various Indian languages")

#     # Initialize the translator
#     @st.cache_resource
#     def load_translator():
#         return IndicTranslator()

#     translator = load_translator()

#     # Language selection
#     language_pair = st.selectbox(
#         "Select target language:",
#         list(LANGUAGE_PAIRS.keys())
#     )

#     # Text input
#     input_text = st.text_area("Enter English text:", height=200)

#     # Translation button
#     if st.button("Translate"):
#         if input_text:
#             with st.spinner("Translating..."):
#                 try:
#                     tgt_lang = LANGUAGE_PAIRS[language_pair]
#                     translation = translator.translate(
#                         input_text,
#                         tgt_lang=tgt_lang
#                     )
                    
#                     st.success("Translation completed!")
#                     st.markdown("### Translation:")
#                     st.write(translation)
#                 except Exception as e:
#                     st.error(f"An error occurred during translation: {str(e)}")
#         else:
#             st.warning("Please enter some text to translate.")

#     # Additional information
#     with st.expander("About this translator"):
#         st.write("""
#         This translator uses the IndicTrans2 model from AI4Bharat to translate English text into various Indian languages.
#         - Supports long text translation up to 2048 tokens
#         - Uses beam search for better translation quality
#         - Handles multiple Indian languages and scripts
#         """)

# if __name__ == "__main__":
#     main()










#indic to eng and vice versa text

# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from IndicTransToolkit.processor import IndicProcessor

# # Clear Streamlit cache
# st.cache_resource.clear()

# class IndicTranslator:
#     def __init__(self, direction="en_to_indic"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.direction = direction
        
#         # Initialize only the required model based on direction
#         if direction == "en_to_indic":
#             self.model_name = "prajdabre/rotary-indictrans2-en-indic-dist-200M"
#         else:
#             self.model_name = "prajdabre/rotary-indictrans2-indic-en-dist-200M"
        
#         # Initialize processor
#         self.processor = IndicProcessor(inference=True)
        
#         # Initialize tokenizer and model for the specified direction
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(
#             self.model_name,
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             trust_remote_code=True
#         ).to(self.device)

#     def translate(self, text, src_lang, tgt_lang, max_length=2048, num_beams=10):
#         if isinstance(text, str):
#             text = [text]
        
#         # Preprocess the input text
#         batch = self.processor.preprocess_batch(text, src_lang=src_lang, tgt_lang=tgt_lang)
        
#         # Tokenize
#         inputs = self.tokenizer(
#             batch,
#             padding="longest",
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt"
#         ).to(self.device)
        
#         # Generate translation
#         with torch.inference_mode():
#             outputs = self.model.generate(
#                 **inputs,
#                 num_beams=num_beams,
#                 length_penalty=1.5,
#                 repetition_penalty=2.0,
#                 num_return_sequences=1,
#                 max_new_tokens=max_length,
#                 early_stopping=True
#             )
        
#         # Decode the outputs
#         translations = self.tokenizer.batch_decode(
#             outputs,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=True
#         )
        
#         # Postprocess the translations
#         translations = self.processor.postprocess_batch(translations, lang=tgt_lang)
        
#         return translations[0] if len(text) == 1 else translations

# # Supported languages
# INDIC_LANGUAGES = {
#     "Hindi": "hin_Deva",
#     "Bengali": "ben_Beng",
#     "Tamil": "tam_Taml",
#     "Telugu": "tel_Telu",
#     "Malayalam": "mal_Mlym",
#     "Kannada": "kan_Knda",
#     "Gujarati": "guj_Gujr",
#     "Marathi": "mar_Deva",
#     "Punjabi": "pan_Guru",
#     "Urdu": "urd_Arab",
# }

# def main():
#     st.set_page_config(page_title="Indic Language Translator", page_icon="🌏")
    
#     st.title("Indic Language Translator")
#     st.markdown("Translate between English and Indian languages")

#     # Translation direction
#     translation_direction = st.radio(
#         "Select translation direction:",
#         ["English to Indian Language", "Indian Language to English"],
#         key="translation_direction"
#     )

#     # Initialize the appropriate translator based on direction
#     @st.cache_resource
#     def load_translator(direction):
#         return IndicTranslator(direction="en_to_indic" if direction == "English to Indian Language" else "indic_to_en")

#     translator = load_translator(translation_direction)

#     if translation_direction == "English to Indian Language":
#         target_lang = st.selectbox(
#             "Select target language:",
#             list(INDIC_LANGUAGES.keys())
#         )
#         input_label = "Enter English text:"
#         target_lang_code = INDIC_LANGUAGES[target_lang]
#         source_lang_code = "eng_Latn"
#     else:
#         source_lang = st.selectbox(
#             "Select source language:",
#             list(INDIC_LANGUAGES.keys())
#         )
#         input_label = f"Enter text in {source_lang}:"
#         source_lang_code = INDIC_LANGUAGES[source_lang]
#         target_lang_code = "eng_Latn"

#     # Text input
#     input_text = st.text_area(input_label, height=200)

#     # Translation button
#     if st.button("Translate"):
#         if input_text:
#             with st.spinner("Translating..."):
#                 try:
#                     translation = translator.translate(
#                         input_text,
#                         src_lang=source_lang_code,
#                         tgt_lang=target_lang_code
#                     )
                    
#                     st.success("Translation completed!")
#                     st.markdown("### Translation:")
#                     st.write(translation)
#                 except Exception as e:
#                     st.error(f"An error occurred during translation: {str(e)}")
#         else:
#             st.warning("Please enter some text to translate.")

#     # Additional information
#     with st.expander("About this translator"):
#         st.write("""
#         This translator uses the IndicTrans2 model from AI4Bharat to translate between English and Indian languages.
#         - Supports long text translation up to 2048 tokens
#         - Uses beam search for better translation quality
#         - Handles multiple Indian languages and scripts
#         - Optimized to load only the required model based on translation direction
#         """)

# if __name__ == "__main__":
#     main()












# csv columns translation for single lang at one time
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import pandas as pd
import io
import chardet

# Clear Streamlit cache
st.cache_resource.clear()

class IndicTranslator:
    def __init__(self, direction="en_to_indic"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.direction = direction
        
        # Initialize only the required model based on direction
        if direction == "en_to_indic":
            self.model_name = "prajdabre/rotary-indictrans2-en-indic-dist-200M"
        else:
            self.model_name = "prajdabre/rotary-indictrans2-indic-en-dist-200M"
        
        # Initialize processor
        self.processor = IndicProcessor(inference=True)
        
        # Initialize tokenizer and model for the specified direction
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)

    def translate(self, text, src_lang, tgt_lang, max_length=2048, num_beams=10):
        if isinstance(text, str):
            text = [text]
        
        # Preprocess the input text
        batch = self.processor.preprocess_batch(text, src_lang=src_lang, tgt_lang=tgt_lang)
        
        # Tokenize
        inputs = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate translation
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
        
        # Decode the outputs
        translations = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Postprocess the translations
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

def detect_encoding(file_content):
    """Detect the encoding of the file"""
    result = chardet.detect(file_content)
    return result['encoding']

def read_file_with_encoding(uploaded_file):
    """Read file with proper encoding for Indic text"""
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            # For Excel files, use openpyxl engine for better Unicode support
            uploaded_file.seek(0)
            return pd.read_excel(
                uploaded_file,
                engine='openpyxl',
                dtype=str  # Read all columns as string to preserve Indic text
            )
        else:
            # For CSV files
            content = uploaded_file.read()
            encoding = detect_encoding(content)
            encodings_to_try = [encoding, 'utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252']
            
            for enc in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file,
                        encoding=enc,
                        dtype=str  # Read all columns as string
                    )
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    st.error(f"Error reading file with {enc} encoding: {str(e)}")
            
            raise Exception("Unable to read file with any known encoding")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def process_file_en_to_indic(df, text_column, translator, target_lang_codes):
    """Process the dataframe and translate from English to multiple Indic languages"""
    result_df = df.copy()
    source_lang_code = "eng_Latn"
    
    total_languages = len(target_lang_codes)
    progress_placeholder = st.empty()
    
    for i, target_lang_code in enumerate(target_lang_codes):
        target_lang_name = next((k for k, v in INDIC_LANGUAGES.items() if v == target_lang_code), target_lang_code)
        progress_placeholder.text(f"Translating to {target_lang_name} ({i+1}/{total_languages})")
        
        translations = []
        errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows = len(df)
        for idx, text in enumerate(df[text_column]):
            try:
                if pd.isna(text) or str(text).strip() == '':
                    translations.append("")
                    errors.append("")
                    continue
                
                # Clean the input text
                cleaned_text = str(text).strip()
                
                translation = translator.translate(
                    cleaned_text,
                    src_lang=source_lang_code,
                    tgt_lang=target_lang_code
                )
                translations.append(translation)
                errors.append("")
                
            except Exception as e:
                translations.append("")
                errors.append(f"Error: {str(e)}")
            
            # Update progress
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1} of {total_rows} for {target_lang_name}")
        
        # Add translations and errors to the dataframe
        result_df[f'{text_column}_translated_{target_lang_name}'] = translations
        result_df[f'{text_column}_translation_errors_{target_lang_name}'] = errors
        
        # Remove error column if no errors occurred
        if not any(errors):
            result_df = result_df.drop(f'{text_column}_translation_errors_{target_lang_name}', axis=1)
    
    progress_placeholder.text("Translation completed for all languages!")
    return result_df

def process_file_indic_to_en(df, text_column, translator, source_lang_codes):
    """Process the dataframe and translate from multiple Indic languages to English"""
    result_df = df.copy()
    target_lang_code = "eng_Latn"
    
    # Create a new column for each source language
    for source_lang_code in source_lang_codes:
        source_lang_name = next((k for k, v in INDIC_LANGUAGES.items() if v == source_lang_code), source_lang_code)
        
        translations = []
        errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Translating from {source_lang_name} to English")
        
        total_rows = len(df)
        for idx, text in enumerate(df[text_column]):
            try:
                if pd.isna(text) or str(text).strip() == '':
                    translations.append("")
                    errors.append("")
                    continue
                
                # Clean the input text
                cleaned_text = str(text).strip()
                
                translation = translator.translate(
                    cleaned_text,
                    src_lang=source_lang_code,
                    tgt_lang=target_lang_code
                )
                translations.append(translation)
                errors.append("")
                
            except Exception as e:
                translations.append("")
                errors.append(f"Error: {str(e)}")
            
            # Update progress
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1} of {total_rows} from {source_lang_name}")
        
        # Add translations and errors to the dataframe
        result_df[f'{text_column}_translated_from_{source_lang_name}'] = translations
        result_df[f'{text_column}_translation_errors_from_{source_lang_name}'] = errors
        
        # Remove error column if no errors occurred
        if not any(errors):
            result_df = result_df.drop(f'{text_column}_translation_errors_from_{source_lang_name}', axis=1)
    
    return result_df

def main():
    st.set_page_config(page_title="Indic Language Translator", page_icon="🌏", layout="wide")
    
    st.title("Indic Language Translator")
    st.markdown("Translate between English and Indian languages")

    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Text Input", "File Upload"],
        key="input_method"
    )

    # Translation direction
    translation_direction = st.radio(
        "Select translation direction:",
        ["English to Indian Language", "Indian Language to English"],
        key="translation_direction"
    )

    @st.cache_resource
    def load_translator(direction):
        return IndicTranslator(direction="en_to_indic" if direction == "English to Indian Language" else "indic_to_en")

    translator = load_translator(translation_direction)

    if translation_direction == "English to Indian Language":
        # Multiple target languages
        target_langs = st.multiselect(
            "Select target language(s):",
            list(INDIC_LANGUAGES.keys()),
            default=[list(INDIC_LANGUAGES.keys())[0]]  # Default to first language
        )
        target_lang_codes = [INDIC_LANGUAGES[lang] for lang in target_langs]
        source_lang_code = "eng_Latn"
    else:
        # Multiple source languages
        source_langs = st.multiselect(
            "Select source language(s):",
            list(INDIC_LANGUAGES.keys()),
            default=[list(INDIC_LANGUAGES.keys())[0]]  # Default to first language
        )
        source_lang_codes = [INDIC_LANGUAGES[lang] for lang in source_langs]
        target_lang_code = "eng_Latn"

    if input_method == "Text Input":
        input_label = "Enter English text:" if translation_direction == "English to Indian Language" else f"Enter text in selected language(s):"
        input_text = st.text_area(input_label, height=200)

        if st.button("Translate Text"):
            if input_text:
                with st.spinner("Translating..."):
                    try:
                        if translation_direction == "English to Indian Language":
                            st.markdown("### Translations:")
                            for target_lang, target_code in [(lang, INDIC_LANGUAGES[lang]) for lang in target_langs]:
                                translation = translator.translate(
                                    input_text,
                                    src_lang=source_lang_code,
                                    tgt_lang=target_code
                                )
                                
                                st.markdown(f"**{target_lang}:**")
                                st.write(translation)
                                st.markdown("---")
                        else:
                            st.markdown("### Translations to English:")
                            for source_lang, source_code in [(lang, INDIC_LANGUAGES[lang]) for lang in source_langs]:
                                translation = translator.translate(
                                    input_text,
                                    src_lang=source_code,
                                    tgt_lang=target_lang_code
                                )
                                
                                st.markdown(f"**From {source_lang}:**")
                                st.write(translation)
                                st.markdown("---")
                    except Exception as e:
                        st.error(f"An error occurred during translation: {str(e)}")
            else:
                st.warning("Please enter some text to translate.")

    else:  # File Upload
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                df = read_file_with_encoding(uploaded_file)
                
                st.write("Preview of uploaded file:")
                st.dataframe(df.head())
                
                text_column = st.selectbox(
                    "Select the column to translate:",
                    df.columns
                )
                
                if st.button("Translate File"):
                    if translation_direction == "English to Indian Language" and not target_langs:
                        st.warning("Please select at least one target language.")
                        return
                    
                    if translation_direction == "Indian Language to English" and not source_langs:
                        st.warning("Please select at least one source language.")
                        return
                        
                    with st.spinner("Processing file..."):
                        if translation_direction == "English to Indian Language":
                            result_df = process_file_en_to_indic(
                                df,
                                text_column,
                                translator,
                                target_lang_codes
                            )
                        else:
                            result_df = process_file_indic_to_en(
                                df,
                                text_column,
                                translator,
                                source_lang_codes
                            )
                        
                        st.write("Preview of translated data:")
                        st.dataframe(result_df.head())
                        
                        # Download buttons
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            result_df.to_excel(writer, sheet_name='Translated', index=False)
                        
                        st.download_button(
                            label="Download translated file (Excel)",
                            data=buffer.getvalue(),
                            file_name="translated_data.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                        
                        # Also offer CSV download
                        # csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                        # st.download_button(
                        #     label="Download translated file (CSV)",
                        #     data=csv,
                        #     file_name="translated_data.csv",
                        #     mime="text/csv"
                        # )
                        
            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
    
    # Additional information
    with st.expander("About this translator"):
        st.write("""
        This translator uses the IndicTrans2 model from AI4Bharat to translate between English and Indian languages.
        - Supports both text input and file upload (CSV/Excel)
        - Handles long text translation up to 2048 tokens
        - Supports multiple Indian languages and scripts
        - For file translation, creates new columns with translations
        - Provides download options in Excel and CSV formats
        - Supports translating to/from multiple""")

if __name__ == "__main__":
>>>>>>> 61961dcee16732a8e6699d9f6350301d8ed6f5be
    main()