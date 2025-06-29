<<<<<<< HEAD
import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import ssl
import gc  # for garbage collection

# SSL and certificate settings
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['REQUESTS_CA_BUNDLE'] = r"C:\Users\s1297621\OneDrive - Syngenta\Desktop\DDS\STT_model\.venv\Lib\site-packages\certifi\cacert.pem"

# Cache the model and tokenizer loading
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "sarvamai/sarvam-translate"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    return model, tokenizer

# Supported languages
LANGUAGE_OPTIONS = ["Hindi", "Bengali", "Tamil", "Telugu", "Malayalam", "Kannada", "Gujarati", "Marathi", "Punjabi", "Urdu"]

def translate_text(input_txt, tgt_lang, model, tokenizer):
    try:
        messages = [
            {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
            {"role": "user", "content": input_txt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        with torch.no_grad():  # Disable gradient calculation
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.01,
                num_return_sequences=1
            )
            
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_text
    
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Multi-Language Translator",
        page_icon="🌏",
        layout="wide"  # Use wide layout for better space utilization
    )
    
    st.title("🌏 Multi-Language Translator")
    st.markdown("Translate English text to various languages")

    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        input_text = st.text_area("Enter English text:", height=200)
        tgt_lang = st.selectbox("Select target language:", LANGUAGE_OPTIONS)
        translate_button = st.button("Translate")

    with col2:
        if translate_button:
            if input_text:
                with st.spinner("Translating..."):
                    try:
                        translation = translate_text(input_text, tgt_lang, model, tokenizer)
                        if translation:
                            st.success("Translation completed!")
                            st.markdown("### Translation:")
                            st.write(translation)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter some text to translate.")

    # Footer
    st.markdown("---")
    with st.expander("About this translator"):
        st.write("""
        This translator uses the Sarvam Translate model to translate English text into various languages.
        - Supports multiple languages
        - Utilizes a chat-style prompt for translation tasks
        """)

    # Garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
=======
import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import ssl
import gc  # for garbage collection

# SSL and certificate settings
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['REQUESTS_CA_BUNDLE'] = r"C:\Users\s1297621\OneDrive - Syngenta\Desktop\DDS\STT_model\.venv\Lib\site-packages\certifi\cacert.pem"

# Cache the model and tokenizer loading
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "sarvamai/sarvam-translate"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    return model, tokenizer

# Supported languages
LANGUAGE_OPTIONS = ["Hindi", "Bengali", "Tamil", "Telugu", "Malayalam", "Kannada", "Gujarati", "Marathi", "Punjabi", "Urdu"]

def translate_text(input_txt, tgt_lang, model, tokenizer):
    try:
        messages = [
            {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
            {"role": "user", "content": input_txt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        with torch.no_grad():  # Disable gradient calculation
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.01,
                num_return_sequences=1
            )
            
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_text
    
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Multi-Language Translator",
        page_icon="🌏",
        layout="wide"  # Use wide layout for better space utilization
    )
    
    st.title("🌏 Multi-Language Translator")
    st.markdown("Translate English text to various languages")

    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        input_text = st.text_area("Enter English text:", height=200)
        tgt_lang = st.selectbox("Select target language:", LANGUAGE_OPTIONS)
        translate_button = st.button("Translate")

    with col2:
        if translate_button:
            if input_text:
                with st.spinner("Translating..."):
                    try:
                        translation = translate_text(input_text, tgt_lang, model, tokenizer)
                        if translation:
                            st.success("Translation completed!")
                            st.markdown("### Translation:")
                            st.write(translation)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter some text to translate.")

    # Footer
    st.markdown("---")
    with st.expander("About this translator"):
        st.write("""
        This translator uses the Sarvam Translate model to translate English text into various languages.
        - Supports multiple languages
        - Utilizes a chat-style prompt for translation tasks
        """)

    # Garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
>>>>>>> 61961dcee16732a8e6699d9f6350301d8ed6f5be
    main()