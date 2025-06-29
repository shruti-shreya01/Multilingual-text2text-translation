<<<<<<< HEAD
import streamlit as st
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import time
from io import BytesIO

class IndicTTS:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        
        self.default_description = """
        A female speaker with an Indian accent delivers a natural, conversational speech 
        with clear articulation and warmth in her voice. The speech has moderate speed and 
        natural pitch variations. The audio quality is pristine with no background noise.
        """

    def generate_speech(self, text, custom_description=None):
        description = custom_description if custom_description else self.default_description
        
        description_inputs = self.description_tokenizer(
            description, 
            return_tensors="pt"
        ).to(self.device)
        
        text_inputs = self.tokenizer(
            text, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            generation = self.model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=text_inputs.input_ids,
                prompt_attention_mask=text_inputs.attention_mask,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
        
        audio_array = generation.cpu().numpy().squeeze()
        return audio_array, self.model.config.sampling_rate

def main():
    st.set_page_config(
        page_title="Text-to-Speech",
        page_icon="🗣️",
        layout="wide"
    )

    st.title("Text-to-Speech Converter")
    st.markdown("---")

    # Initialize TTS engine
    @st.cache_resource
    def load_tts_model():
        return IndicTTS()

    tts = load_tts_model()

    # Voice style options
    VOICE_STYLES = {
        "Natural Conversational": """A female speaker with an Indian accent delivers a natural, 
        conversational speech with clear articulation and warmth in her voice with no background noise.""",
        
        "Professional": """A female speaker with an Indian accent delivers a professional and formal speech 
        with precise articulation and measured pace with no background noise.""",
        
        "Friendly Casual": """A female speaker with an Indian accent speaks in a friendly and casual manner, 
        with natural flow and engaging tone with no background noise."""
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        # Text input
        input_text = st.text_area(
            "Enter Text",
            height=150,
            placeholder="Type or paste your text here..."
        )

    with col2:
        # Voice style selection
        voice_style = st.selectbox(
            "Select Voice Style",
            list(VOICE_STYLES.keys())
        )

        # Speed control
        speed_factor = st.slider(
            "Speech Speed",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.1
        )

    # Generate button
    if st.button("Generate Speech", type="primary"):
        if not input_text:
            st.error("Please enter some text!")
            return

        try:
            with st.spinner("Generating audio... Please wait"):
                # Generate audio
                audio_array, sample_rate = tts.generate_speech(
                    text=input_text,
                    custom_description=VOICE_STYLES[voice_style]
                )

                # Apply speed adjustment
                if speed_factor != 1.0:
                    import librosa
                    audio_array = librosa.effects.time_stretch(audio_array, rate=speed_factor)

                # Save to BytesIO
                audio_buffer = BytesIO()
                sf.write(audio_buffer, audio_array, sample_rate, format='WAV')
                audio_buffer.seek(0)

                # Display audio player and download button
                st.audio(audio_buffer, format='audio/wav')
                
                # Create download button
                st.download_button(
                    label="Download Audio",
                    data=audio_buffer,
                    file_name=f"tts_output_{int(time.time())}.wav",
                    mime="audio/wav"
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Add information section
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        This application converts text to speech with an Indian accent.
        
        **Features:**
        - Different voice styles
        - Adjustable speech speed
        - High-quality audio output
        - Download option for generated audio
        
        **Note:** The quality of the output may vary depending on the length of the input text.
        """)

if __name__ == "__main__":
=======
import streamlit as st
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import time
from io import BytesIO

class IndicTTS:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        
        self.default_description = """
        A female speaker with an Indian accent delivers a natural, conversational speech 
        with clear articulation and warmth in her voice. The speech has moderate speed and 
        natural pitch variations. The audio quality is pristine with no background noise.
        """

    def generate_speech(self, text, custom_description=None):
        description = custom_description if custom_description else self.default_description
        
        description_inputs = self.description_tokenizer(
            description, 
            return_tensors="pt"
        ).to(self.device)
        
        text_inputs = self.tokenizer(
            text, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            generation = self.model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=text_inputs.input_ids,
                prompt_attention_mask=text_inputs.attention_mask,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
        
        audio_array = generation.cpu().numpy().squeeze()
        return audio_array, self.model.config.sampling_rate

def main():
    st.set_page_config(
        page_title="Text-to-Speech",
        page_icon="🗣️",
        layout="wide"
    )

    st.title("Text-to-Speech Converter")
    st.markdown("---")

    # Initialize TTS engine
    @st.cache_resource
    def load_tts_model():
        return IndicTTS()

    tts = load_tts_model()

    # Voice style options
    VOICE_STYLES = {
        "Natural Conversational": """A female speaker with an Indian accent delivers a natural, 
        conversational speech with clear articulation and warmth in her voice with no background noise.""",
        
        "Professional": """A female speaker with an Indian accent delivers a professional and formal speech 
        with precise articulation and measured pace with no background noise.""",
        
        "Friendly Casual": """A female speaker with an Indian accent speaks in a friendly and casual manner, 
        with natural flow and engaging tone with no background noise."""
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        # Text input
        input_text = st.text_area(
            "Enter Text",
            height=150,
            placeholder="Type or paste your text here..."
        )

    with col2:
        # Voice style selection
        voice_style = st.selectbox(
            "Select Voice Style",
            list(VOICE_STYLES.keys())
        )

        # Speed control
        speed_factor = st.slider(
            "Speech Speed",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.1
        )

    # Generate button
    if st.button("Generate Speech", type="primary"):
        if not input_text:
            st.error("Please enter some text!")
            return

        try:
            with st.spinner("Generating audio... Please wait"):
                # Generate audio
                audio_array, sample_rate = tts.generate_speech(
                    text=input_text,
                    custom_description=VOICE_STYLES[voice_style]
                )

                # Apply speed adjustment
                if speed_factor != 1.0:
                    import librosa
                    audio_array = librosa.effects.time_stretch(audio_array, rate=speed_factor)

                # Save to BytesIO
                audio_buffer = BytesIO()
                sf.write(audio_buffer, audio_array, sample_rate, format='WAV')
                audio_buffer.seek(0)

                # Display audio player and download button
                st.audio(audio_buffer, format='audio/wav')
                
                # Create download button
                st.download_button(
                    label="Download Audio",
                    data=audio_buffer,
                    file_name=f"tts_output_{int(time.time())}.wav",
                    mime="audio/wav"
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Add information section
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        This application converts text to speech with an Indian accent.
        
        **Features:**
        - Different voice styles
        - Adjustable speech speed
        - High-quality audio output
        - Download option for generated audio
        
        **Note:** The quality of the output may vary depending on the length of the input text.
        """)

if __name__ == "__main__":
>>>>>>> 61961dcee16732a8e6699d9f6350301d8ed6f5be
    main()