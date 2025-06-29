<<<<<<< HEAD
import streamlit as st
import whisper
import os
import tempfile
import numpy as np
import librosa
import torch
torch.classes.__path__ = []

# Load the Whisper model
model = whisper.load_model("large-v1")

def transcribe_audio(audio_file):
    # Create a temporary file with explicit path
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_audio" + os.path.splitext(audio_file.name)[1])
    
    # Write the uploaded file content to the temporary file
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    try:
        # Load audio with librosa (better MP3 support)
        audio_data, sample_rate = librosa.load(temp_path, sr=16000, mono=True)
        
        # Ensure correct data type (float32)
        audio_data = audio_data.astype(np.float32)
        
        # Pad or trim audio to make it suitable for model input
        audio = whisper.pad_or_trim(audio_data)
        
        # Calculate log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Detect language
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        
        # Use the audio data for transcription
        result = model.transcribe(audio_data, language="en")
        
        return result["text"], detected_language
    except Exception as e:
        st.error(f"Detailed error: {str(e)}")
        return f"Transcription failed: {str(e)}"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    st.title("Voice-to-Text Chatbot")
    st.write("Upload a voice note to receive a text response.")

    # Install required packages notice
    st.info("This app requires whisper and librosa packages. Install with: pip install openai-whisper librosa")

    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_file)
            st.write("Transcription:")
            st.write(transcription)

if __name__ == "__main__":
=======
import streamlit as st
import whisper
import os
import tempfile
import numpy as np
import librosa
import torch
torch.classes.__path__ = []

# Load the Whisper model
model = whisper.load_model("large-v1")

def transcribe_audio(audio_file):
    # Create a temporary file with explicit path
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_audio" + os.path.splitext(audio_file.name)[1])
    
    # Write the uploaded file content to the temporary file
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    try:
        # Load audio with librosa (better MP3 support)
        audio_data, sample_rate = librosa.load(temp_path, sr=16000, mono=True)
        
        # Ensure correct data type (float32)
        audio_data = audio_data.astype(np.float32)
        
        # Pad or trim audio to make it suitable for model input
        audio = whisper.pad_or_trim(audio_data)
        
        # Calculate log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Detect language
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        
        # Use the audio data for transcription
        result = model.transcribe(audio_data, language="en")
        
        return result["text"], detected_language
    except Exception as e:
        st.error(f"Detailed error: {str(e)}")
        return f"Transcription failed: {str(e)}"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    st.title("Voice-to-Text Chatbot")
    st.write("Upload a voice note to receive a text response.")

    # Install required packages notice
    st.info("This app requires whisper and librosa packages. Install with: pip install openai-whisper librosa")

    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_file)
            st.write("Transcription:")
            st.write(transcription)

if __name__ == "__main__":
>>>>>>> 61961dcee16732a8e6699d9f6350301d8ed6f5be
    main()