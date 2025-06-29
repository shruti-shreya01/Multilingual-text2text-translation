# import streamlit as st
# import whisper
# import os
# import tempfile
# import numpy as np
# import librosa
# import torch
# torch.classes.__path__ = []

# # Load the Whisper model
# model = whisper.load_model("medium")

# def transcribe_audio(audio_file, chunk_size=30):
#     # Create a temporary file with explicit path
#     temp_dir = tempfile.gettempdir()
#     temp_path = os.path.join(temp_dir, "temp_audio" + os.path.splitext(audio_file.name)[1])
    
#     # Write the uploaded file content to the temporary file
#     with open(temp_path, "wb") as f:
#         f.write(audio_file.getbuffer())
    
#     try:
#         # Load audio with librosa
#         audio_data, sample_rate = librosa.load(temp_path, sr=16000, mono=True)
        
#         # Ensure correct data type
#         audio_data = audio_data.astype(np.float32)
        
#         # Calculate the number of samples per chunk
#         chunk_length_samples = int(chunk_size * sample_rate)
        
#         # Process audio in chunks
#         transcription_result = []
#         for start in range(0, len(audio_data), chunk_length_samples):
#             end = start + chunk_length_samples
#             audio_chunk = whisper.pad_or_trim(audio_data[start:end])
            
#             # Calculate log-Mel spectrogram
#             mel = whisper.log_mel_spectrogram(audio_chunk).to(model.device)
            
#             # Use the audio chunk for transcription
#             result = model.transcribe(audio_chunk, language="en")
#             transcription_result.append(result["text"])
        
#         # Combine transcriptions
#         full_transcription = " ".join(transcription_result)
        
#         # Detect language (optional, based on first chunk)
#         _, probs = model.detect_language(mel)
#         detected_language = max(probs, key=probs.get)
        
#         return full_transcription, detected_language
#     except Exception as e:
#         st.error(f"Detailed error: {str(e)}")
#         return f"Transcription failed: {str(e)}"
#     finally:
#         # Clean up the temporary file
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

# def main():
#     st.title("Voice-to-Text Chatbot")
#     st.write("Upload a voice note to receive a text response.")

#     # Install required packages notice
#     # st.info("This app requires whisper and librosa packages. Install with: pip install openai-whisper librosa")

#     audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

#     if audio_file is not None:
#         with st.spinner("Transcribing..."):
#             transcription, detected_language = transcribe_audio(audio_file)
#             st.write("Transcription:")
#             st.write(transcription)
#             st.write(f"Detected language: {detected_language}")

# if __name__ == "__main__":
#     main()
import streamlit as st
import whisper
import os
import tempfile
import numpy as np
import librosa
import torch
torch.classes.__path__ = []

# Load the Whisper model
model = whisper.load_model("medium")

def transcribe_audio(audio_file, chunk_size=30):
    # Create a temporary file with explicit path
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_audio" + os.path.splitext(audio_file.name)[1])
    
    # Write the uploaded file content to the temporary file
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    try:
        # Load audio with librosa
        audio_data, sample_rate = librosa.load(temp_path, sr=16000, mono=True)
        
        # Ensure correct data type
        audio_data = audio_data.astype(np.float32)
        
        # Calculate the number of samples per chunk
        chunk_length_samples = int(chunk_size * sample_rate)
        
        # Process audio in chunks
        transcription_result = []
        for start in range(0, len(audio_data), chunk_length_samples):
            end = start + chunk_length_samples
            audio_chunk = whisper.pad_or_trim(audio_data[start:end])
            
            # Calculate log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_chunk).to(model.device)
            
            # Use the audio chunk for transcription
            result = model.transcribe(audio_chunk, language="en")
            transcription_result.append(result["text"])
        
        # Combine transcriptions
        full_transcription = " ".join(transcription_result)
        
        # Detect language (optional, based on first chunk)
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        
        return full_transcription, detected_language
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
    # st.info("This app requires whisper and librosa packages. Install with: pip install openai-whisper librosa")

    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        with st.spinner("Transcribing..."):
            transcription, detected_language = transcribe_audio(audio_file)
            st.write("Transcription:")
            st.write(transcription)
            st.write(f"Detected language: {detected_language}")

if __name__ == "__main__":
    main()