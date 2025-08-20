
# AI-Powered Language Tools

This repository contains three AI-powered projects that focus on multilingual text translation, speech-to-text conversion, and text-to-speech synthesis. These projects are built using state-of-the-art transformer models and speech processing frameworks.

---

## 1. Multilingual Text Translator 🌐

A web application for translating text between English and various Indic languages, and vice versa.
<img width="1698" height="888" alt="Screenshot 2025-08-20 182443" src="https://github.com/user-attachments/assets/95fc01d1-ca97-4b8d-9ccd-c3e92470603c" />

**Features:**
- Translate text from English to Indic languages and from Indic languages to English.
- Support for multiple target languages in one go.
- File translation for TXT and XLSX formats.
- Download translations individually or combined.
- Beautiful, responsive UI with glassmorphism effects and gradient styling.

**Supported Indic Languages:**
- Hindi, Bengali, Tamil, Telugu, Malayalam, Kannada, Gujarati, Marathi, Punjabi, Urdu

**Technologies:**
- Python, Streamlit, PyTorch
- Transformers (`prajdabre/rotary-indictrans2`)
- IndicTransToolkit for preprocessing and postprocessing
- Pandas, Chardet for file handling

**Usage:**
```bash
pip install -r requirements.txt
```

---

## 2. Speech-to-Text (Voice Transcription) 🎤

An app to convert audio files into text using OpenAI Whisper.

**Features:**
- Upload audio files (wav, mp3, m4a) for transcription.
- Handles long audio files by chunked processing.
- Detects the language of the uploaded audio.
- Provides accurate, fast transcription results.

**Supported Indic Languages:**
- Hindi, Bengali, Tamil, Telugu, Malayalam, Kannada, Gujarati, Marathi, Punjabi, Urdu

**Technologies:**
- Python, Streamlit, Whisper
- Librosa for audio loading and processing
- Numpy for numerical operations
- PyTorch for model inference

**Usage:**
```bash
pip install -r requirements.txt
streamlit run faster_whisper_app.py
```
## 3. Text-to-Speech (Voice Generation) 🗣️

An app to convert text into natural-sounding speech with customizable voice styles.

**Features:**
- Multiple Indian-accent voice styles: Natural Conversational, Professional, Friendly Casual.
- Adjustable speech speed.
- Download generated audio in WAV format.
- High-quality audio generation using ParlerTTS.

**Supported Indic Languages:**
- Hindi, Bengali, Tamil, Telugu, Malayalam, Kannada, Gujarati, Marathi, Punjabi, Urdu

**Technologies:**
- Python, Streamlit, PyTorch
- ParlerTTS (ai4bharat/indic-parler-tts)
- Transformers, SoundFile
- Optional Librosa for speed adjustment

**Usage:**
```bash
pip install -r requirements.txt
streamlit run text2speech.py
streamlit run text2text.py
```
---
## Setup Instructions

### 1. Clone this repository:
```bash
git clone https://github.com/shruti-shreya01/Multilingual-text2text-translation.git
cd Multilingual-text2text-translation
```
### 2. Installation:
```bash
pip install -r requirements.txt
```
### 3. Usage:
```bash
streamlit run <script_name.py>
# Example:
# streamlit run text2text.py
# streamlit run speech_to_text.py
# streamlit run text2speech.py

