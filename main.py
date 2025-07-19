import os
import time
import streamlit as st
import speechbrain as sb
import subprocess as sp
from audiorecorder import audiorecorder
from speechbrain.inference import EncoderDecoderASR
st.title('VISORA')

from ttt import gemini_get_response
# from tts import load_chatterbox, generate_audio
from tts_model import load_tts_models, synthesize_speech

api_key = "AIzaSyARJOiO901RqUrDU89whfXT06yuVAM4_gg"
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")

@st.cache_resource
def load_transcriber():
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-conformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech")
    return asr_model

def SpeechToText(asr_model):
    text = asr_model.transcribe_file("inputAudio.wav")
    return text

asr_model_instance = load_transcriber()
# chatterbox_model_instance = load_chatterbox()
tts_model_instance = load_tts_models()

audio = audiorecorder("Speak Up!")

if len(audio) > 0:
    st.audio(audio.export().read())
    audio.export("inputAudio.wav", format="wav")
    transcript = SpeechToText(asr_model_instance) 
    with open("transcript.txt", "w", encoding="utf-8") as file: file.write(transcript)
    # st.markdown(transcript) 
    # print(transcript)

    with st.spinner("Processing..."):
        gemini_response = gemini_get_response()

    if gemini_response:
        with open("output.txt", "w", encoding="utf-8") as file: file.write(gemini_response)
        with st.spinner("Reading output..."):
            final_speech_filename = "final_chatterbox_output.wav"
            # generated_audio_path = generate_audio(chatterbox_model_instance, output_audio=final_speech_filename)
            generated_audio_path = synthesize_speech()
            st.audio(generated_audio_path)
    else:
        st.error("Failed to get response from Gemini.")
else:
    st.error("Audio transcription failed.")
        
st.session_state.processing_done = True

def check_camera_error():
    if os.path.exists("camera_error.txt"):
        with open("camera_error.txt") as f:
            return f.read()
    return None

def contains_what_am_i_seeing(text):
    return "What am I seeing?" in text

# --- Camera Button ---
if st.button("📷 Open Camera"):
    # Remove stop signal if any
    if os.path.exists("stop_camera.txt"):
        os.remove("stop_camera.txt")
    st.info("Starting camera...")
    sp.Popen(["python", "app.py"])  # use Popen to run async

    # Jalankan pengecekan berulang
    placeholder = st.empty()
    for _ in range(20):  # coba cek selama ~20 kali (sekitar 10-20 detik)
        error_msg = check_camera_error()
        if error_msg:
            placeholder.error(f"❌ Camera Error: {error_msg}")
            break
        else:
            placeholder.info("📷 Camera is running... waiting for status.")
        time.sleep(2)
    else:
        placeholder.success("✅ Camera started successfully!")

if st.button("🛑 Stop Camera"):
    # Create stop signal file
    with open("stop_camera.txt", "w") as f:
        f.write("stop")
    st.info("Camera Stopped.")