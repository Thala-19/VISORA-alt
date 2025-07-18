import streamlit as st
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

@st.cache_resource
def load_chatterbox():
    model = ChatterboxTTS.from_pretrained(device="cuda")
    return model

def generate_audio(model, output_audio: str="audio_output.wav"):
    transcript_file_path = "output.txt" 
    input_text_from_file = ""

    try:
        with open(transcript_file_path, "r", encoding="utf-8") as f:
            input_text_from_file = f.read().strip()
        if not input_text_from_file:
            # print(f"Info: The file {transcript_file_path} was empty. The model might not have specific input.")
            input_text_from_file = "Hello, what can you tell me?"
    except FileNotFoundError:
        # print(f"Error: The file {transcript_file_path} was not found. The model will not have user input.")
        input_text_from_file = "File not found. What's on your mind?"

    if input_text_from_file: 
        wav = model.generate(input_text_from_file)
        ta.save(output_audio, wav, model.sr)
        return output_audio
    else:
        print("No input text from output.txt to process.")
        return None