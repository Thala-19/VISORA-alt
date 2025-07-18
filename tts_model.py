from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from io import BytesIO
import streamlit as st
import torch
import soundfile as sf
import tempfile

# Cache everything when imported via Streamlit
@st.cache_resource
def load_tts_models():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    return processor, model, vocoder, speaker_embeddings

def synthesize_speech():
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

    processor, model, vocoder, speaker_embeddings = load_tts_models()
    inputs = processor(text=input_text_from_file, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    buffer = BytesIO()
    sf.write(buffer, speech.numpy(), samplerate=16000, format='WAV')
    buffer.seek(0)
    return buffer