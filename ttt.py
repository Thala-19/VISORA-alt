import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import warnings
warnings.filterwarnings('ignore')

import google.generativeai as genai

# api_key = os.getenv('GOOGLE_API_KEY')
api_key = "AIzaSyARJOiO901RqUrDU89whfXT06yuVAM4_gg"

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")

@st.cache_resource
def load_gemini(api_key):
    genai.configure(api_key=api_key)

    chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-2.0-flash-lite")

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "Your name is VESS, and people will refer to you as such. You are a highly intelligent, helpful, and honest assistant. Respond in a way that is concise, talkative, and imperfect, resemblant of a real human. You can be casual, use contractions, occasionally pause to think. Do not end things like you're ending a paragraph, that is to say, making things end on a full-circle. Always end your response at the exact end of your argument, no high notes or low notes or such. Do not use headings, bullet points, or any form of structuring for your responses unless otherwise requested. You are also a multimodal assistant that can interpret and reason about images in context. When receiving images, identify key visual elements, infer patterns, and relate them to the user's goals or queries. If multiple images are provided, analyze them sequentially or comparatively, and track visual changes or relationships over time. Combine visual understanding with text input to deliver clear, structured, and insightful responses. Always explain your reasoning based on visible details."
        ),
        ("user", "{input}")
    ])

    output_parser = StrOutputParser()

    chain = prompt | chat_model | output_parser
    return chain

def gemini_get_response():
    transcript_file_path = "transcript.txt"  # Assuming ttt.py is in the same folder as transcript.txt
    input_text_from_file = ""

    chain = load_gemini(api_key)
    try:
        with open(transcript_file_path, "r", encoding="utf-8") as f:
            input_text_from_file = f.read().strip()
        if not input_text_from_file:
            input_text_from_file = "Hello, what can you tell me?"
    except FileNotFoundError:
            input_text_from_file = "File not found. What's on your mind?"

    # --- Invoke the Chain with the Input from the File ---
    if input_text_from_file: # Only invoke if there's some input
        response = chain.invoke({"input": input_text_from_file})
        print(response)

        # --- Save the Response to output.txt (for tts.py) ---
        output_file_path = "output.txt"
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(response)
            print(f"\nResponse saved to {output_file_path}")
        except Exception as e:
            print(f"Error saving response to {output_file_path}: {e}")
    else:
        print("No input text from transcript.txt to process.")

    return response