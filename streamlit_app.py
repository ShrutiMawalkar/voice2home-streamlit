import streamlit as st
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Voice2Home AI", page_icon="🏡")

st.title("🎙️ Voice2Home AI")
st.markdown("Speak your **dream home layout idea**, and get an **AI-generated layout description** using Whisper + GPT-Neo!")

# Upload Audio File
audio_file = st.file_uploader("Upload your voice file (.mp3 or .wav)", type=["mp3", "wav"])

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    st.audio("temp_audio.wav")

    # Whisper Transcription
    st.subheader("📝 Step 1: Transcribe your voice input")
    st.text("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("temp_audio.wav")
    prompt = result["text"]
    st.success(f"Transcribed Prompt: {prompt}")

    # LLM Generation
    st.subheader("🏗️ Step 2: Generating Home Layout with GPT-Neo")
    st.text("Loading GPT-Neo model...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    llm_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

    inputs = tokenizer(prompt, return_tensors="pt")
    output = llm_model.generate(**inputs, max_new_tokens=100)
    layout_description = tokenizer.decode(output[0], skip_special_tokens=True)

    st.success("🏠 Layout Description:")
    st.write(layout_description)
else:
    st.warning("Please upload an audio file first.")

