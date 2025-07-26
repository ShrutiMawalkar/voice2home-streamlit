import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Voice2Home AI", page_icon="🏡")

st.title("🏡 Voice2Home AI (Text Version)")
st.markdown("Type your **dream home layout idea**, and get an **AI-generated layout description** using GPT-Neo!")

# User text input instead of voice
prompt = st.text_area("📝 Enter your home design idea (English or Hindi):", placeholder="e.g., I want a 3BHK with open kitchen, garden, and balcony...")

if st.button("Generate Layout") and prompt.strip() != "":
    st.subheader("🏗️ Generating Home Layout...")
    st.text("Loading GPT-Neo model...")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    llm_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

    inputs = tokenizer(prompt, return_tensors="pt")
    output = llm_model.generate(**inputs, max_new_tokens=150, temperature=0.9, do_sample=True)

    layout_description = tokenizer.decode(output[0], skip_special_tokens=True)

    st.success("🏠 AI-Generated Layout Description:")
    st.write(layout_description)
else:
    st.info("Enter your idea and click the button to generate layout.")
