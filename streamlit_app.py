import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.set_page_config(page_title="🏡 Voice2Home AI", page_icon="🏡")

st.title("🏡 Voice2Home AI – Generate 2D Home Layout")
st.markdown("Describe your dream home, and get a 2D layout generated using AI!")

prompt = st.text_area("Enter your home design idea (English only):", placeholder="e.g. A 2BHK flat with an open kitchen, 2 bathrooms, and a balcony...")

if st.button("Generate 2D Layout") and prompt.strip():
    st.subheader("🎨 Generating layout...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

    image = pipe(prompt).images[0]
    st.image(image, caption="🖼️ AI-Generated 2D Layout")
else:
    st.info("Enter a design prompt and click to generate layout.")
