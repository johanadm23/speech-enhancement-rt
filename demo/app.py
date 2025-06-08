import streamlit as st
import librosa
import soundfile as sf
from models.unet import UNet
import torch

st.title("Real-Time Speech Enhancer")
uploaded = st.file_uploader("Upload a noisy WAV file", type=["wav"])
model = UNet()
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location="cpu"))
model.eval()

if uploaded:
    y, sr = librosa.load(uploaded, sr=16000)
    y_tensor = torch.tensor(y).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        enhanced = model(y_tensor).squeeze().numpy()
    sf.write("enhanced.wav", enhanced, sr)
    st.audio("enhanced.wav", format="audio/wav")
