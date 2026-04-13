import streamlit as st
import torch
import torchaudio
import tempfile

# Load model
@st.cache_resource
def load_model():
    model = torch.load("../models/best_model.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

st.title("🎧 Speech Enhancement Demo")

uploaded_file = st.file_uploader("Upload a noisy audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    waveform, sr = torchaudio.load(tmp_path)

    st.audio(tmp_path, format="audio/wav")
    st.write("Original Audio")

    # Preprocess (adjust to your pipeline!)
    input_tensor = waveform.unsqueeze(0)

    with torch.no_grad():
        enhanced = model(input_tensor)

    # Save enhanced audio
    output_path = tmp_path + "_enhanced.wav"
    torchaudio.save(output_path, enhanced.squeeze(0), sr)

    st.audio(output_path, format="audio/wav")
    st.write("Enhanced Audio")
