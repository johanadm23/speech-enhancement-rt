# ── Suppress known Anaconda + PyTorch environment warnings ───────────────────
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # fixes OMP libiomp5md.dll conflict

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # paramiko/cryptography

import streamlit as st
import torch
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import numpy as np
import io
import sys

# ── Import your own model ────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import UNet

# ── Config — must match training exactly (NB6) ───────────────────────────────
MODEL_PATH = "C:/Users/johana/Documents/GitHub/speech-enhancement-rt/models/best_model.pth"
STATS_PATH = "C:/Users/johana/Documents/GitHub/speech-enhancement-rt/models/norm_stats_logmel.pkl"

SR         = 16_000
N_FFT      = 512
HOP_LENGTH = 128
N_MELS     = 64

# ── Load model + norm stats (cached) ─────────────────────────────────────────
@st.cache_resource
def load_model():
    import pickle

    model = UNet()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "rb") as f:
            stats = pickle.load(f)
    else:
        stats = {"noisy_mean": 0.0, "noisy_std": 1.0,
                 "clean_mean": 0.0, "clean_std": 1.0}
    return model, stats

# ── Audio → log-mel (matches NB6 exactly) ────────────────────────────────────
def waveform_to_mel(wav_np, stats):
    """wav_np: 1-D float32 array  →  (1,1,N_MELS,T) tensor + raw dB array"""
    mel      = librosa.feature.melspectrogram(
                   y=wav_np, sr=SR, n_fft=N_FFT,
                   hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - stats["noisy_mean"]) / (stats["noisy_std"] + 1e-8)
    tensor   = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, mel_db

# ── Enhanced mel → audio via phase-preserving mask ───────────────────────────
def mel_to_audio(enhanced_tensor, noisy_wav_np, stats):
    """
    Avoids Griffin-Lim entirely. Instead:
      1. Compute the noisy STFT (keeps original phase)
      2. Convert both noisy mel and enhanced mel to approximate linear magnitude
      3. Build a Wiener-style ratio mask from enhanced / noisy
      4. Apply mask to original STFT magnitude, keep original phase
      5. ISTFT back to waveform  →  natural-sounding, no robotics

    enhanced_tensor : (1, N_MELS, T) or (N_MELS, T) model output (normalised)
    noisy_wav_np    : original 1-D float32 numpy waveform
    stats           : normalisation dict
    """
    if enhanced_tensor.dim() == 3:
        enhanced_tensor = enhanced_tensor.squeeze(0)

    # ── Denormalise enhanced mel back to dB ──────────────────────────────────
    enh_np = enhanced_tensor.detach().cpu().numpy()           # (N_MELS, T_mel)
    enh_db = enh_np * (stats["clean_std"] + 1e-8) + stats["clean_mean"]

    # ── Also get noisy mel in dB (same scale) ────────────────────────────────
    noisy_mel     = librosa.feature.melspectrogram(
                        y=noisy_wav_np, sr=SR,
                        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    noisy_db      = librosa.power_to_db(noisy_mel, ref=np.max)

    # ── Align time axes (model may pad/crop slightly) ────────────────────────
    T = min(enh_db.shape[1], noisy_db.shape[1])
    enh_db   = enh_db[:, :T]
    noisy_db = noisy_db[:, :T]

    # ── Build ratio mask in linear (power) domain ────────────────────────────
    enh_pow   = librosa.db_to_power(enh_db)    # (N_MELS, T)
    noisy_pow = librosa.db_to_power(noisy_db)  # (N_MELS, T)
    # Wiener-style mask: how much cleaner is enhanced vs noisy?
    mask_mel  = np.clip(enh_pow / (noisy_pow + 1e-10), 0.0, 1.0)  # (N_MELS, T)

    # ── Upsample mel mask → full STFT frequency bins ─────────────────────────
    mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)  # (N_MELS, F)
    # Pseudo-inverse maps mel mask back to linear freq bins
    mel_fb_pinv  = np.linalg.pinv(mel_fb)                  # (F, N_MELS)
    mask_linear  = mel_fb_pinv @ mask_mel                   # (F, T)
    mask_linear  = np.clip(mask_linear, 0.0, 1.0)

    # ── Original STFT (complex — preserves phase) ────────────────────────────
    stft_noisy = librosa.stft(noisy_wav_np, n_fft=N_FFT, hop_length=HOP_LENGTH)
    # (F, T_stft) — T_stft may differ slightly from T_mel
    T2 = min(mask_linear.shape[1], stft_noisy.shape[1])
    mask_linear = mask_linear[:, :T2]
    stft_enh    = stft_noisy[:, :T2] * mask_linear          # keep phase, scale magnitude

    # ── Back to waveform ─────────────────────────────────────────────────────
    waveform = librosa.istft(stft_enh, hop_length=HOP_LENGTH, length=len(noisy_wav_np))
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform

# ── Spectrogram plot (returns fig, no st.pyplot inside) ──────────────────────
def make_spec_figure(spec_tensor, title):
    spec = spec_tensor.squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title, color="white", fontsize=11)
    ax.set_xlabel("Time",    color="#aaa")
    ax.set_ylabel("Mel bin", color="#aaa")
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    plt.tight_layout()
    return fig

# ── Bytes helper ─────────────────────────────────────────────────────────────
def to_wav_bytes(audio_np, sr):
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Speech Enhancement", page_icon="🎧", layout="wide")
st.title("🎧 Speech Enhancement Demo")
st.write("Upload noisy speech → compare spectrograms → hear the enhanced audio")

model, stats = load_model()

uploaded_file = st.file_uploader("Upload a noisy .wav file", type=["wav"])

if uploaded_file is not None:

    # Read bytes once, reuse everywhere
    audio_bytes = uploaded_file.read()

    # Load waveform — resample to training SR if needed
    waveform_np, file_sr = sf.read(io.BytesIO(audio_bytes))
    if waveform_np.ndim == 2:                        # stereo → mono
        waveform_np = waveform_np.mean(axis=1)
    waveform_np = waveform_np.astype(np.float32)
    if file_sr != SR:
        st.warning(f"Resampling {file_sr} Hz → {SR} Hz to match training")
        waveform_np = librosa.resample(waveform_np, orig_sr=file_sr, target_sr=SR)

    # ── Original audio player ────────────────────────────────────────────────
    st.subheader("🔊 Original (noisy) audio")
    st.audio(audio_bytes, format="audio/wav")

    # ── Build input tensor ───────────────────────────────────────────────────
    input_tensor, orig_db = waveform_to_mel(waveform_np, stats)  # (1,1,N_MELS,T)

    # ── Inference ────────────────────────────────────────────────────────────
    try:
        with torch.no_grad():
            enhanced = model(input_tensor)           # (1,1,N_MELS,T)

        enhanced_squeezed = enhanced.squeeze(0)      # (1,N_MELS,T)

        # ── Side-by-side spectrograms ─────────────────────────────────────────
        st.subheader("📊 Spectrogram comparison")
        col1, col2 = st.columns(2)
        with col1:
            # Show the raw dB mel (unnormalised) so colours are meaningful
            orig_tensor = torch.tensor(orig_db).unsqueeze(0)
            fig_orig = make_spec_figure(orig_tensor, "Original (Noisy)")
            st.pyplot(fig_orig, use_container_width=True)
            plt.close(fig_orig)
        with col2:
            # Denormalise enhanced back to dB for display
            enh_np  = enhanced_squeezed.squeeze(0).detach().cpu().numpy()
            enh_db  = enh_np * (stats["clean_std"] + 1e-8) + stats["clean_mean"]
            enh_tensor = torch.tensor(enh_db).unsqueeze(0)
            fig_enh = make_spec_figure(enh_tensor, "Enhanced")
            st.pyplot(fig_enh, use_container_width=True)
            plt.close(fig_enh)

        # ── Reconstruct audio ─────────────────────────────────────────────────
        enhanced_np      = mel_to_audio(enhanced_squeezed, waveform_np, stats)
        enhanced_wav_bytes = to_wav_bytes(enhanced_np, SR)

        st.subheader("🎧 Enhanced audio")
        st.audio(enhanced_wav_bytes, format="audio/wav")

        st.download_button(
            label="⬇️ Download enhanced WAV",
            data=enhanced_wav_bytes,
            file_name="enhanced_output.wav",
            mime="audio/wav",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Inference error: {e}")
        st.exception(e)
