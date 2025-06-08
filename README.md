# 🎤 Real-Time Speech Enhancement on Device

This project demonstrates a real-time-capable speech enhancement model using deep learning, trained on noisy/clean paired data and optimized for deployment on edge devices (CoreML).

## 🔍 Overview

- Input: Noisy voice recordings (real-world or simulated)
- Output: Enhanced speech with reduced background noise
- Models: UNet, Conv-TasNet, Spectrogram Denoising CNN
- Deployment: Optimized and exported to CoreML

## 📁 Project Structure

- `notebooks/`: EDA, classical methods, deep learning training
- `models/`: Model architectures
- `scripts/`: Training, evaluation, model export
- `utils/`: Preprocessing, audio metrics
- `demo/`: GUI demo and sample outputs

## 🗃️ Datasets Used

- VoiceBank-DEMAND
- MUSAN
- Optional: Your own noisy recordings

## 🚀 How to Run

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/unet.yaml
python scripts/evaluate.py --model checkpoints/best_model.pt
```

## 🧠 Results

| Model       | PESQ ↑ | STOI ↑ | Runtime (ms) ↓ |
|-------------|--------|--------|----------------|
| UNet        | 3.01   | 0.92   | 12.3           |
| Conv-TasNet | 3.12   | 0.94   | 28.7           |

🎧 Hear results in `demo/samples`

## 📦 CoreML Export

```bash
python scripts/export_coreml.py --model checkpoints/best_model.pt
```

## 🛠 Tech Stack

- Python, PyTorch, torchaudio
- coremltools, ONNX
- Jupyter, Streamlit (for demo)

## 📄 License

MIT License
