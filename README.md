# Real-Time Speech Enhancement on Device

This repository demonstrates a deep learning pipeline for real-time-capable speech enhancement, optimized for deployment on edge devices.

## Overview

- Input: Noisy voice recordings (real-world or simulated)
- Output: Enhanced speech with reduced background noise
- Models: UNet, Conv-TasNet, Spectrogram Denoising CNN
- Deployment: Exported to CoreML for on-device use

## Project Structure

- `notebooks/`: Exploratory analysis, baseline methods, and training notebooks
- `models/`: Model definitions and architectures
- `scripts/`: Training, evaluation, and export scripts
- `utils/`: Preprocessing and evaluation utilities
- `demo/`: Lightweight demo application with example outputs

## Datasets Used

- VoiceBank-DEMAND

## Notebook 01 Features

- Persistent caching of dataset chunks via `.pkl` files
- Custom PyTorch Dataset class for clean/noisy speech
- Configurable audio segment durations & sample rates
- Log-Mel spectrogram visualization
- Ready to integrate with neural network models (next steps)
  
## Usage

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/unet.yaml
python scripts/evaluate.py --model checkpoints/best_model.pt
```

## Example Results

| Model       | PESQ | STOI | Runtime (ms) |
|-------------|------|------|--------------|
| UNet        | 3.01 | 0.92 | 12.3         |
| Conv-TasNet | 3.12 | 0.94 | 28.7         |

## CoreML Export

```bash
python scripts/export_coreml.py --model checkpoints/best_model.pt
```

## Tech Stack

- Python, PyTorch, torchaudio
- coremltools, ONNX
- Jupyter, Streamlit (for demonstration)

## License

This code is provided for personal, non-commercial use only. All rights reserved.
