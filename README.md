---
title: Abhivyakti - ISL Sign to Speech
emoji: 🤟
colorFrom: orange
colorTo: green
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
---

# 🤟 Abhivyakti — Indian Sign Language to Speech Converter

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat-square&logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-4.7-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **अभिव्यक्ति** (Abhivyakti) means *"Expression"* in Sanskrit/Hindi.  
> This project bridges the communication gap for the Deaf and Hard-of-Hearing community in India by converting Indian Sign Language (ISL) gestures into real-time spoken audio.

---

## 🎯 Problem Statement

Over **18 million people** in India are Deaf or Hard-of-Hearing. Indian Sign Language (ISL) is their primary mode of communication, yet very few real-time ISL recognition tools exist — especially ones that work with Indian skin tones, lighting conditions, and word-level signs (not just A-Z alphabets).

**Abhivyakti** solves this with a real-time, webcam-based ISL-to-speech system.

---

## 🌟 Demo

> 📹 *Demo GIF will be added after Phase 3*

**Live App:** [🤗 Hugging Face Space](#) *(link after deployment)*

---

## 🏗️ Architecture

```
Webcam Input
     │
     ▼
MediaPipe Hands
(21 landmarks × 3 axes = 63 values per frame)
     │
     ▼
Sequence Buffer (30 frames)
     │
     ▼
LSTM / Transformer Model
     │
     ▼
Predicted ISL Word + Confidence Score
     │
     ▼
gTTS / pyttsx3 → Audio Output 🔊
```

---

## 📦 ISL Word Classes (Phase 2 — 20 words)

| # | Word | # | Word |
|---|------|---|------|
| 1 | Hello | 11 | Stop |
| 2 | Help | 12 | Come |
| 3 | Water | 13 | Go |
| 4 | Thank You | 14 | Name |
| 5 | Yes | 15 | I/Me |
| 6 | No | 16 | You |
| 7 | Hospital | 17 | Good |
| 8 | Food | 18 | Bad |
| 9 | Pain | 19 | More |
| 10 | Please | 20 | Time |

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/abhivyakti.git
cd abhivyakti
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Collect Your Own Data (Optional)
```bash
python src/collect_data.py
```

### 5. Train the Model
```bash
python src/train_model.py
```

### 6. Run the App
```bash
python app.py
```

---

## 📁 Project Structure

```
abhivyakti/
├── data/
│   └── keypoints/           # Saved keypoint sequences (.npy)
├── models/
│   └── isl_model.h5         # Trained LSTM model
├── notebooks/
│   └── training.ipynb       # Model training notebook
├── src/
│   ├── collect_data.py      # Data collection script
│   ├── train_model.py       # Model training script
│   ├── inference.py         # Real-time inference engine
│   └── tts.py               # Text-to-speech module
├── app.py                   # Gradio web app
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Model Performance

> *Metrics will be added after training in Phase 2*

| Metric | Score |
|--------|-------|
| Test Accuracy | TBD |
| Inference Speed | TBD |
| Signs Supported | 20 |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Hand Tracking | MediaPipe Hands |
| Model | LSTM / Transformer (TensorFlow) |
| Speech | gTTS + pyttsx3 |
| Web App | Gradio |
| Deployment | Hugging Face Spaces |

---

## 🗺️ Roadmap

- [x] Project setup & GitHub repo
- [ ] MediaPipe data collection pipeline
- [ ] LSTM model — 5 signs (pipeline test)
- [ ] Scale to 20 ISL words
- [ ] TTS integration
- [ ] Gradio web app
- [ ] Hugging Face deployment
- [ ] Demo GIF + full README

---

## 🤝 Contributing

Contributions welcome! Especially:
- Adding more ISL word samples
- Improving model accuracy
- UI/UX improvements

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Built with ❤️ for the Indian Deaf community.  
*"Technology should speak every language — including sign language."*
