---
title: Postpartum Support Companion
emoji: 🌸
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "5.23.0"
app_file: app.py
pinned: false
---

# 🌸 Postpartum Support Companion

A compassionate AI chatbot designed to support new mothers experiencing postpartum difficulties.

## How It Works
1. User types how they're feeling
2. A custom-trained emotion classifier (TF-IDF + Logistic Regression trained on GoEmotions dataset) detects the emotional state
3. Gemini AI generates a warm, empathetic response based on the detected emotion
4. High-distress messages automatically include a professional helpline suggestion

## ML Model
- **Dataset:** GoEmotions by Google (43,410 samples)
- **Model:** TF-IDF Vectorizer + Logistic Regression
- **Classes:** positive, neutral, sad, anxious, overwhelmed
- **Trained model on Hugging Face:** [postpartum-emotion-classifier](https://huggingface.co/HoorenSalazar/postpartum-emotion-classifier)

## Tech Stack
- Python, Scikit-learn, Gradio, Google Gemini API

## Important Note
This is a support companion, not a replacement for professional mental health care.