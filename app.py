import os
from dotenv import load_dotenv
import gradio as gr
import joblib
from google import genai
from huggingface_hub import hf_hub_download

load_dotenv()

# Download model files from Hugging Face if not present
if not os.path.exists("emotion_model.pkl"):
    hf_hub_download(
        repo_id="HoorenSalazar/postpartum-emotion-classifier",
        filename="emotion_model.pkl",
        local_dir="."
    )

if not os.path.exists("vectorizer.pkl"):
    hf_hub_download(
        repo_id="HoorenSalazar/postpartum-emotion-classifier",
        filename="vectorizer.pkl",
        local_dir="."
    )

emotion_classifier = joblib.load("emotion_model.pkl")
tfidf_vectorizer = joblib.load("vectorizer.pkl")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

HELPLINE = "\n\n💙 *If you're struggling, please consider reaching out to a professional. Pakistan's Umang helpline: 0317-4288665*"

def detect_emotion(text):
    vec = tfidf_vectorizer.transform([text])
    return emotion_classifier.predict(vec)[0]

def chat(message, history):
    emotion = detect_emotion(message)
    
    prompt = f"""You are a warm, empathetic postpartum support companion/therapist. 
You are NOT a replacement for professional help, but a compassionate listener, guide and a friend.
The user is a new mother who may be experiencing postpartum difficulties or postpartum depression
Based on the analysis, the user appears to be feeling: {emotion}
Respond with empathy, warmth and understanding in 7-8 sentences maximum.
Do not use clinical language. Act like a therapist and give any advices needed.
User message: {message}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    reply = response.text
    
    if emotion in ["sad", "anxious", "overwhelmed"]:
        reply += HELPLINE
    
    return reply

demo = gr.ChatInterface(
    fn=chat,
    title="🌸 Postpartum Support Companion",
    description="A safe space to share how you're feeling. I'm here to listen. 💙",
    examples=[
        "I haven't slept in days and I keep crying for no reason",
        "I feel so overwhelmed, I don't think I'm a good mother",
        "I'm actually feeling really good today, baby slept 4 hours!",
    ]
)

demo.launch()