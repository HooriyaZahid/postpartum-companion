from google import genai

GEMINI_API_KEY = "AIzaSyARblday30RqOgtZkIWXL3WLPGIy6y3yqM"
client = genai.Client(api_key=GEMINI_API_KEY)

for model in client.models.list():
    print(model.name)