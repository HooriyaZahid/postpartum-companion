from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="emotion_model.pkl",
    path_in_repo="emotion_model.pkl",
    repo_id="HoorenSalazar/postpartum-emotion-classifier",
    repo_type="model",
   token="your_hf_token_here"
)

api.upload_file(
    path_or_fileobj="vectorizer.pkl",
    path_in_repo="vectorizer.pkl",
    repo_id="HoorenSalazar/postpartum-emotion-classifier",
    repo_type="model",
    token="your_hf_token_here"
)

print("Model uploaded successfully!")