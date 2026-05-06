from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="./models", repo_id="itsprzvl/pepper_model", repo_type="model")
