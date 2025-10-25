from huggingface_hub import hf_hub_download
import os

# Folder where you want to save the model
local_dir = './models/led-large-16384-arxiv'
os.makedirs(local_dir, exist_ok=True)

# Model repo ID on Hugging Face
model_id = 'allenai/led-large-16384-arxiv'

# Only existing files in the repo
files_to_download = [
    'config.json',
    'pytorch_model.bin',
    'tokenizer_config.json',
    'special_tokens_map.json'
]

for file_name in files_to_download:
    hf_hub_download(repo_id=model_id, filename=file_name, cache_dir=local_dir)
    print(f'Downloaded {file_name}')
