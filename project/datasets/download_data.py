import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from git import Repo


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dictionary mapping specific Folder Names (from your image) to Kaggle IDs
KAGGLE_DATASETS = {
    "WELFake": "saurabhshahane/fake-news-classification",
    "clickbait": "amananandrai/clickbait-dataset",
    "allthenews": "davidmckinley/all-the-news-dataset",
    "ISOT": "csmalarkodi/isot-fake-news-dataset"
}

# Dictionary for Git Repositories
GIT_DATASETS = {
    "fnc1": "https://github.com/FakeNewsChallenge/fnc-1.git"
}

def setup_directory():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        print(f"Created directory: {BASE_DIR}")

def download_kaggle_datasets():
    api = KaggleApi()
    api.authenticate()
    
    for folder_name, dataset_id in KAGGLE_DATASETS.items():
        target_path = os.path.join(BASE_DIR, folder_name)
        
        if os.path.exists(target_path):
            print(f"[{folder_name}] already exists. Skipping.")
            continue
            
        print(f"[{folder_name}] Downloading from Kaggle ({dataset_id})...")
        try:
            api.dataset_download_files(dataset_id, path=target_path, unzip=True)
            print(f"[{folder_name}] Download complete.")
        except Exception as e:
            print(f"[{folder_name}] Error: {e}")

def download_git_datasets():
    for folder_name, git_url in GIT_DATASETS.items():
        target_path = os.path.join(BASE_DIR, folder_name)
        
        if os.path.exists(target_path):
            print(f"[{folder_name}] already exists. Skipping.")
            continue
            
        print(f"[{folder_name}] Cloning from GitHub ({git_url})...")
        try:
            Repo.clone_from(git_url, target_path)
            print(f"[{folder_name}] Clone complete.")
        except Exception as e:
            print(f"[{folder_name}] Error: {e}")

if __name__ == "__main__":
    print(f"--- Downloading datasets into {BASE_DIR} ---")
    setup_directory()
    download_kaggle_datasets()
    download_git_datasets()
    print("--- All Operations Complete ---")