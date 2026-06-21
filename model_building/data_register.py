# Import necessary libraries from huggingface_hub
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# FIX: this script is committed to the repo and then run from two different
# working directories -- locally in Colab (where tourism_project/data exists
# under /content/) and on the GitHub Actions runner (where the repo root IS
# what used to be tourism_project/, so the folder is just "data"). Detect
# whichever path actually exists instead of hardcoding one.
DATA_FOLDER = "tourism_project/data" if os.path.isdir("tourism_project/data") else "data"

# Define repository details
repo_id = "imanandshah/Tourism_Package_Prediction_DATASET"
repo_type = "dataset"

# Initialize API client using the Hugging Face token from environment variables
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Step 2: Upload the data folder to the repo
api.upload_folder(
    folder_path=DATA_FOLDER,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload data folder dataset"
)
