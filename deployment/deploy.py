# Import necessary libraries from huggingface_hub
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# FIX: same reasoning as data_register.py -- this script runs from two
# different working directories (Colab vs GitHub Actions runner), so detect
# whichever deployment folder path actually exists.
DEPLOY_FOLDER = "tourism_project/deployment" if os.path.isdir("tourism_project/deployment") else "deployment"

# Define repository details
repo_id = "imanandshah/Tourism_Package_Prediction"
repo_type = "space"

# Initialize API client using the Hugging Face token from environment variables
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the Space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk="docker",
        private=False
    )
    print(f"Space '{repo_id}' created.")

# Step 2: Upload the deployment folder (app.py, requirements.txt, Dockerfile) to the Space
api.upload_folder(
    folder_path=DEPLOY_FOLDER,
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Deployment files uploaded. The Space will now build and restart automatically.")
