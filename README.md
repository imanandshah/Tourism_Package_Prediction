# Tourism Package Prediction - MLOps Pipeline

End-to-end MLOps pipeline that predicts whether a customer will purchase the "Visit With Us" Wellness Tourism Package, so the sales team can prioritize outreach to likely buyers.

## What this repo does

1. **Data Registration** (`model_building/data_register.py`) — uploads the raw dataset to a Hugging Face dataset repo.
2. **Data Preparation** (`model_building/prep.py`) — cleans the data and creates a stratified train/test split, pushed back to the dataset repo.
3. **Model Training** (`model_building/train.py`) — tunes an XGBoost classifier with GridSearchCV, tracks every run with MLflow, and uploads the best model to a Hugging Face model repo.
4. **Model Deployment** (`deployment/deploy.py`) — pushes a Streamlit app, Dockerfile, and Space README to a Hugging Face Space for live predictions.
5. **CI/CD** (`.github/workflows/docker.yml`) — runs all four stages automatically, in order, on every push to `main`.

## Live demo

https://huggingface.co/spaces/imanandshah/Tourism_Package_Prediction

## Repo structure

```
model_building/   data_register.py, prep.py, train.py, requirements.txt
deployment/       app.py, Dockerfile, requirements.txt, README.md (Space card), deploy.py
data/             tourism.csv
.github/workflows/docker.yml   CI/CD pipeline
```

## Required GitHub Actions secret

- `HF_TOKEN` — Hugging Face write-access token, used by all four pipeline scripts.
