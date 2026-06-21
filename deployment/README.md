---
title: Tourism Package Prediction
emoji: 🏖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
license: mit
---

# Visit With Us - Wellness Tourism Package Prediction

Predicts whether a customer is likely to purchase the Wellness Tourism Package, before the sales team contacts them.

- XGBoost classifier tuned with GridSearchCV, tracked with MLflow
- Data and model versioned on Hugging Face Hub (separate dataset and model repos)
- GitHub Actions CI/CD: data registration -> data preparation -> model training -> deployment

## Links

- GitHub Repository: https://github.com/imanandshah/Tourism_Package_Prediction
- Dataset repo: https://huggingface.co/datasets/imanandshah/Tourism_Package_Prediction_DATASET
- Model repo: https://huggingface.co/imanandshah/Tourism_Package_Prediction_MODEL
