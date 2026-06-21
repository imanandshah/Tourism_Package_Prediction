# import necessary libraries
import os
import json
import joblib
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Suppress noisy MLflow client logs in CI/CD logs
logging.getLogger("mlflow").setLevel(logging.WARNING)

# Set up MLflow tracking (local file store inside the CI/CD runner)
mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_experiment("tourism-package-prediction-experiment")

# Initialize Hugging Face API client using the token from environment variables
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO = "imanandshah/Tourism_Package_Prediction_DATASET"
MODEL_REPO = "imanandshah/Tourism_Package_Prediction_MODEL"

# Ensure the model repo exists before we try to upload the trained model to it later
# in this script (mirrors the create-if-missing pattern used in data_register.py)
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repo '{MODEL_REPO}' not found. Creating new repo...")
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
    print(f"Model repo '{MODEL_REPO}' created.")

# Load the prepared train/test files from the Hugging Face dataset repo
Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").squeeze()
ytest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").squeeze()
print("Train/test data loaded successfully from Hugging Face dataset repo.")

# Define feature groups (identical to dev experiment)
nominal_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus']
ordinal_features = ['Designation']
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('nom', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), nominal_features),
        ('ord', OrdinalEncoder(
            categories=[['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']],
            handle_unknown='use_encoded_value', unknown_value=-1
        ), ordinal_features)
    ]
)

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss', scale_pos_weight=class_weight, random_state=42
)

# Define hyperparameter grid (identical to dev experiment)
param_grid = {
    'xgbclassifier__n_estimators': [150, 175, 200, 225, 250],
    'xgbclassifier__max_depth': [3, 4, 5],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.05, 0.1, 0.15],
    'xgbclassifier__reg_lambda': [0.5, 0.6, 0.7],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_score", results['mean_test_score'][i])
            mlflow.log_metric("std_test_score", results['std_test_score'][i])

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    mlflow.sklearn.log_model(
        best_model,
        name="model",
        serialization_format="cloudpickle"
    )

    print("Best Parameters:", grid_search.best_params_)
    print("Test F1-score:", test_report['1']['f1-score'])

# Save the best model locally
model_filename = "best_tourism_model_v1.joblib"
joblib.dump(best_model, model_filename)
print(f"Model saved locally as {model_filename}")

# Upload the trained model to the Hugging Face model repo
api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Model uploaded to Hugging Face model repo '{MODEL_REPO}'.")
