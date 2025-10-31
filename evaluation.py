import os
import json
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from ml.data import process_data
from ml.model import (
    load_model,
    compute_model_metrics,
    inference
)


project_path = "/home/priyanka/Dropbox/MLOPS-Projects/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/"
# Define Paths
MODEL_PATH = os.path.join(project_path, "model", "model.pkl")
ENCODER_PATH = os.path.join(project_path, "model", "encoder.pkl")
LB_PATH = os.path.join(project_path, "model", "label_encoder.pkl")
TEST_DATA_PATH = os.path.join(project_path, "data", "test_data.csv")
METRICS_FILE = os.path.join(project_path, "model", "test_metrics.json")


# Load Test Data (saved from train_test_split)
print("Loading test data...")
test_data = pd.read_csv(TEST_DATA_PATH)
print(f"The shape of test dataset:", test_data.shape)
print(f"The columns of test dataset:", test_data.columns)


#### Prediction Pipeline
# load the model and the encoders
model = load_model(MODEL_PATH)
encoder = load_model(ENCODER_PATH)
lb_path = load_model(LB_PATH)

print("Loaded model type:", type(model))



# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_test, y_test, _, _ = process_data(
    test_data,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoders=encoder,
    lb=lb_path
)

print("Test data processed")
print(X_test.shape)
print(y_test.shape)
print("Actual values on the validation set:", y_test)


# use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test) # your code here
print("Predicted values on the validation set:", preds)


# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
metrics = {
    "Precision": round(p, 4),
    "Recall": round(r, 4),
    "F1-score": round(fb, 4)
}

print("Test Metrics:", metrics)

with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Test metrics saved to {METRICS_FILE}")
