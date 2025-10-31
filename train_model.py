import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    save_model,
    train_model
)
# load the cencus.csv data
project_path = "/home/priyanka/Dropbox/MLOPS-Projects/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/"
data_path = os.path.join(project_path, "data", "census_cleaned_data.csv")
print(data_path)
data = pd.read_csv(data_path) # your code here
print(f"Sucessfully loaded the dataset from: {data_path}")
print(f"The shape of the dataset:", data.shape)
#print(f"The columns in the dataset:", data.columns)


# split the provided data to have a train dataset and a test dataset
# Split into Train (24,000) & Test (6,000)
# Stratification ensures class distribution is preserved in train and test sets.
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['salary'])
print("The shape of the training data:", train_data.shape)
print("The shape of the test data:", test_data.shape)


# Save test data for evaluation
test_data_path = os.path.join(project_path, "data", "test_data.csv")
test_data.to_csv(test_data_path, index=False)
print("Train-Test Split Done! Test data saved for evaluation.")


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

# use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    X=train_data,
    categorical_features=cat_features,
    label="salary",
    training=True
    )


# use the train_model function to train the model on the training dataset
# Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}
model = train_model(X_train, y_train)

print("Saving model of type:", type(model))


# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
lb_path = os.path.join(project_path, "model", "label_encoder.pkl")
save_model(lb, lb_path)



