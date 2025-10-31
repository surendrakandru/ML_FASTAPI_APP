import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# loading functions from data and model files 
# func will be used to load the model artifacts, process the user input and final predictions 
from ml.data import apply_label, process_data
from ml.model import inference, load_model

project_path = os.getcwd()  # Dynamically get the current working directory
# Construct full paths to the model artifacts
ENCODER_PATH = os.path.join(project_path, "model", "encoder.pkl")
MODEL_PATH = os.path.join(project_path, "model", "model.pkl")

# Load the pre-trained encoder (used to transform categorical features)
encoder = load_model(ENCODER_PATH)
#  Load the trained machine learning model (used for making predictions)
model = load_model(MODEL_PATH)


print("Loaded model type:", type(model))


# Pydantic data class defines the input format for incoming POST requests
# Each feature is strongly typed using Python type hints
# age: int → Defines the field type (integer).
# ... (Ellipsis) means this field is required
# example=37 provides an example value in the Swagger UI docs (/docs).
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# Sets up a FastAPI app to handle web requests
app = FastAPI() 

# create a GET endpoint, which returns a welcome message 
# Helps users know the API is running and where to send data
@app.get("/")
async def get_root():
    """ Welcome message """
    return {"message": "Welcome to the Income Prediction API! Use /data/ to make predictions."}


# create a POST endpoint that acceps inputs via Data model for model inference
@app.post("/data/")
async def post_inference(data: Data):
    # Converts request to a dictionary
    data_dict = data.dict()

    # Compute "has_capital_gain" Feature with values 1 and 0 (Ensure it matches training logic)
    data_dict["has_capital_gain"] = 1 if (data_dict["capital_gain"] - data_dict["capital_loss"]) > 0 else 0

    # Remove `capital-gain` and `capital-loss` since they were NOT used in training
    del data_dict["capital_gain"]
    del data_dict["capital_loss"]

    # clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

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

    # process_data() function to transform incoming data into the format expected by the model (just like during training).
    # this is essential to ensure consistency between training and inference.
    # Leaves numeric columns untouched (like age, hours-per-week)
    new_data_processed, _, _, _ = process_data(
        data, # A Pandas DataFrame of user input 
        categorical_features=cat_features, # List of columns that are categorical and need encoding
        training=False, # Tells the function not to fit new encoders — just use pre-fitted ones
        encoders=encoder # Passes in the pre-trained LabelEncoders (loaded from pickle)
    )

    # this is new transformed array
    print(new_data_processed.shape)

    inference = model.predict(new_data_processed) # predict the result on the transformed input
    # transform prediction values 1 and 0 to orginal labels before returning the prediction
    return {"result": apply_label(inference)}
