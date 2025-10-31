import pickle
import os
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score
from ml.data import process_data
# TODO: add necessary import

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model : XGBClassifier
        Best-trained XGBoost model using GridSearchCV.
    """
    # Compute class weight for imbalanced dataset
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # 0 -> `<=50K`, 1 -> `>50K`

     # Define XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight)

    # Define hyperparameter grid for tuning
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    # Perform 5-fold Cross-Validation with GridSearchCV
    grid_search = GridSearchCV(model, param_grid, scoring="f1", cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)  # Train and find best parameters

    # Get the best model
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = f1_score(y, preds, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Now, predict using the preprocessed data
    preds = model.predict(X)
    return preds

def save_model(model, model_path):
    """ 
    Serializes the trained machine learning model and encoder to files.

    Inputs
    ------
    model : object
        Trained machine learning model (e.g., XGBClassifier).
    encoder : object
        Trained LabelEncoder or OneHotEncoder.
    model_path : str
        Path to save the model pickle file.
    encoder_path : str
        Path to save the encoder pickle file.

    Returns
    -------
    None
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved successfully at: {model_path}")

    except Exception as e:
        print(f"Error saving model or encoder: {e}")


def load_model(model_path):
    """ 
    Loads a serialized model and encoder from pickle files.

    Inputs
    ------
    model_path : str
        Path to the saved model pickle file.
    encoder_path : str
        Path to the saved encoder pickle file.

    Returns
    -------
    model : object
        Trained machine learning model.
    encoder : object
        Trained LabelEncoder or OneHotEncoder.
    """
    try:
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Successfully loaded model from: {model_path}")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None  # Return None if loading fails

'''
def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # TODO: implement the function
    X_slice, y_slice, _, _ = process_data(
        # your code here
        # for input data, use data in column given as "column_name", with the slice_value 
        # use training = False
    )
    preds = None # your code here to get prediction on X_slice using the inference function
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
'''