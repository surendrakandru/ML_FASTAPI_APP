import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def process_data(
    X, categorical_features=[], label=None, training=True, encoders=None, lb=None
):
    """Process the data used in the machine learning pipeline.

    Processes the data using label encoding for categorical features and a
    label binarizer for the labels. This can be used in either training or inference.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoders : dict
        Dictionary of trained LabelEncoders for categorical features, used only if training=False.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer for the target variable, used only if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoders : dict
        Dictionary of trained LabelEncoders for categorical features if training=True.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer if training=True.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])
    
    encoders = encoders if encoders is not None else {}

    if training:
        encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])  # Fit and transform categorical values to integers
            encoders[col] = le  # Store the trained encoder so we can reuse it later for inference
        lb = LabelEncoder()  # Use LabelEncoder instead of LabelBinarizer
        y = lb.fit_transform(y)  # Convert labels to 0 and 1,  the target variable (salary) is converted into 0/1.

    else:
        for col in categorical_features:
            if col in encoders:
                X[col] = encoders[col].transform(X[col]) # Use pre-fitted encoders

        if lb is not None and label is not None:
            y = lb.transform(y)
    print(X.columns)

    return X.to_numpy(), y, encoders, lb # converts the Pandas DataFrame to a NumPy array for compatibility with ML models. NumPy arrays are faster for large datasets.

def apply_label(inference):
    """ Convert the binary label in a single inference sample into string output."""
    if inference[0] == 1:
        return ">50K"
    elif inference[0] == 0:
        return "<=50K"

