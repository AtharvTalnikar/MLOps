import joblib

def predict_data(X, model_type="decision_tree"):
    """
    Predict the class labels for the input data using the specified model.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
        model_type (str): 'decision_tree', 'random_forest', or 'logistic_regression'.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    if model_type == "random_forest":
        model_path = "../model/iris_model_randomforest.pkl"
    elif model_type == "logistic_regression":
        model_path = "../model/iris_model_logreg.pkl"
    else:
        model_path = "../model/iris_model.pkl"
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    return y_pred
