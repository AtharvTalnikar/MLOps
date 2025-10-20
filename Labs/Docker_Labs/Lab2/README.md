# Iris Classifier API

Flask-based API that serves predictions for the Iris dataset using two models:

- TensorFlow Keras model: `my_model.keras`
- Scikit-learn RandomForest model: `iris_rf.joblib`


## What Changed

- Added a multistage Dockerfile: `dockerfile.multistage`
  - Purpose: reduce final image size by separating build/training and runtime stages.
  - Uses `python:3.9-slim` for both stages.
  - Trains models in the builder stage and copies only the trained artifacts and a prebuilt virtualenv into the runtime image.
- Added a new scikit-learn training script: `src/train_sklearn.py`
  - Trains a `RandomForestClassifier` on Iris, persists a bundle (`scaler` + `model`) to `iris_rf.joblib`.
- Updated the Flask app: `src/main.py`
  - Loads `iris_rf.joblib` on startup.
  - New endpoint: `POST /predict_sklearn` for scikit-learn predictions.

Originally there was no multistage Dockerfile. The new `dockerfile.multistage` reduces image size by avoiding build tools and caches in the final image and shipping only the runtime artifacts.


## Build and Run (Multistage Dockerfile)

1) Build the image:

```
docker build -f dockerfile.multistage -t iris-app .
```

2) Run the container (Flask on port 4000):

```
docker run --rm -p 4000:4000 iris-app
```


## API Endpoints

- `GET /` — health/welcome.
- `GET /predict` — returns the HTML form (TensorFlow UI).
- `POST /predict` — TensorFlow model prediction (form-encoded body):
  - `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- `POST /predict_sklearn` — Scikit-learn prediction (JSON or form body):
  - `sepal_length`, `sepal_width`, `petal_length`, `petal_width`

