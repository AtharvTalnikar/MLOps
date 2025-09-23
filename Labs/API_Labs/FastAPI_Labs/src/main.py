from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    response:int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}



@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    """Predict using the Decision Tree model (default)."""
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]
        prediction = predict_data(features, model_type="decision_tree")
        return IrisResponse(response=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_rf", response_model=IrisResponse)
async def predict_iris_rf(iris_features: IrisData):
    """Predict using the Random Forest model."""
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]
        prediction = predict_data(features, model_type="random_forest")
        return IrisResponse(response=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_lr", response_model=IrisResponse)
async def predict_iris_lr(iris_features: IrisData):
    """Predict using the Logistic Regression model."""
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]
        prediction = predict_data(features, model_type="logistic_regression")
        return IrisResponse(response=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    
