import pickle
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

model_file = r'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

class Prediction(BaseModel):
    churn_probability: float
    churn: bool

    class Config:
        orm_mode = True
class Customer(BaseModel):
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    tenure: int
    monthlycharges: float
    totalcharges: float

app = FastAPI()

@app.post('/predict/' , response_model=Prediction)
def predict(customer: Customer):
    customer_dict = dict(customer)
    
    X = dv.transform([customer_dict])

    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5
    
    
    result = {
        "churn_probability": y_pred,
        "churn": churn
    }

    
    return result
