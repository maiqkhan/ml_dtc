import pickle
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

model_file = r'model2.bin'
dv_file = r'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

class Prediction(BaseModel):
    churn_probability: float
    churn: bool

    class Config:
        orm_mode = True

class Client(BaseModel):
    job: str
    duration: int
    poutcome: str
    

app = FastAPI()

@app.post('/predict/' , response_model=Prediction)
def predict(client: Client):
    client_dict = dict(client)
    
    X = dv.transform([client_dict])

    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5
    
    
    result = {
        "churn_probability": y_pred,
        "churn": churn
    }

    
    return result
