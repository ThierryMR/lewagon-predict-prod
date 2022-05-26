from fastapi import FastAPI
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



@app.get("/")
def root():
    return {"greeting": "hello"}



@app.get("/predict")
def predict(pickup_datetime = '2013-07-06 17:18:00',  # 2013-07-06 17:18:00
            lon1 = -73.950655,             # -73.950655
            lat1 = 40.783282,             # 
            lon2 = -73.984365,             # 
            lat2 = 40.769802,             # 
            passcount= 1):
    
    
    #Pass the inputs on the function as parameters
    
    #create a list
    
    #Create a DataFrame
    X_pred = pd.DataFrame({
    "key": ["truc"],
    "pickup_datetime": [pickup_datetime + " UTC"],
    "pickup_longitude": [float(lon1)],
    "pickup_latitude": [float(lat1)],
    "dropoff_longitude": [float(lon2)],
    "dropoff_latitude": [float(lat2)],
    "passenger_count": [int(passcount)]})

    #Instaciate the model
    model = joblib.load('random_forest.joblib')
    
    #Predict the inputed values
    prediction = model.predict(X_pred)
    
    return {"fare": prediction[0]}