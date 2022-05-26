FROM python:3.8.6-buster

COPY random_forest.joblib /random_forest.joblib
COPY requirements.txt /requirements.txt
COPY api /api 
COPY taxifare /taxifare 

RUN pip install --upgrade pip
RUN pip install -e .


CMD uvicorn api.app:app --host 0.0.0.0 --port $PORT