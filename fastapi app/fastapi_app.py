import warnings
warnings.filterwarnings('ignore')

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import joblib
from database.database import DataBase_Manager

dbm=None
try:
    dbm=DataBase_Manager()
    print("Database Manager initialized successfully.")
except Exception as e:
    print(print(f"Database bypass: Running without local MySQL. Reason: {e}"))

model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))

app=FastAPI()   

class Base_for_pred(BaseModel):
    age:int
    workclass:str
    education:str
    marital_status:str
    occupation:str
    relationship:str
    race:str
    sex:str
    capital_gain:int
    capital_loss:int
    hours_per_week:int
    native_country:str

@app.post('/predict')
def make_predict(data:Base_for_pred):
    input_dict=data.dict()
    df_input=pd.DataFrame([input_dict])

    df_input=df_input.rename(columns={
        'marital_status':'marital.status',
        'capital_gain':'capital.gain',
        'capital_loss':'capital.loss',
        'hours_per_week':'hours.per.week',
        'native_country':'native.country'
    })
    
    pred_val=model.predict(df_input)[0]

    df_to_save = df_input.copy()
    df_to_save['prediction'] = int(pred_val)

    if dbm is not None:
        try:
            dbm.save_dataframe(df_to_save, table_name='predictions_log', if_exists='append', index=False)
            db_message = 'Log saved to Dockerized MySQL successfully!'
        except Exception as e:
            db_message = f'Database write failed: {str(e)}'

    else:
        db_message = 'Cloud Mode: Prediction processed without database logging.'

    return {
        'status': 'Success',
        'prediction': int(pred_val),
        'message': db_message
    }