import pandas as pd
from database import DataBase_Manager
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


df=pd.read_csv('dataset/adult.csv')
dbm=DataBase_Manager()
dbm.save_dataframe(df,'census_income','replace',False)
