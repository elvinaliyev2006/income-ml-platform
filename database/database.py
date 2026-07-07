from sqlalchemy import create_engine 
import os
import pandas as pd
from dotenv import load_dotenv

class DataBase_Manager:
    def __init__(self):
        load_dotenv()
        self.user=os.getenv('MYSQL_USER')
        self.password = os.getenv("MYSQL_PASSWORD")
        self.db_name = os.getenv("MYSQL_DATABASE")
        self.host = self.host = os.getenv("DB_HOST", "localhost")
        self.port = "3306"
        self.connection_string = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
        self.engine=self.create_engine_()

    def create_engine_(self):
        return create_engine(self.connection_string)
   
    def save_dataframe(self,df,table_name,if_exists,index=False):
        df.to_sql(table_name,con=self.engine,if_exists=if_exists,index=index)

    def load_table(self,table_name):
        return pd.read_sql_table(table_name=table_name,con=self.engine)
    
