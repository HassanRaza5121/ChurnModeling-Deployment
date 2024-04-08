import os
import sys
from src.mlproject.exception import CustomExceptiom
from src.mlproject.logger import logging
import pandas as pd
import pymysql
from dotenv import load_dotenv
load_dotenv()
host=os.getenv('host')
user=os.getenv('user')
password=os.getenv('password')
db = os.getenv('db')
def read_sql_data():
    logging.info('Reading database started')
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        
        )
        logging.info("connection has started",mydb)
        df = pd.read_sql_query('select * from churn_modelling',mydb)
        print(df.head())
        return df
    except Exception as ex:
        raise CustomExceptiom(ex)
