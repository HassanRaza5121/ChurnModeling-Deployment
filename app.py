from src.mlproject.logger import logging
from src.mlproject.exception import CustomExceptiom
from src.mlproject.components.Data_ingestion import DataIngestion
import sys
from src.mlproject.components.Data_ingestion import DataIngestionConfig
import os

import pandas as pd

if __name__=="__main__":
    logging.info('The execution has started')
    try:
        Data_ingestion_config=DataIngestionConfig()
        Data_ingestion= DataIngestion()
        Data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info('CustomException')
        raise CustomExceptiom(e,sys)
