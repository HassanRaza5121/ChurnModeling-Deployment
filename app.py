from src.mlproject.logger import logging
from src.mlproject.exception import CustomExceptiom
from src.mlproject.components.Data_ingestion import DataIngestion
import sys
from src.mlproject.components.Data_ingestion import DataIngestionConfig
from src.mlproject.components.Data_transformation import DatatransformatioConfig,Datatransformation
from src.mlproject.components.Model_trainer import ModelTrainerConfig,ModelTrainer
import os

import pandas as pd

if __name__=="__main__":
    logging.info('The execution has started')
    try:

        Data_ingestion_config=DataIngestionConfig()
        Data_ingestion= DataIngestion()
        train_data_path,test_data_path = Data_ingestion.initiate_data_ingestion()


        Data_transformation_config=DatatransformatioConfig()
        Data_tranformation= Datatransformation()
        train_arr,test_arr,_ = Data_tranformation.initiate_datatransformatio(train_data_path,test_data_path)

        model_trainer_config = ModelTrainerConfig()
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.info('CustomException')
        raise CustomExceptiom(e,sys)
