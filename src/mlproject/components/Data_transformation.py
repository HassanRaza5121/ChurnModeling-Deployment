import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.mlproject.exception import CustomExceptiom
from src.mlproject.logger import logging
from src.mlproject.utils import save_obj
import os
@dataclass
class DatatransformatioConfig:
    preprocesspr_obj_file_path = os.path.join("artifacts",'preprocessor.pkl')
class Datatransformation:
    def __init__(self):
        self.data_transformation_congif = DatatransformatioConfig()
    def get_data_transformer_obj(self):
        '''
        this function is responsible for the data transformation

        '''
        try:
            numerical_col = [ 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard','IsActiveMember', 'EstimatedSalary']
            categorical_col=['Geography','Gender']
            num_pipline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            categorical_pipline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))

                ]
            )
            logging.info(f'Categorical columns:{categorical_col}')
            logging.info(f'Numerical columns:{numerical_col}')
            preprocessor = ColumnTransformer(
                [
                    ('num_pipline',num_pipline,numerical_col),
                    ('categorical_pipline',categorical_pipline,categorical_col)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomExceptiom(e,sys)
    def initiate_datatransformatio(self,train_path,test_path):
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading the training and the test data sets')
            logging.info(f'Train columns: {train_df.columns}')
            logging.info(f'Test columns: {test_df.columns}')

            logging.info('Reading the training and the test data sets')
            #logging.info(f'train path {test_df}')


            preprocessor_obj = self.get_data_transformer_obj()


            target_column_name = 'Exited'
            
            col = ['CustomerId','RowNumber','Surname','Exited']

            '''Divides the training features and the testing value'''
            input_feature_train_df = train_df.drop(columns=col)

            #logging.info(f'input_feature_train_df{input_feature_train_df}')
            traget_feature_train_df = train_df[target_column_name]



            '''Divides the testing features and target value'''
            input_feature_test_df = test_df.drop(columns=col)

            traget_feature_test_df = test_df[target_column_name]

            logging.info(f'Train columns before transformation: {input_feature_train_df.columns}')
            logging.info(f'Test columns before transformation: {input_feature_test_df.columns}')

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[
                input_feature_train_arr,np.array(traget_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(traget_feature_test_df)
            ]
            logging.info(f'saved the preprocessing object')


            save_obj(
                self.data_transformation_congif.preprocesspr_obj_file_path,
                obj = preprocessor_obj

            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_congif.preprocesspr_obj_file_path
            )





        except Exception as e:
            raise CustomExceptiom(e,sys)
