import numpy as np
import pandas as pd
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from src.mlproject.logger import logging
from src.mlproject.exception import CustomExceptiom
from sklearn.metrics import r2_score,f1_score,recall_score,precision_score
import mlflow
from urllib.parse import urlparse

from dataclasses import dataclass
import os
import sys
from src.mlproject.utils import save_obj,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def eval_metrics(self,actual,pred):
        f1sc  = f1_score(actual,pred)
        prcn = precision_score(actual,pred)
        recall = recall_score(actual,pred)
        return(
            f1sc,
            prcn,
            recall
        )
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('split training and test input data')
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            
            
            
            models = {

                'DecisionTreeClassifier':DecisionTreeClassifier(),
                'LogisticRegression':LogisticRegression(),
                'RandomForestClassifier':RandomForestClassifier(),
                

                }
            
            
            
            
            params={

                'DecisionTreeClassifier':
                {
                    'criterion':['entropy','log_loss','gini']

                
                },
                'RandomForestClassifier': 
                {
                    'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]
                 }
,
                'LogisticRegression':{
                    'C': [0.1, 1, 10]
                },
                
            }
            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models,params)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name_index = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            logging.info(f'best model index is {best_model_name_index}')

            best_model_name = models[best_model_name_index]
            logging.info(f'best model name is :{best_model_name}')
            print('Best Model name is:')
            print(f'{best_model_name}')
            model_names = list(params.keys())
            logging.info(f'model names are :{model_names}')
            actual_model = ""
            for model in models:
                if model == best_model_name_index:
                    actual_model = best_model_name_index
            logging.info(f'Actual model is {actual_model}')
            
            best_params = params[actual_model]
           
           
           
            mlflow.set_registry_uri('https://dagshub.com/HassanRaza5121/mldeployement.mlflow')
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            
            
            
            # mlflow
            with mlflow.start_run():
                predicted_qualities = best_model_name.predict(X_test)
                (f1sc , prcn, recall) = self.eval_metrics(y_test,predicted_qualities)
                mlflow.log_params(best_params)

                mlflow.log_metric('f1_score',f1sc) 
                mlflow.log_metric('precision',prcn)
                mlflow.log_metric('recall', recall)


            # Model registry does not work with the file store
            if tracking_url_type_store != "file":
                #register the model
                #
                mlflow.sklearn.log_model(best_model_name,"model",registered_model_name=actual_model)
            else:
                mlflow.sklearn.log_model(best_model_name ,'model')









            if best_model_score<0.6:
                print('No best Model found')
            logging.info(f'best Model found on both training and testing data score is :{best_model_score} and the model name is :{best_model_name} ')
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model_name
                
            )

            predicted = best_model_name.predict(X_test)

            f1 = f1_score( y_test,predicted)
            r = recall_score( y_test,predicted)
            p = precision_score( y_test,predicted)
            return (
                f1,
                r,
                p
            )
        
        except Exception as e:
            raise CustomExceptiom(e,sys)