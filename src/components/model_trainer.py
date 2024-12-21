import os 
import sys 
from src.logger import logging
from src.exception import CustomeException
from src.utils import save_object, evaluate_models

import pandas as pd
import numpy as np
from dataclasses import dataclass

# importing models from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor 

from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor 
)
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):

        try:
            logging.info('splitting training and test input data')
            X_train, y_train, X_test, y_test=(
                                                train_array[:,:-1],
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1] )
                                                 
            
            models = {
                'LinearRegression': LinearRegression(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'XGBRegressor': XGBRegressor(),
                'KNeighboursRegressor': KNeighborsRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(verbose=False),
                'CatBoostRegressor': CatBoostRegressor()
            }
                        
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,
                                                 X_test=X_test, y_test=y_test,
                                             models=models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomeException('No best model found')
            logging.info('Best model found on both training and testing data')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info('Model saved successfully')


            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square

           
        except CustomeException as e:
            raise CustomeException(e,sys)