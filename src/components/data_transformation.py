import os
import sys
from src.exception import CustomeException
from src.logger import logging
from src.utils import save_object
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

<<<<<<< HEAD

=======
>>>>>>> b44ffe5625ed150365346000b9b81ca7df56c7d2
@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path =os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
         
         try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("std_scaler", StandardScaler())
                ]

            )
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    # ('std_scaler', StandardScaler(with_mean=False))    
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            # combine numerical pipeline and categorical pipeline
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)

                ] 
                    )

            return preprocessor

         except Exception as e:
             raise CustomeException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read the train and test data as dataframes')

            logging.info('Objtaining preprocessor objects')

            preprocess_obj = self.get_data_transformer_obj()
            logging.info('Fitting the preprocessor object on train data')

            target_column = 'math_score'
            numerical_columns = ["writing_score", "reading_score"]


            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_test_df = test_df[target_column]

            # Apply preprocessing
            input_features_train_arr = preprocess_obj.fit_transform(input_feature_train_df)
            input_features_test_arr = preprocess_obj.transform(input_feature_test_df)

            # train_arr = np.c_[input_features_train_arr, np.array(input_feature_train_df)]
            # test_arr = np.c_[input_features_test_arr, np.array(input_feature_test_df)]

            # Combine processed inputs and target variable
            train_arr = np.c_[input_features_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_test_df)]

            logging.info(f'saved preprocessing object')

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocess_obj)
            
            return(
                train_arr, 
                test_arr, 
                self.transformation_config.preprocessor_obj_file_path
            )
<<<<<<< HEAD
                
=======
        
>>>>>>> b44ffe5625ed150365346000b9b81ca7df56c7d2
        except Exception as e:
            raise CustomeException(e,sys)


        
