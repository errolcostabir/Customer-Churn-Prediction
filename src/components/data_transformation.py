import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this fucntion is responsible for data transformation
        '''
        try:
            numerical_columns=['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
            categorical_columns=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']


            num_pipeline=Pipeline(
                steps=[
                    ("scaler",StandardScaler(with_mean=False))
                ]  
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ("label_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]        
            )

            logging.info("categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data completed")
            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            input_feature_train_df=train_df.drop(['customerID','Churn'],axis=1)
            target_feature_train_df=train_df['Churn']
            target_feature_train_df= np.where(target_feature_train_df == "No", 0, 1)
            
            input_feature_test_df=test_df.drop(['customerID','Churn'],axis=1)
            target_feature_test_df=test_df['Churn']
            target_feature_test_df= np.where(target_feature_test_df == "No", 0, 1)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
          
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)