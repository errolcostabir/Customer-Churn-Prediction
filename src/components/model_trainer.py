import imp
import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import r2_score,accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from imblearn.over_sampling import SMOTE


@dataclass
class ModeltrainerConfig:
    trained_mode_file_path=os.path.join("artifacts",'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModeltrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            #Perform oversampling 
            oversample = SMOTE(k_neighbors=5)
            X_smote, y_smote = oversample.fit_resample(x_train, y_train)
            x_train, y_train = X_smote, y_smote
            
            #print(y_train.value_counts())
            unique, counts = np.unique(y_train, return_counts=True)
            print(dict(zip(unique, counts)))
            
            logging.info("Best model found on training and testing dataset")
            
            model=RandomForestClassifier(random_state=1)
            model.fit(x_train,y_train)

            save_object(
                file_path=self.model_trainer_config.trained_mode_file_path,
                obj=model
            )
            predicted=model.predict(x_test)
            print("Accuracy: ",accuracy_score(predicted,y_test))
            r2_scr=r2_score(y_test,predicted)
            print("Final: ",r2_scr)
            return r2_scr

        except Exception as e:
            raise CustomException(e,sys)