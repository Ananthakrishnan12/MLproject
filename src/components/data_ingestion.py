import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","data.csv")
    
class Dataingestion:
    def __init__(self):
        self.ingestionconfig=DataIngestionConfig() 
        
    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            data=pd.read_csv("notebook\Data\stud.csv")
            logging.info("Read the dataset as DataFrame")
            
            os.makedirs(os.path.dirname(self.ingestionconfig.train_data_path),exist_ok=True)
        
            data.to_csv(self.ingestionconfig.raw_data_path,index=False,header=True)
        
            logging.info("Train Test split initiated")
        
            train_set,test_set=train_test_split(data,test_size=0.2,random_state=42)
        
            train_set.to_csv(self.ingestionconfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestionconfig.test_data_path,index=False,header=True)
        
            logging.info("Ingestion of the data is completed")
            
            return(
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
                
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=Dataingestion()
    obj.initiate_data_ingestion()
            