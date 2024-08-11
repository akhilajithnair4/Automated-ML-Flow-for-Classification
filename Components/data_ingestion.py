import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_path=os.path.join("artifact","train.csv")
    test_path=os.path.join("artifact","test.csv")
    raw_data=os.path.join("artifact","data.csv")
    
    
class DataIngestion:
        
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

        
    def inititate_data_ingestion(self):
       try: 
        logging.info("Reading the Data")
        df=pd.read_csv("income.csv")
       
        logging.info("Train Test Split Initiated")
        
        train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)
        
        #logginginfo("Train Test Split Completed")
        mode = 0o666
        os.makedirs(os.path.dirname(self.data_ingestion_config.train_path),mode,exist_ok=True)
        
        df.to_csv(self.data_ingestion_config.raw_data)
        
        train_data.to_csv(self.data_ingestion_config.train_path,index=False)
        
        test_data.to_csv(self.data_ingestion_config.test_path,index=False)
        
        return (
            self.data_ingestion_config.train_path,
            self.data_ingestion_config.test_path
        )
        
        
       except Exception as e:
            print(e)