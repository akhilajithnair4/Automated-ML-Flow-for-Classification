import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from utils import save_object


@dataclass
class DataTransformationConfig:
    data_transformation_config=os.path.join("artifact","preprocessing.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    
    def get_transformer_object(self):
        try:
            df=pd.read_csv("income.csv")
            
            
            target_feature="income"
            df1=df.drop(columns=target_feature)
            
            
           
            
            numerical_columns=[ num_columns for num_columns in list(df1.columns) if df1[num_columns].dtypes!=object]
            
        
            #manually pass in the column names you want to do label encoding    
            #categorical_columns_label_encoder=[]
            
            #df.drop(columns=categorical_columns_label_encoder,inplace=True)
            
            categorical_columns=[ cat_columns for cat_columns in list(df1.columns) if df1[cat_columns].dtypes==object]
            
         
        #This is where the Developer is allowed to make changes. [Here he can also add Labelencoding incase of Ordinal Data]
        
        
            numerical_pipeline=Pipeline(
                [
                    ("imputer",SimpleImputer(strategy="mean")),
                    ("scaler",StandardScaler())
                    
                    
                ]
            )
            
            categorical_pipeline= Pipeline(
                [
                    ("imputer",SimpleImputer(strategy= 'most_frequent')),
                    ("onehoteoncoder",OneHotEncoder())
                    
                ]
            )
            
            
            #categorilcal_pipeline_label_encoding=Pipeline(
                #[
             #       ("imputer",SimpleImputer(strategy= 'most_frequent')),
              #      ("labelecncoder",LabelEncoder())
                    
               # ]
            # )#
            
            
            preprocessor=ColumnTransformer(
                [
                    ("num_transformer", numerical_pipeline,numerical_columns),
                    
                    ("cat_transformer",categorical_pipeline,categorical_columns)
                    
                ]
                
            )
                   
          
            return preprocessor        
            
        
        except Exception as e:
            
            print(e)
            
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            
            
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            
            target_feature="income"
            
            #seperating the train input features and target feature
            train_arr_input_features=train_data.drop(columns=target_feature)
               
            
          
            train_arr_target_feature=train_data[target_feature]
            
            
            
            #seperating the test input features and target feature
            test_arr_input_features=test_data.drop(columns=target_feature)
            
            test_arr_target_feature=test_data[target_feature]
            
            
            #accesing the preprocessor object from data transformation object
            
            preprocessor_obj=self.get_transformer_object()
            
            
            #fitting and transforming the train arr using preprcoessor object 
            train_arr_input=preprocessor_obj.fit_transform(train_arr_input_features).toarray()
            
          
            
            
            #fitting and transforming the test arr using fitted preprcoessor object on train data 
            test_arr_input=preprocessor_obj.transform(test_arr_input_features).toarray()
            
            #concatenating train arr input features and target feature so that it can be sent to model training class
            train_arr=np.c_[
                train_arr_input,train_arr_target_feature
            ]
            
            #concatenating test arr input features and target feature so that it can be sent to model training class
            
            test_arr=np.c_[
                test_arr_input,test_arr_target_feature
            ]
            
            save_object(file=self.data_transformation_config.data_transformation_config,model=preprocessor_obj)
            
         
            return (
               train_arr,test_arr
            )
        
        
        except Exception as e:
            print(e)
      
        