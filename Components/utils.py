import pickle
import os
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pandas as pd



def save_object(file,model):
    try:
      with open(file,'wb') as files:
        pickle.dump(model, files)
        
    except Exception as e:
        print(e)
        
        
        
        
def model_evaluate(X_train,y_train,X_test,y_test,model_dict):
    
    try:
       
        report_list=[]
        
        for models in (list(model_dict.keys())):
            report_dict={}
            model=model_dict[models]
            
          #without using hyperparameter->comment this off if you are using hyperparameter  
            model.fit(X_train,y_train)
            
            prediction=model.predict(X_test)
            
          #with hyperparameter-> uncomment this if you are using this 
            
            #need to add this in function parameter->params
            #params=params[models]
            
            #grid_search=GridSearchCV(model,params,cv=3)
            
            
            
            #grid_search.set_params(**gs.best_params_)
            #grid_search.fit(X_train,y_train)
            
            #prediction=grid_search.predict(X_test)
        
        # USING HYPERPARAMETER TUNING ENDS HERE
            
            accuracy=accuracy_score(y_test,prediction)
            precision=precision_score(y_test,prediction)
            recall=recall_score(y_test,prediction)
            f1=f1_score(y_test,prediction)
            
            report_dict["Model"]=models
            report_dict['Accuracy']=accuracy
            report_dict['Precision']=precision
            report_dict['Recall']=recall
            report_dict['F1Score']=f1
            
            report_list.append(report_dict)
            
       
        report_df=pd.DataFrame(report_list)
        
        report_df=report_df.sort_values(by=['Accuracy','Precision','Recall','F1Score'],ascending=[False,False,False,False])
      
        return report_df
            
                
    except Exception as e:
        print(e)