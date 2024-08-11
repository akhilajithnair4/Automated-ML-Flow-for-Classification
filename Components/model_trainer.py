from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import os

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from utils import model_evaluate



class ModelTrainingConfig:
    model_training_config=os.path.join("artifact","model.pkl")
    
class ModelTrainer:
    def __init__(self):
       self.model_trainer=ModelTrainingConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            
            X_train,y_train,X_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            
            
       # SAMPLING TECHNQIUE IF REQUIRED. CHOOSE THE REQUIRED AND UNCOMMENT IT AND COMMENT OFF THE OTHER ONES      
            #SMOTE
            smote=SMOTE()
            X_OverSmote, Y_OverSmote = smote.fit_resample(X_train,y_train)
            
            
            #Random Undersampling
            
            # Down-Sampling majority class
            UnderS = RandomUnderSampler(random_state=42,
                                        replacement=True)

            # Fit predictor (x variable)
            # and target (y variable) using fit_resample()
            X_Under, Y_Under = UnderS.fit_resample(X_train,y_train)
            
            #Random Oversampling
            
            # Over Sampling Minority class
            OverS = RandomOverSampler(random_state=42)

            # Fit predictor (x variable)
            # and target (y variable) using fit_resample()
            
            X_Over, Y_Over = OverS.fit_resample(X_train,y_train)

           #Cost Sensitive Learning
        
            # Creating and training a BalancedRandomForestClassifier
            clf = BalancedRandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)

            ###  Class Balancing Techniques Ends Here ###
            
            
            
            model_dictionary={
                
                    "LogisticRegression":LogisticRegression(),
                    "SVC":SVC(),
                    "KNeighborsClassifier":KNeighborsClassifier(),
                    "GaussianNB":GaussianNB(),
                    "DecisionTreeClassifier":DecisionTreeClassifier(),
                    "RandomForestClassifier":RandomForestClassifier(),
                    "GradientBoostingClassifier":GradientBoostingClassifier(),
                    "AdaBoostClassifier": AdaBoostClassifier()

            }  
            
            params = {
                        "LogisticRegression": {
                            "penalty": ['l1', 'l2', 'elasticnet', 'none'],
                            "C": [0.01, 0.1, 1, 10, 100],
                            "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                            "max_iter": [100, 200, 300, 500],
                            "l1_ratio": [None, 0.1, 0.5, 0.7]  # only used for 'elasticnet'
                        },
                        "RidgeClassifier": {
                            "alpha": [0.01, 0.1, 1, 10, 100],
                            "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                            "max_iter": [None, 100, 200, 300],
                            "tol": [1e-3, 1e-4, 1e-5]
                        },
                        "SVC": {
                            "C": [0.1, 1, 10, 100],
                            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                            "gamma": ['scale', 'auto'],
                            "degree": [2, 3, 4],
                            "probability": [True, False],
                            "tol": [1e-3, 1e-4, 1e-5]
                        },
                        "KNeighborsClassifier": {
                            "n_neighbors": [3, 5, 7, 9],
                            "weights": ['uniform', 'distance'],
                            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            "leaf_size": [10, 20, 30, 40],
                            "p": [1, 2, 3]
                        },
                        "GaussianNB": {
                            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
                        },
                        "DecisionTreeClassifier": {
                            "criterion": ['gini', 'entropy', 'log_loss'],
                            "splitter": ['best', 'random'],
                            "max_depth": [None, 10, 20, 30],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4],
                            "max_features": [None, 'auto', 'sqrt', 'log2']
                        },
                        "RandomForestClassifier": {
                            "n_estimators": [50, 100, 200, 300],
                            "criterion": ['gini', 'entropy', 'log_loss'],
                            "max_depth": [None, 10, 20, 30],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4],
                            "max_features": ['auto', 'sqrt', 'log2'],
                            "bootstrap": [True, False]
                        },
                        "GradientBoostingClassifier": {
                            "loss": ['log_loss', 'deviance', 'exponential'],
                            "learning_rate": [0.01, 0.05, 0.1, 0.2],
                            "n_estimators": [50, 100, 200, 300],
                            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                            "criterion": ['friedman_mse', 'mse', 'mae'],
                            "max_depth": [3, 5, 7, 9],
                            "max_features": ['auto', 'sqrt', 'log2']
                        },
                        "AdaBoostClassifier": {
                            "n_estimators": [50, 100, 200, 300],
                            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                            "algorithm": ['SAMME', 'SAMME.R']
                        }
                    }

            
            model_report=model_evaluate(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_dict=model_dictionary)
            
            print(model_report)
            
            

        except Exception as e:
            
            print(e)
            


