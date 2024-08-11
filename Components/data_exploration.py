
import pandas as pd


class DataExploration:
    
    def initiate_data_exploration(self):
        
        try:
            
            
            df=pd.read_csv("income.csv")

            print("Random Walk Through Data")

            print("Top 5")

            print(" ")

            print(df.head())

            print("Bottom 5")

            print(" ")

            print(df.tail())
            
            print(" ")
            
            print("Check for Invalid Data")
            
            print(df.info())

            print(" ")
            
            print("Shape of the Data")
            
            print(df.shape)
            
            print(" ")
            
            print("Columns of the data")
            
            print(list(df.columns))
            
            
            print(" ")
            print(" ")
            
            print("Check for Null Values")
            
            null=df.isna().sum().sum()
            
            print(null)
            
            print(" ")
            
            if null==0:
                print(f'There are {null} Null values')
            else:
                print(f'There are {null} Null values')
                
            print(" ")
            
            print("Statistics of the Data")
            
            print(df.describe())
            
            print(" ")
            
            print("")
            
            print("Checkig for Unique Values in each Categorical Columns")
            print(" ")
            
            for columns in list(df.columns):
                if df[columns].dtypes==object:
                    print("Categories in {} columns are {}".format((columns),(df[columns].unique())))
                    print(" ")
                    
            print(" ")
            
            print(" Printing Numerical Columns")
            
            num_columns=[ num_columns for num_columns in  list(df.columns) if df[num_columns].dtypes!=object]
            print("We have {} numerical columns: {}".format( (len(num_columns)),(num_columns) ))  
            
            print(" ")
            
            print("Printing Categorical Columns")
            
            cat_columns=[ cat_columns for cat_columns in  list(df.columns) if df[cat_columns].dtypes==object]
            
            print("We have {} categorical columns: {}".format( (len(cat_columns)),(cat_columns)  ))
            
            
            
            
        except Exception as e:
            print(e)

    