import pandas as pd
import os
import logging
from util.FileLoader import smart_load

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Remove empty rows 
def clean_file(file_path):
    try: 
        df = smart_load(file_path)
        #removes rows containg empty data
        df = df.dropna()
        df.columns = (df.columns
                    .str.strip() #removes spaces at the front/end of string
                    .str.lower() #make everything lowercase
                    .str.replace(' ', "_") #replace spaces with _
                    .str.replace(r'[^0-0a-zA-Z]', '', regex=True) #removes any unwanted characters
                    )
        #changes any data types 
        change_data_type(df.columns)

        #if data is missing or empty replace those values with temporary values
        for col in df.columns:
            if df[col].dtype.kinf in 'biufc':
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('unkown')

        return df
    except Exception as e:
        logger.error(f"error in cleaning file {file_path}: {str (e)}")
        raise 
    

def change_data_type(col):   
    """
    Tries to infer if the data type is a number, datetime, or timedelta rather then a generic object(string)
    only works if it is an object 
    input: a df of a panda column
    """
    if col.dtype == "object":
        #checks if a datetime
        try:
            col_new = pd.to_datetime(col.dropna().unique())
            return col_new.dtype
        except:
            #checks if a numeric value
            try:
                col_new = pd.to_numeric(col.dropna().unique())
                return col_new.dtype
            except:
                #checks if a timedelta
                try:
                    col_new = pd.to_timedelta(col.dropna().unique())
                    return col_new.dtype
                except: 
                    #if none of the conditions found returns back as an object
                    return "object"
    else:
        return col.dtype
        
