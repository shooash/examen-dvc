import os

DIR_DATA_SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/raw'))
DIR_DATA_PROCESSED = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/processed'))
DIR_LOGS = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../..'))

def get_processed(filename : str):
    return os.path.abspath(os.path.join(DIR_DATA_PROCESSED, filename))

def get_log(filename : str):
    return os.path.abspath(os.path.join(DIR_LOGS, filename))
    
