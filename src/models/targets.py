import os

DIR_MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../models'))
DIR_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data'))
DIR_DATA_PROCESSED = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../data/processed'))
DIR_LOGS = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../..'))
DIR_METRICS = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../metrics'))


def get_processed(filename : str):
    return os.path.abspath(os.path.join(DIR_DATA_PROCESSED, filename))

def get_log(filename : str):
    return os.path.abspath(os.path.join(DIR_LOGS, filename))

def get_model(filename : str):
    return os.path.abspath(os.path.join(DIR_MODELS, filename))

def get_data(filename : str):
    return os.path.abspath(os.path.join(DIR_DATA, filename))
    
def get_metrics(filename : str):
    return os.path.abspath(os.path.join(DIR_METRICS, filename))
