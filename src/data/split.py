"""
Split des données en ensemble d'entraînement et de test. Notre variable cible est silica_concentrate et se trouve dans la dernière colonne du dataset. L'issu de ce script seront 4 datasets (X_test, X_train, y_test, y_train) que vous pouvez stocker dans data/processed.
"""
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
from targets import DIR_DATA_SOURCE, DIR_DATA_PROCESSED, get_processed, get_log

# Globals
TARGET_COL = 'silica_concentrate'

# Log
logger = logging.getLogger('Ex DVC Data')
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(get_log('logs.log')))
for h in logger.handlers:
    h.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.setLevel(logging.DEBUG)

def main():
    source_file = os.path.abspath(os.path.join(DIR_DATA_SOURCE, 'raw.csv'))
    if not os.path.isfile(source_file):
        raise ValueError(f'Inexisting path for data source: {source_file}.')
    if not os.path.isdir(DIR_DATA_PROCESSED):
        raise ValueError(f'Inexisting path for data output: {DIR_DATA_PROCESSED}.')
    logger.debug(f'Loading data from file {source_file}.')
    df = pd.read_csv(source_file)
    logger.debug(f'Splitting data.')
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=TARGET_COL), df[TARGET_COL], test_size=0.2)
    X_train.to_csv(get_processed('X_train.csv'), index=False)
    X_test.to_csv(get_processed('X_test.csv'), index=False)
    y_train.to_csv(get_processed('y_train.csv'), index=False)
    y_test.to_csv(get_processed('y_test.csv'), index=False)
    logger.info(f'Data split completed. CSV files saved to {DIR_DATA_PROCESSED}')
    
if __name__ == '__main__':
    main()