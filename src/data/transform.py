"""
Normalisation des données. Comme vous pouvez le noter, les données sont dans des échelles très variés donc une normalisation est nécessaire. Vous pouvez utiliser des fonctions pré-existantes pour la construction de ce script. En sortie, ce script créera deux nouveaux datasets : (X_train_scaled, X_test_scaled) que vous sauvegarderez également dans data/processed.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from targets import get_processed, get_log

# Log
logger = logging.getLogger('Ex DVC Data')
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(get_log('logs.log')))
for h in logger.handlers:
    h.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.setLevel(logging.DEBUG)

def main():
    X_train_file = get_processed('X_train.csv')
    X_test_file = get_processed('X_test.csv')
    if not os.path.isfile(X_train_file):
        raise ValueError(f'Train dataset file does not exists: {X_train_file}')
    if not os.path.isfile(X_test_file):
        raise ValueError(f'Train dataset file does not exists: {X_test_file}')
    logger.debug(f'Loading train dataset {X_train_file}.')
    X_train = pd.read_csv(X_train_file)
    logger.debug(f'Loading test dataset {X_test_file}.')
    X_test = pd.read_csv(X_test_file)
    logger.debug('Scaling data...')
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.drop(columns='date'))
    X_test = scaler.transform(X_test.drop(columns='date'))
    logger.debug('Saving data...')
    np.savetxt(get_processed('X_train_scaled.csv'), X_train, delimiter=',')
    np.savetxt(get_processed('X_test_scaled.csv'), X_test, delimiter=',')
    logger.info(f"""Data transformation completed. Files saved to {get_processed('X_train_scaled.csv')} and {get_processed('X_test_scaled.csv')}""")

if __name__ == '__main__':
    main()