"""
Entraînement du modèle. En utilisant les paramètres retrouvés à travers le GridSearch, on entraînera le modèle en sauvegardant le modèle entraîné dans le dossier models.
"""

import os
import logging
import numpy as np
import joblib
from targets import get_log, get_model, get_processed

# Globals
TARGET_COL = 'silica_concentrate'

# Log
logger = logging.getLogger('Ex DVC Models')
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(get_log('logs.log')))
for h in logger.handlers:
    h.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.setLevel(logging.DEBUG)


def load_model(model_target : str):
    logger.info(f'Loadin model from {model_target}')
    return joblib.load(model_target)

def save_model(model : object):
    model_target = get_model('best_model_trained.pkl')
    logger.info(f'Saving {model.__class__.__name__} model to {model_target}')
    joblib.dump(model, model_target)


def main():
    model_target = get_model('best_model.pkl')
    if not os.path.isfile(model_target):
        raise ValueError(f'Best model must be saved to {model_target}.')
    X_train_path = get_processed('X_train_scaled.csv')
    y_train_path = get_processed('y_train.csv')
    
    if not os.path.isfile(X_train_path):
        raise ValueError(f'Scaled train dataset does not exist: {X_train_path}')
    if not os.path.isfile(y_train_path):
        raise ValueError(f'Train target dataset does not exist: {y_train_path}')
    logger.debug('Loading data...')
    X_train = np.loadtxt(X_train_path, delimiter=',')
    y_train = np.loadtxt(y_train_path, delimiter=',', skiprows=1)
    model = load_model(model_target)
    model.fit(X_train, y_train)
    save_model(model)
    logger.info('Best model trained and saved.')

if __name__ == '__main__':
    main()
