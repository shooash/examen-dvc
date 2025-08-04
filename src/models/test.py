"""
Evaluation du modèle. Finalement, en utilisant le modèle entraîné on évaluera ses performances et on fera des prédictions avec ce modèle de sorte qu'à la fin de ce script on aura un nouveau dataset dans data qui contiendra les predictions ainsi qu'un fichier scores.json dans le dossier metrics qui récupérera les métriques d'évaluation de notre modèle (i.e. mse, r2, etc).
"""

import json
import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error as RMSE, mean_absolute_error as MAE, r2_score
import joblib
from targets import get_log, get_model, get_processed, get_data, get_metrics

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

def main():
    model_target = get_model('best_model_trained.pkl')
    if not os.path.isfile(model_target):
        raise ValueError(f'Best model must be trained and saved to {model_target}.')
    X_test_orig_path = get_processed('X_test.csv')
    X_test_path = get_processed('X_test_scaled.csv')
    y_test_path = get_processed('y_test.csv')
    
    if not os.path.isfile(X_test_path):
        raise ValueError(f'Scaled test dataset does not exist: {X_test_path}')
    if not os.path.isfile(y_test_path):
        raise ValueError(f'Test target dataset does not exist: {y_test_path}')
    if not os.path.isfile(X_test_orig_path):
        raise ValueError(f'Original test dataset does not exist: {X_test_orig_path}')
    logger.debug('Loading data...')
    X_test = np.loadtxt(X_test_path, delimiter=',')
    y_test = np.loadtxt(y_test_path, delimiter=',', skiprows=1)
    
    model = load_model(model_target)
    
    logger.debug('Predicting...')
    y_pred = model.predict(X_test)
    logger.debug('Processing and saving prediction with original data.')    
    X_orig = pd.read_csv(X_test_orig_path)
    X_orig[TARGET_COL] = y_test
    X_orig[TARGET_COL + '_pred'] = y_pred
    X_orig.to_csv(get_data('prediction.csv'), index=False)
    logger.info(f"""Predictions saved to {get_data('prediction.csv')}""")

    logger.debug('Calculating and saving metrics...')
    metrics = {
        'RMSE' : RMSE(y_test, y_pred),
        'MAE' : MAE(y_test, y_pred),
        'R2' : r2_score(y_test, y_pred)
    }
    metrics_file = get_metrics('scores.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)
    logger.info('Best model used for prediction, results and scores saved.')

if __name__ == '__main__':
    main()
