"""
GridSearch des meilleurs paramètres à utiliser pour la modélisation. Vous déciderez le modèle de regression à implémenter et des paramètres à tester. À l'issue de ce script vous aurez les meilleurs paramètres sous forme de fichier .pkl que vous sauvegarderez dans le dossier models.
"""
import os
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet
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

GRID_TASKS = [
    # {
    #     'model' : LinearRegression(),
    #     'params' : {
    #         'fit_intercept': [True, False],
    #         'positive': [True, False]
    #     }
    # },
    {
        'model' : BayesianRidge(),
        'params' : {
            'alpha_1': [1e-6, 1e-3],
            'alpha_2': [1e-6, 1e-3],
            'lambda_1': [1e-6, 1e-3],
            'lambda_2': [1e-6, 1e-3],
        }
    },
    # {
    #     'model' : ElasticNet(),
    #     'params' : {
    #         'alpha' : [1e-3, 1e-2, 1.0],
    #         'l1_ratio': [0.1, 0.5, 0.9],
    #         'selection': ['cyclic', 'random'],
    #     }
    # }
]

def save_model(model : object):
    model_target = get_model('best_model.pkl')
    logger.info(f'Saving {model.__class__.__name__} model to {model_target}')
    joblib.dump(model, model_target)
    
def main():
    X_train_path = get_processed('X_train_scaled.csv')
    y_train_path = get_processed('y_train.csv')
    
    if not os.path.isfile(X_train_path):
        raise ValueError(f'Scaled train dataset does not exist: {X_train_path}')
    if not os.path.isfile(y_train_path):
        raise ValueError(f'Train target dataset does not exist: {y_train_path}')
    logger.debug('Loading data...')
    X_train = np.loadtxt(X_train_path, delimiter=',')
    y_train = np.loadtxt(y_train_path, delimiter=',', skiprows=1)
    grid = GridSearchCV(GRID_TASKS[0]['model'], GRID_TASKS[0]['params'])
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    logger.info(f'Best scored model: {model.__class__.__name__}.')
    logger.info(f'Best scored params: {grid.best_params_}.')
    logger.info(f'Best score: {grid.best_score_}.')
    save_model(model)
    logger.info('Best model found and saved.')

if __name__ == '__main__':
    main()
