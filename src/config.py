"""
Configurações gerais do projeto Hands-on Machine Learning
"""

import os
from pathlib import Path

# Diretórios do projeto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"

# Criar diretórios se não existirem
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                  MODELS_DIR, EXPERIMENTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configurações de plotting
PLOT_CONFIG = {
    'figsize': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8',
    'color_palette': 'husl',
    'context': 'notebook'
}

# Configurações de modelo
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.2,
    'cv_folds': 5
}

# Configurações de logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'ml_project.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# URLs úteis para datasets
DATASET_URLS = {
    'california_housing': 'https://github.com/ageron/handson-ml2/raw/master/datasets/housing/housing.csv',
    'mnist': 'https://github.com/ageron/handson-ml2/raw/master/datasets/mnist/',
    'iris': 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
}

# Configurações do MLflow
MLFLOW_CONFIG = {
    'experiment_name': 'handson-ml-experiments',
    'tracking_uri': 'sqlite:///mlflow.db',
    'artifact_location': str(EXPERIMENTS_DIR / 'mlflow_artifacts')
}

# Configurações do Kaggle
KAGGLE_CONFIG = {
    'competitions': [
        'titanic',
        'house-prices-advanced-regression-techniques',
        'digit-recognizer',
        'nlp-getting-started'
    ]
} 