"""
Utility functions for DataFrame operations and logging.

Provides reusable functions for column detection and action logging.

@author: Feurking
"""

from datetime import datetime
from functools import wraps
from typing import Callable

import pandas as pd
import numpy as np

def log_action(action: str) -> Callable:
    """
    Decorator to log actions performed on the data.

    @param action: The action to log
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            print(f"[INFO {start_time}] - {action}")
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            if hasattr(wrapper, 'log'):
                wrapper.log.append({"action": action, "duration": duration, "timestamp": start_time})
            return result
        return wrapper
    return decorator

def verify_column_exists(df, col_name):
    """ Vérifie si la colonne existe et propose des alternatives si besoin. """
    if col_name not in df.columns:
        import difflib
        matches = difflib.get_close_matches(col_name, df.columns, n=5, cutoff=0.4)
        raise ValueError(f"⚠️ Colonne '{col_name}' introuvable. Suggestions proches : {matches}")

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep=",", encoding="utf-8")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        raise


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Get numeric columns in the DataFrame.

    @param df: The DataFrame to analyze
    @return: A list of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> list:
    """
    Get categorical columns in the DataFrame.

    @param df: The DataFrame to analyze
    @return: A list of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def get_datetime_columns(df: pd.DataFrame) -> list:
    """
    Get datetime columns in the DataFrame.

    @param df: The DataFrame to analyze
    @return: A list of datetime column names
    """
    return df.select_dtypes(include=['datetime']).columns.tolist()

def get_ordinal_columns(df: pd.DataFrame) -> list:
    """
    Get ordinal columns (categorical with limited unique values).

    @param df: The DataFrame to analyze
    @return: A list of ordinal column names
    """
    return [col for col in get_categorical_columns(df) if df[col].nunique() <= 10]

def get_nominal_columns(df: pd.DataFrame) -> list:
    """
    Get nominal columns (categorical with many unique values).

    @param df: The DataFrame to analyze
    @return: A list of nominal column names
    """
    return [col for col in get_categorical_columns(df) if df[col].nunique() > 10]
