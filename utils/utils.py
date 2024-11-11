import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any 
from pathlib import Path

"""
Utility functions and classes for ML_Ecohydrology project.
Contains common data processing, file handling, visualization and other helpers.
"""

class DataProcessor:
    """Class for handling data processing operations."""
    
    def __init__(self):
        """Initialize the DataProcessor class."""
        pass
    
    @staticmethod
    def normalize_data(data: np.ndarray) -> np.ndarray:
        """
        Normalize data to range [0,1].
        
        Args:
            data (np.ndarray): Input data array
            
        Returns:
            np.ndarray: Normalized data
        """
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    @staticmethod
    def remove_outliers(
        df: pd.DataFrame, 
        columns: List[str] = None, 
        outlier_threshold: float = 3.0,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Remove rows where any column has data beyond specified standard deviations from mean.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str], optional): Specific columns to check for outliers. 
                If None, all columns are used. Defaults to None.
            outlier_threshold (float, optional): Number of standard deviations to use
                as threshold. Defaults to 3.0.
            verbose (bool, optional): Whether to print removal statistics. Defaults to False.
        
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        df_include = df if columns is None else df[columns]
        df_reduced = df[
            ((np.abs(df_include - df_include.mean()) / df_include.std()) < outlier_threshold).all(axis=1)
        ]
        
        if verbose:
            rows_removed = len(df) - len(df_reduced)
            percent_removed = 100 * rows_removed / len(df)
            print(f"Removed {rows_removed} rows ({percent_removed:.1f}%) marked as outliers.")
        
        return df_reduced
    
    @staticmethod
    def get_number_of_parameters(model: Any) -> int:
        """
        Calculate total number of trainable parameters in a model.
        
        Args:
            model: Machine learning model with trainable_weights attribute
            
        Returns:
            int: Total number of trainable parameters
        """
        return np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])


# class FileHandler:
#     """Class for handling file operations."""
    
#     @staticmethod
#     def ensure_dir(directory: Union[str, Path]) -> Path:
#         """
#         Ensure directory exists, create if it doesn't.
        
#         Args:
#             directory (Union[str, Path]): Directory path
            
#         Returns:
#             Path: Path object of directory
#         """
#         directory = Path(directory)
#         directory.mkdir(parents=True, exist_ok=True)
#         return directory
    
#     @staticmethod
#     def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
#         """
#         Load data from various file formats.
        
#         Args:
#             filepath (Union[str, Path]): Path to data file
            
#         Returns:
#             pd.DataFrame: Loaded data
#         """
#         filepath = Path(filepath)
#         if filepath.suffix == '.csv':
#             return pd.read_csv(filepath)
#         elif filepath.suffix in ['.xlsx', '.xls']:
#             return pd.read_excel(filepath)
#         else:
#             raise ValueError(f"Unsupported file format: {filepath.suffix}")


# class Logger:
#     """Simple logging utility."""
    
#     def __init__(self, log_file: Union[str, Path]):
#         """
#         Initialize logger.
        
#         Args:
#             log_file (Union[str, Path]): Path to log file
#         """
#         self.log_file = Path(log_file)
#         FileHandler.ensure_dir(self.log_file.parent)
    
#     def log(self, message: str):
#         """
#         Log a message with timestamp.
        
#         Args:
#             message (str): Message to log
#         """
#         with open(self.log_file, 'a') as f:
#             timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
#             f.write(f'[{timestamp}] {message}\n')


# # Constants
# PROJECT_ROOT = Path(__file__).parent.parent
# DATA_DIR = PROJECT_ROOT / 'data'
# RESULTS_DIR = PROJECT_ROOT / 'results'

# # Ensure essential directories exist
# FileHandler.ensure_dir(DATA_DIR)
# FileHandler.ensure_dir(RESULTS_DIR)