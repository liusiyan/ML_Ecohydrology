import pandas as pd
import intake
import numpy as np
import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, Any, Optional, Callable

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ML_Ecohydrology.utils.utils import DataProcessor

class Custom_DataLoader:
    """A general-purpose data loader class for processing and preparing datasets."""
    
    def __init__(self, catalog_path: str, random_state: int = 42, test_size: float = 0.20, val_size: float = 0.10, **kwargs):
        """
        Initialize the DataLoader with a catalog path.
        
        Args:
            - catalog_path: Path to the intake catalog file
            - random_state: The seed used by the random number generator
            - test_size: The proportion of the dataset to include in the test split
            - val_size: The proportion of the dataset to include in the validation split
        Returns:
        - x_train, x_val, x_test, y_train, y_val, y_test: The training, validation, and testing datasets
        """
        self.catalog_path = catalog_path
        self.catalog = intake.open_catalog(catalog_path, persist_mode='default')
        # self.preprocessors: Dict[str, Callable] = {}
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size

        # check if batch size is in kwargs
        if "batch_size" in kwargs:
            self.batch_size = kwargs["batch_size"]
        else:
            self.batch_size = 32
        
    def load_saunders2021(self, bool_create_dataloaders=False):
        cat = intake.open_catalog(self.catalog_path, persist_mode='default')
        ### Saunders2021 data processing
        src = cat["Saunders2021"]
        if not src.is_persisted:
            src.persist()
        dfs = src.read()
        dfs["PARin"] = dfs["solar"] * 0.45 * 4.57  # L26
        dfs = dfs.rename(columns={"VPDleaf": "VPD"})
        dfs = dfs[["PARin", "SWC", "VPD", "Cond", "Species"]]

        ### Anderegg2018 data processing
        src = cat["Anderegg2018"]
        if not src.is_persisted:
            src.persist()
        dfa = src.read()
        N = 1.56
        M = 1 - 1 / N
        ALPHA = 0.036
        dfa["SWC"] = dfa["SWC"].fillna(
            1 / ((1 + (-1 * (dfa["LWPpredawn"]) / ALPHA) ** N) ** M)
        )  # L17
        dfa = dfa[["PARin", "SWC", "VPD", "Cond", "Species"]]

        # Combine datasets and create binary columns from unique values in Species
        df = pd.concat([dfa, dfs])
        df = DataProcessor.remove_outliers(df, ["PARin", "VPD", "SWC", "Cond"], verbose=True)
        df = pd.get_dummies(df, columns=["Species"])

        # Split the data into features and target
        x = df.drop(columns="Cond")
        y = df["Cond"]

        # Scale the features
        x_scaled = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled, y, test_size=self.test_size, random_state=self.random_state
        )

        # further split the training set into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=self.val_size, random_state=self.random_state
        )

        # convert to PyTorch tensors and create dataloaders (optional)
        if bool_create_dataloaders:
            x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
            x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
            x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
            test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            input_size = x_train.shape[1]
            output_size = 1
            return train_loader, val_loader, test_loader, input_size, output_size, x_test_tensor, y_test_tensor
        else:
            return x_train, x_val, x_test, y_train, y_val, y_test


   