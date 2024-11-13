import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
from typing import Tuple, List
import matplotlib.pyplot as plt
import pickle


class RF_Model:
    def __init__(self, 
                 n_estimators: int = 100, # number of trees
                 max_depth: int = None,   # maximum depth of trees
                 random_state: int = 20,  # random seed 
                 **kwargs):
        """
        Initialize RandomForest model.
        
        Args:
            n_estimators: Number of trees in forest
            max_depth: Maximum depth of trees
            random_state: Random seed
        """
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        if "experiments_path" in kwargs:
            self.experiments_path = kwargs["experiments_path"]
        else:
            self.experiments_path = os.path.join(os.getcwd(), 'experiments')
            print('--- No experiments path provided, using default path:', self.experiments_path)
    
    def train(self, 
            train_loader,
            val_loader, 
            epochs: int = 1,  # kept for compatibility
            save_best_model: bool = True,
            save_losses: bool = True,
            plot_losses: bool = True,
            **kwargs) -> Tuple[List[float], List[float]]:
        """
        Train the model using training data.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Not used (kept for compatibility)
            save_best_model: Whether to save best model
            save_losses: Whether to save loss history
            plot_losses: Whether to plot training curves
        """
        # Convert DataLoader to numpy arrays
        X_train = []
        y_train = []
        for features, targets in train_loader:
            X_train.append(features.numpy())
            y_train.append(targets.numpy())
        X_train = np.vstack(X_train)
        y_train = np.vstack(y_train).ravel()
        
        # Train model
        print('--- Start training RandomForest model...')
        self.model.fit(X_train, y_train)
        
        # Calculate training and validation losses
        train_loss = self._calculate_loss(train_loader)
        val_loss = self._calculate_loss(val_loader)
        
        if save_best_model:
            model_path = os.path.join(self.experiments_path, 'best_model_RF.pkl')
            with open(model_path, 'wb') as file:
                pickle.dump(self.model, file)
            print(f"--- Model saved to {model_path}")
        
        # Keep single point for compatibility with plotting
        train_losses = [train_loss]
        val_losses = [val_loss]
        
        if plot_losses:
            self._plot_losses(train_losses, val_losses)
        
        return train_losses, val_losses
    

    def evaluate(self, test_loader) -> Tuple[float, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (RMSE, R²)
        """
        X_test = []
        y_test = []
        for features, targets in test_loader:
            X_test.append(features.numpy())
            y_test.append(targets.numpy())
        X_test = np.vstack(X_test)
        y_test = np.vstack(y_test).ravel()
        
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return rmse, r2
    
    def _calculate_loss(self, data_loader) -> float:
        """Calculate MSE loss for given data loader."""
        X, y = [], []
        for features, targets in data_loader:
            X.append(features.numpy())
            y.append(targets.numpy())
        X = np.vstack(X)
        y = np.vstack(y).ravel()
        
        y_pred = self.model.predict(X)
        return mean_squared_error(y, y_pred)
    
    def _plot_losses(self, train_losses: List[float], val_losses: List[float]):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.experiments_path, 'loss_curves.png'))
        plt.close()