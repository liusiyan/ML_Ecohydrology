import torch
import numpy as np
import copy
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import random
import os

import torch.nn as nn
import torch.optim as optim

class ANN_MLP:
    def __init__(self, input_size, output_size, hidden_layers, learning_rate, random_state=20, **kwargs):
        """
        Initialize the ANN_MLP model.
        
        Parameters:
        - input_size: Size of the input layer
        - output_size: Size of the output layer
        - hidden_layers: List of hidden layer sizes
        - learning_rate: Learning rate for optimization
        - random_state: Random seed for reproducibility
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.set_seed(random_state)
        
        self.model, self.criterion, self.optimizer = self._initialize_model()

        if "experiments_path" in kwargs:
            self.experiments_path = kwargs["experiments_path"]
        else: # default path
            self.experiments_path = os.path.join(os.getcwd(), 'experiments')
            print('--- No experiments path provided, using default path:', self.experiments_path)
        
    @staticmethod
    def set_seed(seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_model(self):
        """Build the neural network architecture"""
        model_layers = [nn.Linear(self.input_size, self.hidden_layers[0]), nn.ReLU()]
        for i in range(len(self.hidden_layers) - 1):
            model_layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            model_layers.append(nn.ReLU())
        model_layers.append(nn.Linear(self.hidden_layers[-1], self.output_size))
        return nn.Sequential(*model_layers)

    def _initialize_model(self):
        """Initialize model, criterion, and optimizer"""
        model = self._build_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, criterion, optimizer

    def train(self, train_loader, val_loader, num_epochs, patience=5, min_delta=1e-4, verbose=True, save_best=False):
        """Train the model with early stopping"""
        best_val_loss = float('inf')
        counter = 0
        best_model = None
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)

            # Validation phase
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            if verbose:
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'Training Loss: {train_loss:.4f}')
                print(f'Validation Loss: {val_loss:.4f}')

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter = 0
                best_model = copy.deepcopy(self.model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        if best_model is not None:
            self.model.load_state_dict(best_model)

        if save_best:
            ## make the complete path plus model name
            save_model_path = os.path.join(self.experiments_path, 'best_model.pth')
            torch.save(self.model.state_dict(), save_model_path)
            print('--- Saved best model to:', save_model_path)
        
        return train_losses, val_losses

    def validate(self, val_loader):
        """Compute validation loss"""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def evaluate(self, test_loader):
        """Evaluate model performance"""
        self.model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                all_labels.extend(labels.numpy())
                all_predictions.extend(outputs.numpy())
        
        rmse = mean_squared_error(all_labels, all_predictions, squared=False)
        r2 = r2_score(all_labels, all_predictions)
        return rmse, r2

    def predict(self, inputs):
        """Make predictions on new data"""
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs)