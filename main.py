
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import time
import intake
import numpy as np
import os
import pandas as pd
# from common import remove_outliers ### our customized function
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
import copy
import argparse

from utils.utils import DataProcessor
from dataloaders.dataloaders import Custom_DataLoader
from models.ANN_MLP import ANN_MLP

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

RANDOM_STATE = 20

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_STATE)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train neural network model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config from specified path
    config = load_config(args.config)

    hidden_layers = [layer['hidden_size'] for layer in config['model']['layers']]
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    catalog_path = config['data']['catalog_path']
    experiments_path = config['experiments']['experiments_path']
    bool_save_best_model = config['experiments']['save_best_model']
    bool_save_losses = config['experiments']['save_losses']
    bool_plot_losses = config['experiments']['plot_losses']

    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)

    # Create DataLoaders and get input/output sizes
    data_loader = Custom_DataLoader(catalog_path=catalog_path, random_state=RANDOM_STATE, test_size=0.2, val_size=0.1)

    # Load data and preprocess
    train_loader, val_loader, test_loader, input_size, output_size, x_test_tensor, y_test_tensor = data_loader.load_saunders2021(bool_create_dataloaders=True)
    
    # Initialize the model
    ANN_MLP_model = ANN_MLP(input_size, output_size, hidden_layers, learning_rate, random_state=RANDOM_STATE, experiments_path=experiments_path)

    # train the model
    train_losses, val_losses = ANN_MLP_model.train(train_loader, val_loader, num_epochs, patience=100, min_delta=1e-4, verbose=True, save_best_model=bool_save_best_model, save_losses=bool_save_losses, plot_losses=bool_plot_losses)
    if bool_save_losses: ### train_losses and val_losses are lists
        np.save(os.path.join(experiments_path, 'train_losses.npy'), train_losses)
        np.save(os.path.join(experiments_path, 'val_losses.npy'), val_losses)
    # evaluate the model
    rmse, r2 = ANN_MLP_model.evaluate(test_loader)

    print(f'--- Test RMSE: {rmse:.4f}')
    print(f'--- Test R²: {r2:.4f}')

    # # Plot the training and validation losses together
    # import matplotlib.pyplot as plt
    # plt.plot(train_losses, label='Training loss')
    # plt.plot(val_losses, label='Validation loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Losses')
    # plt.legend()
    # plt.grid()
    # plt.savefig('training_validation_losses.png')


