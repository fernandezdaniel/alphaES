import sys
import json
import os

import torch
import numpy as np

def create_path(folder):
    # Check if the folder path exists
    if not os.path.exists(folder):
        # Create the folder path if it did not exist
        os.makedirs(folder)

def torch_pdf(x, mean=torch.tensor(0.0), var=torch.tensor(1.0)):
    pdf = (1 / torch.sqrt(2 * torch.pi * var)) * torch.exp(-((x - mean) ** 2) / (2 * var))
    return pdf

def torch_cdf(X, loc=torch.tensor(0.0), scale=torch.tensor(1.0)):
    normal = torch.distributions.normal.Normal(
            loc=torch.zeros(1, device=X.device, dtype=X.dtype),
            scale=torch.ones(1, device=X.device, dtype=X.dtype),
        )
    return normal.cdf(X)

def preprocess_outputs(y_vals):

    y_mean = np.mean(y_vals)
    y_std = np.std(y_vals)

    y_vals = (y_vals - y_mean) / y_std
    y_train = torch.from_numpy(y_vals).double()

    return y_train, y_mean, y_std

def reset_random_state(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config_data = json.load(file)
        
        print(f"[OK] Configuration loaded successfully.")
        return config_data

    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"There was an error decoding the file {file_path}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
