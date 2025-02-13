import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime
import json
import numpy as np
from sklearn.metrics import accuracy_score


def get_data(root_path, type_):
    
    hd = json.load(open(os.path.join(root_path, f"{type_}_last_token.json"), encoding='utf-8'))

    enddata = []

    for hs in hd:
        enddata.append({
            "hd": hs["hd"],
            "label": hs["label"]
        })
        
    return enddata
    

def get_direction(data):

    label_1_samples = [entry["hd"] for entry in data if entry["label"] == 1]
    label_0_samples = [entry["hd"] for entry in data if entry["label"] == 0]


    label_1_mean_hd = np.mean(label_1_samples, axis=0)
    label_0_mean_hd = np.mean(label_0_samples, axis=0)

    direction_vector = label_1_mean_hd - label_0_mean_hd
    
    return direction_vector


def compute_variance_proportion(data, direction):

    X = np.array([entry["hd"] for entry in data])

    X_centered = X - np.mean(X, axis=0)

    cov_matrix = np.cov(X_centered, rowvar=False)

    total_variance = np.trace(cov_matrix)

    direction = direction / np.linalg.norm(direction)
    variance_in_direction = direction.T @ cov_matrix @ direction

    proportion = variance_in_direction / total_variance
    
    return proportion
        
    
class Model():
    def __init__(self, args):
        self.args = args

    def run(self):
        data_type = ["animals", "cities", "companies", "elements", "facts", "inventions"]
        data_dict = {}

        for dt in data_type:
            data = get_data(self.args.data_path, dt)
            data_dict[dt] = data

        proportions = []
        for dt in data_type:
            data = data_dict[dt]

            direction = get_direction(data)
            proportion = compute_variance_proportion(data, direction)
            proportions.append(proportion)
            print(f"{dt} proportion: {proportion}")
            
        average_proportion = sum(proportions) / len(proportions)
        print(f"all proportion: {average_proportion}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./prompt_1_hd/", type=str)
#    parser.add_argument("--data_path", default="../hd_data/true/llama2chat7b/", type=str)
    args = parser.parse_args()

    model = Model(args)
    model.run()