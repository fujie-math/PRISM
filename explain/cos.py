import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime
import json
import numpy as np
from sklearn.metrics import accuracy_score
import random
from sklearn.decomposition import PCA


def get_data(root_path, type_):
    
    hd = json.load(open(os.path.join(root_path, f"{type_}_last_token.json"), encoding='utf-8'))
    
    halu_hd = []
    
    for hs in hd:
        halu_hd.append(hs["hd"])

    enddata = []

    for i in range(len(halu_hd)):
        enddata.append({
            "hd": halu_hd[i],
            "label": hd[i]["label"]
        })
    
    all_hd = np.array([entry["hd"] for entry in enddata])
    mean_hd = np.mean(all_hd, axis=0)
    
    for entry in enddata:
        entry["hd"] = np.array(entry["hd"]) - mean_hd
        
    label_1_samples = [entry["hd"] for entry in enddata if entry["label"] == 1]
    label_0_samples = [entry["hd"] for entry in enddata if entry["label"] == 0]

    label_1_mean_hd = np.mean(label_1_samples, axis=0) if label_1_samples else None
    label_0_mean_hd = np.mean(label_0_samples, axis=0) if label_0_samples else None

    return [
        {"hd": label_1_mean_hd, "label": 1},
        {"hd": label_0_mean_hd, "label": 0}
    ]


def compute_average_directions(data_types, data_paths):
    directions = []

    for data_type, data_path in zip(data_types, data_paths):
        data = get_data(data_path, data_type)
        
        total_vector_1 = None
        total_vector_0 = None
        
        for entry in data:
            if entry["label"] == 1:
                total_vector_1 = entry["hd"] if total_vector_1 is None else total_vector_1 + entry["hd"]
            elif entry["label"] == 0:
                total_vector_0 = entry["hd"] if total_vector_0 is None else total_vector_0 + entry["hd"]

        average_direction = total_vector_1 - total_vector_0 if total_vector_1 is not None and total_vector_0 is not None else None
        directions.append(average_direction)

    return directions


data_type_1 = ["animals", "cities", "companies","elements", "facts", "inventions"]

data_path_1 = "./prompt_1_hd"
data_path_2 = "../hd_data/true/llama2chat7b"

data_types = data_type_1 
data_paths = [data_path_1] * len(data_type_1)

average_directions = compute_average_directions(data_types, data_paths)


def compute_cosine_matrix(directions):
    num_directions = len(directions)
    cosine_matrix = np.zeros((num_directions, num_directions))

    for i in range(num_directions):
        for j in range(num_directions):
            if directions[i] is not None and directions[j] is not None:
                norm_i = np.linalg.norm(directions[i])
                norm_j = np.linalg.norm(directions[j])
                inner_product = np.dot(directions[i], directions[j])
                cosine_matrix[i, j] = inner_product / (norm_i * norm_j) if norm_i != 0 and norm_j != 0 else None

    return cosine_matrix

cosine_matrix = compute_cosine_matrix(average_directions)
cosine_matrix_rounded = np.round(cosine_matrix, 4)

print("Cosine Similarity Matrix:")
print(cosine_matrix_rounded)
