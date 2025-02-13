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

    
def binary_eval(predy, testy):

    acc = accuracy_score(testy, predy)
    
    return acc
 

class Model():
    def __init__(self, args):
    
        self.args = args
        

    def run(self):
        data_type = ["animals", "cities", "companies", "elements", "facts", "inventions"]

        train_data = get_data(self.args.data_path, data_type[self.args.train_number])
        print(f"Training on: {data_type[self.args.train_number]}")

        direction = get_direction(train_data)
        direction_tensor = torch.tensor(direction, dtype=torch.float32, device=self.args.device)

        for test_idx, test_dt in enumerate(data_type):
            if test_idx != self.args.train_number:
                test_data = get_data(self.args.data_path, test_dt)
                print(f"Testing on: {test_dt}")

                predy, label = [], []
                for x in test_data:
                    x_tensor = torch.tensor(x["hd"], dtype=torch.float32, device=self.args.device)
                    score = torch.sigmoid(torch.matmul(x_tensor, direction_tensor))
                    pred = score.round()

                    predy.append(pred.item())
                    label.append(x["label"])

                acc = binary_eval(predy, label)
                print(f"Accuracy on {test_dt}: {acc}")
            
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="../../output/", type=str)
    parser.add_argument("--data_path", default="../../hd_data_prompt/true/llama2chat7b/", type=str)
    
    parser.add_argument("--train_number", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    
    args = parser.parse_args()
    
    model = Model(args)

    model.run()
