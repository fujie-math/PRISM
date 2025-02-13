import os
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime
import json
import numpy as np
from sklearn.metrics import accuracy_score
import random
import copy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def binary_eval(predy, testy):
    acc = accuracy_score(testy, predy)
    return acc
 
    
def get_data(root_path, type_):
    
    hd = json.load(open(os.path.join(root_path, f"{type_}_last_token.json"), encoding='utf-8'))

    enddata = []

    for hs in hd:
        enddata.append({
            "hd": hs["hd"],
            "label": hs["label"]
        })
        
    return enddata


class TrainDataset(Dataset):
    def __init__(self, train_data, args, typ="train"):
        self.all_data = []

        for _, data in enumerate(train_data):
            self.all_data.append({
                "label": data["label"],
                "hd": data["hd"]
            })
        self.halu_num = len([d for d in self.all_data if d["label"]])
        print(f"{typ} data: [0, 1] - [{len(self.all_data) - self.halu_num}, {self.halu_num}]")
            
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        data = self.all_data[idx]  
        return {
            "input": torch.tensor(data["hd"]),
            "y": torch.LongTensor([data["label"]]),
        }
    

class Model():
    def __init__(self, args):
        self.args = args
        input_size = self.args.input_size
        self.model = nn.Sequential()
        self.model.add_module("dropout", nn.Dropout(args.dropout))
        self.model.add_module(f"linear1", nn.Linear(input_size, 256))
        self.model.add_module(f"relu1", nn.ReLU())
        self.model.add_module(f"linear2", nn.Linear(256, 128))
        self.model.add_module(f"relu2", nn.ReLU())
        self.model.add_module(f"linear3", nn.Linear(128, 64))
        self.model.add_module(f"relu3", nn.ReLU())
        self.model.add_module(f"linear4", nn.Linear(64, 2))
        self.model.to(args.device)

    def run(self, optim):
        epoch, epoch_start = self.args.train_epoch, 1
        
        data_type_1 = ["animal_class", "cities", "element_symb", "facts", "inventors", "sp_en_trans"]
        data_type_2 = ["", "_neg", "_conj", "_disj"]
        
        data_types = [f"{t1}{t2}" for t1 in data_type_1 for t2 in data_type_2]
        
        train_number = self.args.train_number
        if train_number < 0 or train_number >= len(data_type_2):
            raise ValueError("train_number must be between 0 and {}".format(len(data_type_2) - 1))
        
        train_data = []
        for t1 in data_type_1:
            train_data.extend(get_data(self.args.data_path, f"{t1}{data_type_2[train_number]}"))
        print(f"Train data loaded for: all data with suffix '{data_type_2[train_number]}'")
        
        train_size = int(0.8 * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = random_split(train_data, [train_size, val_size])
        
        eval_data = {}
        for i, suffix in enumerate(data_type_2):
            if i != train_number:
                eval_data_type = []
                for t1 in data_type_1:
                    eval_data_type.extend(get_data(self.args.data_path, f"{t1}{suffix}"))
                eval_data[suffix] = eval_data_type
                print(f"Eval data loaded for: all data with suffix '{suffix}'")


        train_dataset = TrainDataset(train_data, self.args)
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      num_workers=4)
        
        val_dataset = TrainDataset(val_data, self.args)
        val_dataloader = DataLoader(dataset=val_dataset,
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      num_workers=4)

        eval_datasets = {dt: TrainDataset(data, self.args, typ="valid") for dt, data in eval_data.items()}
        eval_dataloaders = {dt: DataLoader(dataset=dataset,
                                           batch_size=self.args.batch_size // 2,
                                           shuffle=False,
                                           num_workers=4) for dt, dataset in eval_datasets.items()}

        nSamples = [len(train_dataset) - train_dataset.halu_num, train_dataset.halu_num]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = torch.FloatTensor(normedWeights).to(self.args.device)
        loss_func = nn.CrossEntropyLoss(weight=normedWeights).to(self.args.device)
                
        best_val_acc = 0.0
        best_model_state = None

        for ei in range(epoch_start, epoch + 1):
            self.model.train()
            train_loss = 0
            predy, trainy, hallu_sm_score = [], [], []
            for step, batch in enumerate(train_dataloader):
                input_ = batch["input"].to(self.args.device)
                label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
                score = self.model(input_)
                _, pred = torch.max(score, dim=1)

                trainy.extend(label_ids.tolist())
                predy.extend(pred.tolist())
                loss = loss_func(score, label_ids)
                train_loss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()

            train_acc = binary_eval(predy, trainy)
            print("Train Epoch {} end! Loss: {}; Train Acc: {}".format(ei, train_loss, train_acc))
            
            self.model.eval()
            val_preds, val_labels = [], []
            for step, batch in enumerate(val_dataloader):
                input_ = batch["input"].to(self.args.device)
                label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
                score = self.model(input_)
                _, pred = torch.max(score, dim=1)
                val_labels.extend(label_ids.tolist())
                val_preds.extend(pred.tolist())

            val_acc = binary_eval(val_preds, val_labels)
            print(f"Validation Accuracy: {val_acc:.4f}")
             
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_ei = ei


        print(f"Best_val_acc: {best_val_acc:.4f}")
        self.model.load_state_dict(best_model_state)
        self.model.eval()
        self.model.to(args.device)
        with torch.no_grad():
            for eval_name, eval_loader in eval_dataloaders.items():
                predy, validy = [], []
                for step, batch in enumerate(eval_loader):
                    input_ = batch["input"].to(self.args.device)
                    label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
                    score = self.model(input_)
                    _, pred = torch.max(score, dim=1)
                    validy.extend(label_ids.tolist())
                    predy.extend(pred.tolist())

                eval_acc = binary_eval(predy, validy)
                print("Eval on {} - Epoch {}: Accuracy: {}".format(eval_name, best_ei, eval_acc))


        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup_seed", default=0, type=int)
    parser.add_argument("--output_path", default="../../output/", type=str)
    parser.add_argument("--data_path", default="../../hd_data_prompt/logic/llama2chat7b/", type=str)
    parser.add_argument("--train_number", default=0, type=int)

    parser.add_argument("--input_size", default=4096, type=int)
    parser.add_argument("--train_epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)
    
    args = parser.parse_args()    
    
    setup_seed(args.setup_seed)
    
    model = Model(args)
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optim_func = torch.optim.Adam
    named_params = list(model.model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd, 'lr': args.lr},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr}
    ]
    optimizer = optim_func(optimizer_grouped_parameters)

    model.run(optimizer)
