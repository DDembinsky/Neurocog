from models import Linear_NN
from model_evaluate import train_epoch
from preprocess_data import preprocess_dataset, create_dataloader

import torch
import pandas as pd
import time
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
optim_kwargs = {"lr" : 1e-2, "weight_decay" : 1e-5}
num_epochs = 15

module_hidden = [
     {"num_hidden" : []},
     {"num_hidden" : [10]},
     {"num_hidden" : [20]},
     {"num_hidden" : [40]},
     {"num_hidden" : [10,10]},
     {"num_hidden" : [20,20]},
     {"num_hidden" : [40,40]},
     {"num_hidden" : [40,40,40]},
     {"num_hidden" : [40,40,40,40]},
    ]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


df = pd.DataFrame(columns = ["name", "num_hidden", "dataset", "num_weights", "batch_time"])

data_condensed = preprocess_dataset("condensed")
data_concat = preprocess_dataset("concat")

loader_condensed, weights_condensed = create_dataloader(data_condensed,BATCH_SIZE,return_weights = True)
loader_concat, weights_concat = create_dataloader(data_concat,BATCH_SIZE,return_weights = True)


for mod_hidden in module_hidden:

    mod_condensed = Linear_NN(in_features=16,     **mod_hidden).to(device)
    #mod_concat = Linear_NN(in_features=8*3000, **mod_hidden).to(device),

    optim_condensed = torch.optim.Adam(mod_condensed.parameters(),**optim_kwargs)
    #optim_concat = torch.optim.Adam(mod_concat.parameters(),**optim_kwargs)

    start = time.time()
    # Condensed
    for epoch in range(num_epochs):
        _ = train_epoch(mod_condensed,loader_condensed,optim_condensed, weight_classes=weights_condensed, verbose = False)
    
    end = time.time()
    diff = (end-start)/num_epochs
    

    df = pd.concat([df, pd.DataFrame({"name": [mod_condensed.description()], "num_hidden" : [mod_hidden], "dataset" : ["Condensed"], "num_weights" : [count_parameters(mod_condensed)], "batch_time" : [diff]})])

for mod_hidden in [
     {"num_hidden" : []},
     {"num_hidden" : [10]},
     {"num_hidden" : [20]},
     {"num_hidden" : [40]},
     {"num_hidden" : [10,10]},
     {"num_hidden" : [20,20]},
     {"num_hidden" : [40,40]},
     {"num_hidden" : [40,40,40]},
     {"num_hidden" : [40,40,40,40]},
    ]:
    
#    mod_condensed = Linear_NN(in_features=16,     **mod_hidden).to(device),
    mod_concat = Linear_NN(in_features=8*3000, **mod_hidden).to(device)
    

#    optim_condensed = torch.optim.Adam(mod_condensed.parameters(),**optim_kwargs)
    optim_concat = torch.optim.Adam(mod_concat.parameters(),**optim_kwargs)

    start = time.time()   
    
    # Concatenated
    for epoch in range(num_epochs):
        _ = train_epoch(mod_concat,loader_concat,optim_concat, weight_classes=weights_concat, verbose = False)
    
    end = time.time()
    diff = (end-start)/num_epochs
    
    df = pd.concat([df, pd.DataFrame({"name": [mod_concat.description()], "num_hidden" : [mod_hidden], "dataset" : ["Concatenated"], "num_weights" : [count_parameters(mod_concat)], "batch_time" : [diff]})])


with open("res/logs/model_performance.pickle", "wb") as LOG_FILE:
    pickle.dump(df, LOG_FILE)
    
