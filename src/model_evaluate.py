import torch
from models import EOGModel, Linear_NN, OneD_Conv, Rec_NN
from preprocess_data import preprocess_dataset, create_dataloader

from collections import deque
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256


def train_epoch(model : EOGModel, dataloader : torch.utils.data.DataLoader, optimizer : torch.optim.Optimizer, weight_classes : list = None, verbose = False) -> float:
    """Trains the model on the dataset

    Args:
        model (EOGModel): The model to be trained.
        dataloader (torch.utils.data.DataLoader): The dataset.
        optimizer (torch.optim.Optimizer): The optimizer for the given model

    Returns:
        float: The accuraccy of the model
    """
    
    __device = next(model.parameters()).device
    model.train()
    correct, total, acc = 0,0, 0
    loss_window = deque()
    loss_window.extend([0]*5)


    if weight_classes is not None:
        loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight_classes, dtype=torch.float32))
    else:
        loss_function = torch.nn.CrossEntropyLoss()
    
    for itr, (datas,labels) in enumerate(dataloader):
        if verbose:
            print("\rTraining. Batch {}/{}. Running Acc: {:.2f}%     Loss window: {:.4f}".format(itr,len(dataloader), acc, sum(list(loss_window))/len(loss_window)), end = "")
        optimizer.zero_grad()

        # Predict
        output = model(datas.to(__device)).cpu()
        
        # Metrics
        _, pred = output.max(1, keepdims=True)
        correct += pred.eq(labels).sum().item()
        total += len(datas)
        acc = correct / total * 100
        
        # Update
        loss = loss_function(output, labels.squeeze(-1))
        
        loss_window.append(loss)
        loss_window.popleft()
        
        
        loss.backward()
        optimizer.step()
        
    if verbose:
        print("Training. Batch {}/{}. Running Acc: {:.2f}%".format(itr + 1,len(dataloader), acc))
        
    return acc

def test_epoch(model : EOGModel, dataloader : torch.utils.data.DataLoader, verbose = False) -> float : 
    """Evaluates the model on the dataset 

    Args:
        model (EOGModel): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The dataset.

    Returns:
        float: The accuraccy of the model
    """
    __device = next(model.parameters()).device
    
    model.eval()
    correct, total, acc = 0,0, 0
   

    for itr, (datas,labels) in enumerate(dataloader):
        if verbose:
            print("\rTesting. Batch {}/{}. Running Acc: {:.2f}%".format(itr,len(dataloader), acc), end = "")
       
        # Predict
        output = model(datas.to(__device)).cpu()
        
        # Metrics
        _, pred = output.max(1, keepdims=True)
        correct += pred.eq(labels).sum().item()
        total += len(datas)
        acc = correct / total * 100
        
    if verbose:
        print("\rTesting. Batch {}/{}. Running Acc: {:.2f}%".format(itr + 1,len(dataloader), acc))
        
    return acc

    


def evaluate_model_cross(model : EOGModel, num_epochs : int, dfs : dict, optim_kwargs = {"lr" : 1e-2, "weight_decay" : 1e-5}):
    """Evaluates a model, by training/testing in a "leave-one-participant-out" fashion

    Args:
        model (EOGModel)
    """

    # Iterate over all candidates to leave out
    overall_acc_train = []
    overall_acc_test  = []
    for leave_out in range(1,11,1) :
        print("\rPerson {} is left out".format(leave_out), end = "", flush=True)
        model.reset_parameters()
        
        keys_test  = ["p{}_d1".format(leave_out),"p{}_d2".format(leave_out)]
        keys_train = [k  for k in dfs.keys() if k not in keys_test]
                 
        loader_train, weights = create_dataloader({key : dfs[key] for key in keys_train},BATCH_SIZE,return_weights = True)
        loader_test  = create_dataloader({key : dfs[key] for key in keys_test} ,BATCH_SIZE, undersampling=True)
        
        optim = torch.optim.Adam(model.parameters(),**optim_kwargs)
        
        for epoch in range(num_epochs):
            acc_train = train_epoch(model,loader_train,optim, weight_classes=weights, verbose = False)
        acc_test = test_epoch(model,loader_test, verbose = False)   
        
        overall_acc_train.append(acc_train)
        overall_acc_test .append(acc_test)

    print("\rModel {} Acc Mean:  Train: {:.2f}    Test: {:.2f}".format(model.description(), sum(overall_acc_train)/len(overall_acc_train), sum(overall_acc_test)/len(overall_acc_test) ), flush=True)
    return (overall_acc_train, overall_acc_test)

def show_model_performances_cross(models : list, dfs : dict, name:str):
    """Wraps the evaluate_model function to pass it multiple models at once and produce a nice plot from it

    Args:
        models (list): list of models
    """
    
    
    logging = pd.DataFrame(columns= ["model", "acc", "set"])
    
    try:
        # Is empty
        LOG_FILE = open("res/logs/cross_validation_{}.pickle".format(name), "xb")
        LOG_SAVES = {}
        pickle.dump(LOG_SAVES, LOG_FILE)
        LOG_FILE.close()
    except FileExistsError:
        with open("res/logs/cross_validation_{}.pickle".format(name), "rb") as LOG_FILE:
            LOG_SAVES = pickle.load(LOG_FILE)   
        
    for mod in models:

        if mod.description() in LOG_SAVES:
            overall_acc_train, overall_acc_test = LOG_SAVES[mod.description()]
                
        else:
            overall_acc_train, overall_acc_test = evaluate_model_cross(mod,5,dfs)
            LOG_SAVES[mod.description()] = (overall_acc_train, overall_acc_test )
            with open("res/logs/cross_validation_{}.pickle".format(name), "wb") as LOG_FILE:
                pickle.dump(LOG_SAVES, LOG_FILE)
        logging = pd.concat([logging, 
        
            pd.DataFrame({
                "model" : [mod.description() for _ in overall_acc_train],
                "acc" : overall_acc_train,
                "set" : ["Train" for _ in overall_acc_train]
            }),
            pd.DataFrame({
                "model" : [mod.description() for _ in overall_acc_test],
                "acc" : overall_acc_test,
                "set" : ["Test" for _ in overall_acc_test]
            })
        ])
    
    sns.boxplot(data=logging, x="acc", y="model", hue="set")
    #sns.stripplot(data=logging, x="acc", y="model", hue="set")
    sns.despine()
    
    plt.grid(axis = "x", alpha = 0.6)
    plt.grid(axis = "x", alpha = 0.4, which ="minor")
    ax = plt.gca()
    start, end = ax.get_xlim()
    ax.set_xticks(np.arange(int(np.floor(start)), int(np.ceil(end)), 1), minor = True)
    ax.set_title("leave-one-participant-out performance for {} models".format(name))
    
    plt.tight_layout()
    plt.savefig("res/{}_cross_performance.png".format(name))
    #plt.savefig("res/{}_cross_performance.pdf".format(name))
    plt.close()
    

def model_in_person_accuracy(model : EOGModel, num_epochs, dfs : dict, optim_kwargs = {"lr" : 1e-2, "weight_decay" : 1e-5}):
    """Evaluates a model, by training/testing on one day of a person and predicting the other day.
    Per person has to experiments: d1→d2 and d2→d1

    Args:
        model (EOGModel)
    """    
    # Iterate over all candidates 
    overall_acc_train = []
    overall_acc_test  = []
    for person in range(1,11,1) :
        print("\rPerson {}".format(person), end = "")
       
        loader_d1_train, weights_d1  = create_dataloader({"p{}_d1".format(person) : dfs["p{}_d1".format(person)]},BATCH_SIZE, return_weights=True)
        loader_d1_test   = create_dataloader({"p{}_d1".format(person) : dfs["p{}_d1".format(person)]},BATCH_SIZE, undersampling=True)
        loader_d2_train, weights_d2  = create_dataloader({"p{}_d2".format(person) : dfs["p{}_d2".format(person)]},BATCH_SIZE, return_weights=True)
        loader_d2_test   = create_dataloader({"p{}_d2".format(person) : dfs["p{}_d2".format(person)]},BATCH_SIZE, undersampling=True)
        
        
        # Day1 -> Day2
        
        model.reset_parameters()
        optim = torch.optim.Adam(model.parameters(),**optim_kwargs)
        
        for epoch in range(num_epochs):
            acc_train = train_epoch(model,loader_d1_train,optim, weight_classes=weights_d1, verbose = False)
        acc_test = test_epoch(model,loader_d2_test, verbose = False)   
        
        overall_acc_train.append(acc_train)
        overall_acc_test .append(acc_test)
        
        # Day1 -> Day2
        
        model.reset_parameters()
        optim = torch.optim.Adam(model.parameters(),**optim_kwargs)
        
        for epoch in range(num_epochs):
            acc_train = train_epoch(model,loader_d2_train,optim, weight_classes=weights_d2, verbose = False)
        acc_test = test_epoch(model,loader_d1_test, verbose = False)   
        
        overall_acc_train.append(acc_train)
        overall_acc_test .append(acc_test)
        

    print("\rModel {} Acc Mean:  Train: {:.2f}    Test: {:.2f}".format(model.description(), sum(overall_acc_train)/len(overall_acc_train), sum(overall_acc_test)/len(overall_acc_test) ))
    return (overall_acc_train, overall_acc_test)

def show_model_performances_in_person(models : list, dfs : dict, name:str):
    """Wraps the model_in_person_accuracy function to pass it multiple models at once and produce a nice plot from it

    Args:
        models (list): list of models
    """
    
    logging = pd.DataFrame(columns= ["model", "acc", "set"])
    
    try:
        # Is empty
        LOG_FILE = open("res/logs/in_person_{}.pickle".format(name), "xb")
        LOG_SAVES = {}
        pickle.dump(LOG_SAVES, LOG_FILE)
        LOG_FILE.close()
    except FileExistsError:
        with open("res/logs/in_person_{}.pickle".format(name), "rb") as LOG_FILE:
            LOG_SAVES = pickle.load(LOG_FILE)   
        
    for mod in models:

        if mod.description() in LOG_SAVES:
            overall_acc_train, overall_acc_test = LOG_SAVES[mod.description()]
                
        else:
            overall_acc_train, overall_acc_test = model_in_person_accuracy(mod,5,dfs)
            LOG_SAVES[mod.description()] = (overall_acc_train, overall_acc_test )
            with open("res/logs/in_person_{}.pickle".format(name), "wb") as LOG_FILE:
                pickle.dump(LOG_SAVES, LOG_FILE)
                
        logging = pd.concat([logging, 
        
            pd.DataFrame({
                "model" : [mod.description() for _ in overall_acc_train],
                "acc" : overall_acc_train,
                "set" : ["Train" for _ in overall_acc_train]
            }),
            pd.DataFrame({
                "model" : [mod.description() for _ in overall_acc_test],
                "acc" : overall_acc_test,
                "set" : ["Test" for _ in overall_acc_test]
            })
        ])
    
    sns.boxplot(data=logging, x="acc", y="model", hue="set")
    #sns.stripplot(data=logging, x="acc", y="model", hue="set")
    sns.despine()
    
    plt.grid(axis = "x", alpha = 0.6)
    plt.grid(axis = "x", alpha = 0.4, which ="minor")
    ax = plt.gca()
    start, end = ax.get_xlim()
    ax.set_xticks(np.arange(int(np.floor(start)), int(np.ceil(end)), 1), minor = True)
    
    ax.set_title("leave-one-day-out performance for {} models".format(name))
    
    plt.tight_layout()
    plt.savefig("res/{}_in_person_performance.png".format(name))
    #plt.savefig("res/{}_in_person_performance.pdf".format(name))
    plt.close()
    
    
def model_calibration_accuracy(model : EOGModel, num_epochs, dfs : dict, optim_kwargs = {"lr" : 1e-2, "weight_decay" : 1e-5}):
    """Evaluates a model, by training it on the entire dataset except one. This includes also the other day by the same person. 
    Per person has to experiments: d1→d2 and d2→d1

    Args:
        model (EOGModel)
    """
    
    # Iterate over all candidates 
    overall_acc_train = []
    overall_acc_test  = []
    for person in range(1,11,1) :
        print("\rPerson {}".format(person), end = "")
       
        loader_d1_test  = create_dataloader({"p{}_d1".format(person) : dfs["p{}_d1".format(person)]},BATCH_SIZE, undersampling=True)
        loader_d2_test  = create_dataloader({"p{}_d2".format(person) : dfs["p{}_d2".format(person)]},BATCH_SIZE, undersampling=True)
        
        keys_not_d1 = [k  for k in dfs.keys() if k != "p{}_d1"]
        keys_not_d2 = [k  for k in dfs.keys() if k != "p{}_d2"]
                
        loader_not_d1_train, weights_not_d1 = create_dataloader({key : dfs[key] for key in keys_not_d1},BATCH_SIZE, return_weights= True)
        loader_not_d2_train, weights_not_d2 = create_dataloader({key : dfs[key] for key in keys_not_d2},BATCH_SIZE, return_weights= True)
        

        # Day1 -> Day2
        
        model.reset_parameters()
        optim = torch.optim.Adam(model.parameters(),**optim_kwargs)
        
        for epoch in range(num_epochs):
            acc_train = train_epoch(model, loader_not_d2_train,optim, weight_classes=weights_not_d2, verbose = False)
        acc_test = test_epoch(model,loader_d2_test, verbose = False)   
        
        overall_acc_train.append(acc_train)
        overall_acc_test .append(acc_test)
        
        # Day1 -> Day2
        
        model.reset_parameters()
        optim = torch.optim.Adam(model.parameters(),**optim_kwargs)
        
        for epoch in range(num_epochs):
            acc_train = train_epoch(model,loader_not_d1_train,optim, weight_classes=weights_not_d1, verbose = False)
        acc_test = test_epoch(model,loader_d1_test, verbose = False)   
        
        overall_acc_train.append(acc_train)
        overall_acc_test .append(acc_test)
        

    print("\rModel {} Acc Mean:  Train: {:.2f}    Test: {:.2f}".format(model.description(), sum(overall_acc_train)/len(overall_acc_train), sum(overall_acc_test)/len(overall_acc_test) ))
    return (overall_acc_train, overall_acc_test)

def show_model_calibration(models : list, dfs: dict, name:str):
    """Wraps the model_in_person_accuracy function to pass it multiple models at once and produce a nice plot from it

    Args:
        models (list): list of models
    """
    
    logging = pd.DataFrame(columns= ["model", "acc", "set"])
    
    try:
        # Is empty
        LOG_FILE = open("res/logs/in_person_{}.pickle".format(name), "xb")
        LOG_SAVES = {}
        pickle.dump(LOG_SAVES, LOG_FILE)
        LOG_FILE.close()
    except FileExistsError:
        with open("res/logs/in_person_{}.pickle".format(name), "rb") as LOG_FILE:
            LOG_SAVES = pickle.load(LOG_FILE)   
        
    for mod in models:

        if mod.description() in LOG_SAVES:
            overall_acc_train, overall_acc_test = LOG_SAVES[mod.description()]
                
        else:
            overall_acc_train, overall_acc_test = model_calibration_accuracy(mod,5,dfs)
            LOG_SAVES[mod.description()] = (overall_acc_train, overall_acc_test )
            with open("res/logs/in_person_{}.pickle".format(name), "wb") as LOG_FILE:
                pickle.dump(LOG_SAVES, LOG_FILE)
        logging = pd.concat([logging, 
        
            pd.DataFrame({
                "model" : [mod.description() for _ in overall_acc_train],
                "acc" : overall_acc_train,
                "set" : ["Train" for _ in overall_acc_train]
            }),
            pd.DataFrame({
                "model" : [mod.description() for _ in overall_acc_test],
                "acc" : overall_acc_test,
                "set" : ["Test" for _ in overall_acc_test]
            })
        ])
    
    sns.boxplot(data=logging, x="acc", y="model", hue="set")
    #sns.stripplot(data=logging, x="acc", y="model", hue="set")
    sns.despine()
    
    plt.grid(axis = "x", alpha = 0.6)
    plt.grid(axis = "x", alpha = 0.4, which ="minor")
    ax = plt.gca()
    start, end = ax.get_xlim()
    ax.set_xticks(np.arange(int(np.floor(start)), int(np.ceil(end)), 1), minor = True)
    ax.set_title("train-and-calibrate performance for {} models".format(name))
    
    plt.tight_layout()
    plt.savefig("res/{}_calibration_performance.png".format(name))
    #plt.savefig("res/{}_calibration_performance.pdf".format(name))
    plt.close()





def main(models, data, name):
    # Models are Linear, CNN and LSTM
    # 
    # Data is normal or PCA_X
    #
    if models == "linear":
        
        if data == "cond_normal":
            _data = preprocess_dataset("condensed")
            in_features = 16
        if data == "cond_PC1":
            _data = preprocess_dataset("condensed", pca=1)
            in_features = 2
        if data == "cond_PC2":
            _data = preprocess_dataset("condensed", pca=2)
            in_features = 4
        if data == "cond_PC3":
            _data = preprocess_dataset("condensed", pca=3)
            in_features = 6
        if data == "cond_PC5":
            _data = preprocess_dataset("condensed", pca=5)
            in_features = 10
            
            
        if data == "concat_normal":
            _data = preprocess_dataset("concat")
            in_features = 8 * 3000
        if data == "concat_PC1":
            _data = preprocess_dataset("concat", pca=1)
            in_features = 1 * 3000
        if data == "concat_PC2":
            _data = preprocess_dataset("concat", pca=2)
            in_features = 2 * 3000
        if data == "concat_PC3":
            _data = preprocess_dataset("concat", pca=3)
            in_features = 3 * 3000
        if data == "concat_PC5":
            _data = preprocess_dataset("concat", pca=5)
            in_features = 5 * 3000
            
        _models = [
        Linear_NN(in_features=in_features, num_hidden=[]).to(device),
        Linear_NN(in_features=in_features, num_hidden=[10]).to(device),
        Linear_NN(in_features=in_features, num_hidden=[20]).to(device),
        Linear_NN(in_features=in_features, num_hidden=[40]).to(device),
        Linear_NN(in_features=in_features, num_hidden=[10,10]).to(device),
        Linear_NN(in_features=in_features, num_hidden=[20,20]).to(device),
        Linear_NN(in_features=in_features, num_hidden=[40,40]).to(device),
        Linear_NN(in_features=in_features, num_hidden=[40,40,40]).to(device),
        Linear_NN(in_features=in_features, num_hidden=[40,40,40,40]).to(device),
    ]
        
        
        
    else:   

        if data == "concat_normal":
            _data = preprocess_dataset("concat")
            in_features = 8
        if data == "concat_PC1":
            _data = preprocess_dataset("concat", pca=1)
            in_features = 1
        if data == "concat_PC2":
            _data = preprocess_dataset("concat", pca=2)
            in_features = 2
        if data == "concat_PC3":
            _data = preprocess_dataset("concat", pca=3)
            in_features = 3
        if data == "concat_PC5":
            _data = preprocess_dataset("concat", pca=5)
            in_features = 5
        
        if models == "CNN":  
            _models = [
            OneD_Conv(in_features = in_features, num_hidden = [[10],[140]]).to(device),
            OneD_Conv(in_features = in_features, num_hidden = [[10],[140,40]]).to(device),
            OneD_Conv(in_features = in_features, num_hidden = [[10],[140,40,40]]).to(device),
            OneD_Conv(in_features = in_features, num_hidden = [[20],[280]]).to(device),
            OneD_Conv(in_features = in_features, num_hidden = [[20],[280,40]]).to(device),
            OneD_Conv(in_features = in_features, num_hidden = [[20],[280,40,40]]).to(device),
            OneD_Conv(in_features = in_features, num_hidden = [[40],[560]]).to(device),
            OneD_Conv(in_features = in_features, num_hidden = [[40],[560,40]]).to(device),
            OneD_Conv(in_features = in_features, num_hidden = [[40],[560,40,40]]).to(device),
        ]
        
        if models == "LSTM":
            _models = [
            Rec_NN(in_features = in_features, num_hidden = [(20,1,False),[]]).to(device),
            Rec_NN(in_features = in_features, num_hidden = [(20,1,False),[20]]).to(device),
            Rec_NN(in_features = in_features, num_hidden = [(20,1,False),[40]]).to(device),
            Rec_NN(in_features = in_features, num_hidden = [(20,1,False),[40,40]]).to(device),
            Rec_NN(in_features = in_features, num_hidden = [(20,1,False),[40,40,40]]).to(device),
            Rec_NN(in_features = in_features, num_hidden = [(40,1,False),[40,40,40]]).to(device),
            Rec_NN(in_features = in_features, num_hidden = [(40,2,False),[40,40,40]]).to(device),
            Rec_NN(in_features = in_features, num_hidden = [(40,3,False),[40,40,40]]).to(device),
            Rec_NN(in_features = in_features, num_hidden = [(40,1,True),[40,40,40]]).to(device),
            Rec_NN(in_features = in_features, num_hidden = [(40,2,True),[40,40,40]]).to(device),
            Rec_NN(in_features = in_features, num_hidden = [(40,3,True),[40,40,40]]).to(device),
        ]
        
    
    print("\nCross-Validation\n", flush=True)
    show_model_performances_cross(_models, _data, name)
    print("\In-Person\n")
    show_model_performances_in_person(_models, _data, name)
    print("\nCalibration\n")
    show_model_calibration(_models, _data, name)
    

if __name__ == "__main__":
    import argparse

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-v", "--version", type=int)

    try:
        v = argParser.parse_args().version
        
    except:
        v = -1
    print("Version = " + str(v))

    if v == 1:
        main("linear", "cond_normal",   "Condensed_Linear")
    if v == 2:
        main("linear", "cond_PC1",      "Condensed_Linear_PC1")
    if v == 3:      
        main("linear", "cond_PC2",      "Condensed_Linear_PC2")
    if v == 4:      
        main("linear", "cond_PC3",      "Condensed_Linear_PC3")
    if v == 5:      
        main("linear", "cond_PC5",      "Condensed_Linear_PC5")
        
    if v == 11:
        main("linear", "concat_normal",   "Concatenated_Linear")
    if v == 12:
        main("linear", "concat_PC1",      "Concatenated_Linear_PC1")
    if v == 13:      
        main("linear", "concat_PC2",      "Concatenated_Linear_PC2")
    if v == 14:      
        main("linear", "concat_PC3",      "Concatenated_Linear_PC3")
    if v == 15:      
        main("linear", "concat_PC5",      "Concatenated_Linear_PC5")
        
    if v == 21:      
        main("CNN", "concat_normal",    "Concatenated_CNN")
    if v == 22:      
        main("CNN", "concat_PC1",      "Concatenated_CNN_PC1")
    if v == 23:      
        main("CNN", "concat_PC2",      "Concatenated_CNN_PC2")
    if v == 24:      
        main("CNN", "concat_PC3",      "Concatenated_CNN_PC3")
    if v == 25:      
        main("CNN", "concat_PC5",      "Concatenated_CNN_PC5")
        
    if v == 31:      
        main("LSTM", "concat_normal",    "Concatenated_LSTM")
    if v == 32:      
        main("LSTM", "concat_PC1",    "Concatenated_LSTM_PC1")
    if v == 33:      
        main("LSTM", "concat_PC2",    "Concatenated_LSTM_PC2")
    if v == 34:      
        main("LSTM", "concat_PC3",    "Concatenated_LSTM_PC3")
    if v == 35:      
        main("LSTM", "concat_PC5",    "Concatenated_LSTM_PC5")
