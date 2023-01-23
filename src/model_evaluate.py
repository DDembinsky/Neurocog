import torch
from models import EOGModel, Linear_NN
from preprocess_data import preprocess_dataset, create_dataloader

from collections import deque
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
import numpy as np
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
            print("Training. Batch {}/{}. Running Acc: {:.2f}%     Loss window: {:.4f}".format(itr,len(dataloader), acc, sum(list(loss_window))/len(loss_window)), end = "\r")
        optimizer.zero_grad()

        # Predict
        output = model(datas.to(__device))
        
        # Metrics
        _, pred = output.cpu().max(1, keepdims=True)
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
    model.eval()
    correct, total, acc = 0,0, 0
   

    for itr, (datas,labels) in enumerate(dataloader):
        if verbose:
            print("Testing. Batch {}/{}. Running Acc: {:.2f}%".format(itr,len(dataloader), acc), end = "\r")
       
        # Predict
        output = model(datas)
        
        # Metrics
        _, pred = output.cpu().max(1, keepdims=True)
        correct += pred.eq(labels).sum().item()
        total += len(datas)
        acc = correct / total * 100
        
    if verbose:
        print("Testing. Batch {}/{}. Running Acc: {:.2f}%".format(itr + 1,len(dataloader), acc))
        
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
        print("Person {} is left out".format(leave_out), end = "\r")
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

    print("Model {} Acc Mean:  Train: {:.2f}    Test: {:.2f}".format(model.description(), sum(overall_acc_train)/len(overall_acc_train), sum(overall_acc_test)/len(overall_acc_test) ))
    return (overall_acc_train, overall_acc_test)

def show_model_performances_cross(models : list, dfs : dict, name:str):
    """Wraps the evaluate_model function to pass it multiple models at once and produce a nice plot from it

    Args:
        models (list): list of models
    """
    LOG_SAVES = {
          }
    
    logging = pd.DataFrame(columns= ["model", "acc", "set"])
    
    for mod in models:
        if mod.description() in LOG_SAVES:
            overall_acc_train, overall_acc_test = LOG_SAVES[mod.description()]
            
        else:
            overall_acc_train, overall_acc_test = evaluate_model_cross(mod,5,dfs)
            print('"{}" : ({},{}),'.format(mod.description(), overall_acc_train, overall_acc_test) )
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
    sns.despine(trim = True)
    
    plt.grid(axis = "x", alpha = 0.6)
    plt.grid(axis = "x", alpha = 0.4, which ="minor")
    ax = plt.gca()
    start, end = ax.get_xlim()
    ax.set_xticks(np.arange(int(np.floor(start)), int(np.ceil(end)), 1), minor = True)
    
    ri = randint(100000,999999)
    plt.savefig("res/cross_performance_{}_{}.png".format(name,ri))
    plt.savefig("res/cross_performance_{}_{}.pdf".format(name,ri))
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
        print("Person {}".format(person), end = "\r")
       
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
        

    print("Model {} Acc Mean:  Train: {:.2f}    Test: {:.2f}".format(model.description(), sum(overall_acc_train)/len(overall_acc_train), sum(overall_acc_test)/len(overall_acc_test) ))
    return (overall_acc_train, overall_acc_test)

def show_model_performances_in_person(models : list, dfs : dict, name:str):
    """Wraps the model_in_person_accuracy function to pass it multiple models at once and produce a nice plot from it

    Args:
        models (list): list of models
    """
    LOG_SAVES = {
        }
    
    logging = pd.DataFrame(columns= ["model", "acc", "set"])
    
    for mod in models:
        
        if mod.description() in LOG_SAVES:
            overall_acc_train, overall_acc_test = LOG_SAVES[mod.description()]
            
        else:
            overall_acc_train, overall_acc_test = model_in_person_accuracy(mod,5,dfs)
            print('"{}" : ({},{}),'.format(mod.description(), overall_acc_train, overall_acc_test) )
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
    sns.despine(trim = True)
    
    plt.grid(axis = "x", alpha = 0.6)
    plt.grid(axis = "x", alpha = 0.4, which ="minor")
    ax = plt.gca()
    start, end = ax.get_xlim()
    ax.set_xticks(np.arange(int(np.floor(start)), int(np.ceil(end)), 1), minor = True)
    
    
    ri = randint(100000,999999)
    plt.savefig("res/in_person_performance_{}_{}.png".format(name,ri))
    plt.savefig("res/in_person_performance_{}_{}.pdf".format(name,ri))
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
        print("Person {}".format(person), end = "\r")
       
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
        

    print("Model {} Acc Mean:  Train: {:.2f}    Test: {:.2f}".format(model.description(), sum(overall_acc_train)/len(overall_acc_train), sum(overall_acc_test)/len(overall_acc_test) ))
    return (overall_acc_train, overall_acc_test)

def show_model_calibration(models : list, dfs: dict, name:str):
    """Wraps the model_in_person_accuracy function to pass it multiple models at once and produce a nice plot from it

    Args:
        models (list): list of models
    """
    LOG_SAVES = {
      }
    
    logging = pd.DataFrame(columns= ["model", "acc", "set"])
    
    for mod in models:
        if mod.description() in LOG_SAVES:
            overall_acc_train, overall_acc_test = LOG_SAVES[mod.description()]
            
        else:
            overall_acc_train, overall_acc_test = model_in_person_accuracy(mod,5,dfs)
            print('"{}" : ({},{}),'.format(mod.description(), overall_acc_train, overall_acc_test) )
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
    sns.despine(trim = True)
    
    plt.grid(axis = "x", alpha = 0.6)
    plt.grid(axis = "x", alpha = 0.4, which ="minor")
    ax = plt.gca()
    start, end = ax.get_xlim()
    ax.set_xticks(np.arange(int(np.floor(start)), int(np.ceil(end)), 1), minor = True)
    
    ri = randint(100000,999999)
    plt.savefig("res/calibration_performance_{}_{}.png".format(name,ri))
    plt.savefig("res/calibration_performance_{}_{}.pdf".format(name,ri))
    plt.close()





def main(models, data):
    # Models are Linear, CNN and LSTM
    # 
    # Data is normal or PCA_X
    #
    if models == "linear":
        _models = [
        Linear_NN(num_hidden=[]),
        Linear_NN(num_hidden=[10]),
        Linear_NN(num_hidden=[20]),
        Linear_NN(num_hidden=[40]),
        Linear_NN(num_hidden=[10,10]),
        Linear_NN(num_hidden=[20,20]),
        Linear_NN(num_hidden=[40,40]),
    ]
    
        if data == "normal":
            _data = preprocess_dataset("condensed")
                
    show_model_performances_cross(_models, _data, name="lin_normal")
    show_model_performances_in_person(_models, _data, name="lin_normal")
    show_model_calibration(_models, _data, name="lin_normal")
    

if __name__ == "__main__":
    main("linear", "normal")