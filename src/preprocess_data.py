import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import IncrementalPCA
import torch
from torch.utils.data import DataLoader, TensorDataset

import os

def __form_window_condensed(df: pd.DataFrame , consecutive_steps: int, stride : int) -> pd.DataFrame:
    """Creates a new dataset of time windows by calculating meat-informaion of the window's features. 
    The columns are   (left | right | acc_x | acc_y | acc_z | roll | pitch | yaw) *(mean | std) + label
    The label is calculated as the median of the labels in the time frame.
    
    
    Args:
        df (pd.DataFrame): The original dataframe. Expects to have timestamp and datetime removed.
        consecutive_steps (int): The number of consecutive steps that form a window.
        stride (int): By how many  steps each sliding window is shifted
    """
    
    assert stride > 0
    
    if len(df.columns) == 9:
        __COLS = ["left mean", "right mean","acc_x mean","acc_y mean","acc_z mean","roll mean","pitch mean","yaw mean",
            "left std","right std","acc_x std", "acc_y std","acc_z std", "roll std", "pitch std","yaw std",  
            "label"]
    else:
        __COLS = ["PC{} mean".format(pc) for pc in range(len(df.columns) - 1)] + [
                    "PC{} std".format(pc) for pc in range(len(df.columns) - 1)] + ["label"]
    
    # Convert to NumPY
    df_n = df.to_numpy()
    
    # Create windows
    indices = np.array([[k for k in range(i,i+consecutive_steps,1)] for i in range(0,len(df_n) - consecutive_steps,stride)] )
    
    df_windowed = df_n[indices]
    
    # Calculate features
    df_win_mean = df_windowed[:,:,:-1].mean(axis=1)
    df_win_std  = df_windowed[:,:,:-1].std(axis=1)
    df_labels = np.median(df_windowed[:,:,-1] ,axis=1, keepdims=True).astype(int)
    
    
    # Create new df
    df_new = pd.DataFrame(data = np.concatenate([df_win_mean, df_win_std, df_labels], axis = 1),
                          columns= __COLS)

    return df_new

def __form_window_concat(df: pd.DataFrame , consecutive_steps: int, stride : int) -> pd.DataFrame:
    """Creates a new dataset of time windows by concatenating all features of the window.
    The new dataframe contains the columns (left | right | acc_x | acc_y | acc_z | roll | pitch | yaw) * #consecutive_steps + label
    The label is calculated as the median of the labels in the time frame.

    Args:
        df (pd.DataFrame): The original dataframe. Expects to have timestamp and datetime removed.
        consecutive_steps (int): The number of consecutive steps that form a window.
        stride (int): By how many  steps each sliding window is shifted.
    """
    assert stride > 1
    
    if len(df.columns) == 9:
        var = ['left', 'right', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw',]
    else:
        var = ["PC{}".format(pc) for pc in range(len(df.columns)-1)]
    
    __COLS = []
    for i in range(consecutive_steps):
        __COLS.extend(["{}_{}".format(v,i) for v in var])
    __COLS.append("label")


    # Convert to NumPY
    df_n = df.to_numpy()

    # Create windows
    indices = np.array([[k for k in range(i,i+consecutive_steps,1)] for i in range(0,len(df_n) - consecutive_steps,stride)] )
    

    df_windowed = df_n[indices]

    # Calculate features
    df_features =  df_windowed[:,:,:-1].reshape((-1,consecutive_steps*len(var)))
    
    df_labels = np.median(df_windowed[:,:,-1] ,axis=1, keepdims=True).astype(int)

    # Create new df
    df_new = pd.DataFrame(data = np.concatenate([df_features, df_labels], axis = 1),
                        columns= __COLS)
    
    return df_new


def __normalize_dataframe(df : pd.DataFrame) -> pd.DataFrame:
    """Normalizes the dataset per feature without changing the labels
    """
    cols_to_norm = df.columns.drop("label")

    df_new = df.copy()
    df_new[cols_to_norm] = (df[cols_to_norm]-df[cols_to_norm].mean())/df[cols_to_norm].std()
    return df_new   


def preprocess_dataset(method: str, consecutive_steps : int = 3000, stride : int = 1000, pca : int = None, persons : list = None, days : list = None ) -> dict:
    """Preprocesses all raw csv files

    Args:
        method (str): Either "condensed" or "concat", for the used method of forming windows.
        consecutive_steps (int): How many consecutive steps form a window.
        stride (int): By how many  steps each sliding window is shifted.
        pca (int): If set, PCA is eprformed and the firt x PCs are used to transform the data. If None, does not do PCA.
        persons (list): Specify which persons should be included in the dataset. If None, includes all.
        days (list): Specify which days should be included in the dataset. If None, includes all.
       
    """
    normalize = True
    
    func = __form_window_condensed if method == "condensed" else __form_window_concat if method == "concat" else None
    if func is None:
        raise ValueError("method has to be either 'condensed' or 'concat', but '''{}''' was given".format(method))
    
    dfs= {}

    
    if persons is None:
        persons = list(range(1,11,1))
    if days is None:
        days = list(range(1,3,1))
    
    # Load each raw dataset
    for p in persons:
        for d in days:
            file_name = "p{}_d{}".format(p,d)
            
            df = pd.read_csv("data/" + file_name + ".csv")
            
            # Remove unnecessary columns
            df = df.drop(labels=["#timestamp","date_time"], axis=1)
            
            
            
            if normalize:
                df = __normalize_dataframe(df)
            dfs[file_name] = df
            print(file_name + " loaded            ", end = "\r")
            
    if pca is not None:
        dfs = __pca_dataset(dfs, pca)   
    
    for p in persons:
        for d in days:
            file_name = "p{}_d{}".format(p,d)    
            df = dfs[file_name]
            df_new = func(df = df, consecutive_steps = consecutive_steps , stride = stride)
            dfs[file_name] = df_new
            
            print(file_name + " preprocessed         " , end = "\r")
            
            
   
    print("Finished dataset")
    return dfs      
            
def __pca_dataset(dfs : dict, num_pc : int = None) -> dict:
    """Performs PCA over the entire dataset and transform each individual dataset accordingly

    Args:
        dfs (dict): Dict of datasets as provided by ```preprocess_dataset```
        num_pc (int, optional): How many PCs to use, if None uses all. Defaults to None.

    Returns:
        dict: Dict with same keys as dfs but each dataset is transformed with PCA
    """

    pca = IncrementalPCA(n_components=num_pc)
    
    for k in dfs.keys():
        print("PCA fit  " + k, end = "\r")
        pca.partial_fit(dfs[k].to_numpy()[:,:-1])
        

    dfs_new = {}
    keys =  list(dfs.keys())
    for k in keys:
        np_buffer = np.concatenate([pca.transform(dfs[k].to_numpy()[:,:-1]) , dfs[k].to_numpy()[:,-1, None]], axis = 1)
        del dfs[k]
        dfs_new[k] = pd.DataFrame(np_buffer, columns=["PC{}".format(i+1) for i in range(pca.n_components_) ] + ["label"] )
        
    
    return dfs_new

def __balance_class_distribution(ds : np.ndarray, randomize : bool = False) -> np.ndarray:
    """Performs undersampling on the majority class, to get a balanced distirbution of the dataset
        Warning, dataset will be sorted by labels afterwards, if this is undesired, set ```randomize=True``` 
    Args:
        ds (np.ndarray): dataset
        randomize (bool): If True, randomizes the order of the samples, else they will be sorted

    """
    # Get amount of smallest class
    classes, classes_rep = np.unique(ds[:,-1], return_counts=True)
    min_class_rep = min(classes_rep)

    # We allow classes to have 10% more samples than minority class
    thresh = int(min_class_rep * 1.1)
    
    new_ds = np.empty([0,ds.shape[1]])
    for c,a in zip(classes, classes_rep):
        ds_h = ds[ds[:,-1] == c]
        
        if a > thresh:
            ind = np.random.choice(a,thresh,replace=False)
            ds_h = ds_h[ind]
        
        new_ds = np.concatenate([new_ds,ds_h])
    
    if randomize:
        np.random.shuffle(new_ds)
        
    return new_ds  
    

def create_dataloader(datasets : dict, batch_size : int, shuffle : bool = True, undersampling: bool = True, return_weights : bool = False) -> torch.utils.data.DataLoader:
    """Creates a single dataloader from all data provided in datasets

    Args:
        datasets (dict): A dictionary of datasets
        batch_size (int): The batch size forwarded to the data_loader
        shuffle (bool, optional): If the samples sould be reshuffled each epoch. Defaults to True.
        undersampling (bool, optional): Wether to perform undersampling to balance the class distribution. Defaults to True.
        return_weights (bool, optional): Additionally returns the weights of the individual classes. Defaults to False.
        
    Returns:
        torch.utils.data.DataLoader: The dataloader
    """
    # Get datasets
    
    datasets = list(datasets.values())
    
    # Transform to numpy
    datasets = [ds.to_numpy() if ds is pd.DataFrame else ds for ds in datasets]
    
    # Concat into single numpy matrix
    datasets = np.concatenate(datasets)
    
    if undersampling:
        datasets = __balance_class_distribution(datasets)
    
    # Split data from label
    data = torch.from_numpy(datasets[:,:-1].astype(np.float32))
    labels = torch.from_numpy(datasets[:,-1, None].astype(np.int64))
    

    # Construct TensorDataset
    dataloader = DataLoader(TensorDataset(data, labels), batch_size = batch_size, shuffle = shuffle, num_workers=0)
    
    if return_weights:
        _ , weights = np.unique(labels, return_counts=True)
        weights = 1000 / weights
        return dataloader, weights
    return dataloader
 
    
    
def main():
    #dfs = preprocess_dataset("condensed", persons=[1,])
    #dfs = preprocess_dataset("condensed", pca = 5, persons=[1,])
    #dfs = preprocess_dataset("concat", persons = [1])
    dfs = preprocess_dataset("concat", pca = 5, persons=[1])

    
    
    loader_train, weights = create_dataloader(dfs, 256, undersampling = False, return_weights= True)
    loader_test = create_dataloader(dfs, 256, undersampling = True)
   
    print(len(loader_train))
    print(len(loader_test))
    
    print(weights)

    for d,l in loader_test:
        print(d.shape)
        print(l.shape)
        break

    

    
    
    

if __name__ == "__main__":
    main()