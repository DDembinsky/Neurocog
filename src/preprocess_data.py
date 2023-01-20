import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA

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
    
    __COLS = ["left mean", "right mean","acc_x mean","acc_y mean","acc_z mean","roll mean","pitch mean","yaw mean",
        "left std","right std","acc_x std", "acc_y std","acc_z std", "roll std", "pitch std","yaw std",  
        "label"]
    
    
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
    
    var = ['left', 'right', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw',]
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
    df_features =  df_windowed[:,:,:-1].reshape((-1,consecutive_steps*8))
    
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


def preprocess_dataset(method: str, consecutive_steps : int = 3000, stride : int = 1000, normalize : bool = True, persons : list = None, days : list = None ) -> dict:
    """Preprocesses all raw csv files

    Args:
        method (str): Either "condensed" or "concat", for the used method of forming windows.
        consecutive_steps (int): How many consecutive steps form a window.
        stride (int): By how many  steps each sliding window is shifted.
        normalize (bool): If the input data should be normalized first. Default is True.
        persons (list): Specify which persons should be included in the dataset. If None, includes all.
        days (list): Specify which days should be included in the dataset. If None, includes all.
       
    """
    
    func = __form_window_condensed if method == "condensed" else __form_window_concat if method == "concat" else None
    if func is None:
        raise ValueError("method has to be either 'condensed' or 'concat', but '''{}''' was given".format(method))
    
    dfs = {}
    
    if persons is None:
        persons = list(range(1,11,1))
    if days is None:
        days = list(range(1,3,1))
    
    # Load each raw dataset
    for p in persons:
        for d in days:
            _st_t = datetime.now()
            file_name = "p{}_d{}".format(p,d)
            
            df = pd.read_csv("data/" + file_name + ".csv")
            
            # Remove unnecessary columns
            df = df.drop(labels=["#timestamp","date_time"], axis=1)
            
            if normalize:
                df = __normalize_dataframe(df)

            df_new = func(df = df, consecutive_steps = consecutive_steps , stride = stride)
            dfs[file_name] = df_new
            
            _en_t = datetime.now()
            _t = _en_t - _st_t
            print("Finished {}. Took {} minutes, {} seconds".format(file_name,_t.seconds//60, _t.seconds%60)) 
    
    return dfs      
            
def pca_dataset(dfs : dict, num_pc : int = None) -> dict:
    """Performs PCA over the entire dataset and transform each individual dataset accordingly

    Args:
        dfs (dict): Dict of datasets as provided by ```preprocess_dataset```
        num_pc (int, optional): How many PCs to use, if None uses all. Defaults to None.

    Returns:
        dict: Dict with same keys as dfs but each dataset is transformed with PCA
    """

    data = np.concatenate([
        df.to_numpy() for df in dfs.values()
    ])
    
    
    pcs = PCA(n_components=num_pc).fit(data[:,:-1])


    dfs_new = {
        key : pd.DataFrame(
            np.concatenate([pcs.transform(dfs[key].to_numpy()[:,:-1]) , dfs[key].to_numpy()[:,-1, None]], axis = 1),
            columns=["PC{}".format(i+1) for i in range(pcs.n_components_) ] + ["label"]
        )
        for key in dfs.keys()
    }
    
    return dfs_new
    
    
    
    
    
    
def main():
    data = preprocess_dataset("concat", 5, False, persons=[1], days=[1])
    print(len(data))
    print(data.keys())
    
    df = data["p1_d1"]
    print(df.shape)
    print(df.head())
    
    print("\n\nPCA\n\n")

    data = pca_dataset(data, num_pc = 3)
    print(len(data))
    print(data.keys())
    
    df = data["p1_d1"]
    print(df.shape)
    print(df.head())

    

    
    
    

if __name__ == "__main__":
    main()