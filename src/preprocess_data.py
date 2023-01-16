import pandas as pd
import numpy as np
from datetime import datetime

import os

def form_window_condensed(df: pd.DataFrame , consecutive_steps: int, overlapping : bool = False,) -> pd.DataFrame:
    """Creates a new dataset of time windows by calculating meat-informaion of the window's features. 
    The columns are   (left | right | acc_x | acc_y | acc_z | roll | pitch | yaw) *(mean | std) + label
    The label is calculated as the median of the labels in the time frame.
    
    
    Args:
        df (pd.DataFrame): The original dataframe. Expects to have timestamp and datetime removed.
        consecutive_steps (int): The number of consecutive steps that form a window.
        overlapping (bool, optional): Wether windows should overlap or not. Defaults to False.
    """
    
    __COLS = ["left mean", "right mean","acc_x mean","acc_y mean","acc_z mean","roll mean","pitch mean","yaw mean",
        "left std","right std","acc_x std", "acc_y std","acc_z std", "roll std", "pitch std","yaw std",  
        "label"]
    
    
    # Convert to NumPY
    df_n = df.to_numpy()
    
    # Create windows
    if overlapping:
        indices = np.array([[k for k in range(i,i+consecutive_steps,1)] for i in range(0,len(df_n) - consecutive_steps,1)] )
    else:
        indices = np.array([[k for k in range(i,i+consecutive_steps,1)] for i in range(0,len(df_n) - consecutive_steps,consecutive_steps)] )
    
    df_windowed = df_n[indices]
    
    # Calculate features
    df_win_mean = df_windowed[:,:,:-1].mean(axis=1)
    df_win_std  = df_windowed[:,:,:-1].std(axis=1)
    df_labels = np.median(df_windowed[:,:,-1] ,axis=1, keepdims=True).astype(int)
    
    
    # Create new df
    df_new = pd.DataFrame(data = np.concatenate([df_win_mean, df_win_std, df_labels], axis = 1),
                          columns= __COLS)

    return df_new

def form_window_concat(df: pd.DataFrame , consecutive_steps: int, overlapping : bool = False) -> pd.DataFrame:
    """Creates a new dataset of time windows by concatenating all features of the window.
    The new dataframe contains the columns (left | right | acc_x | acc_y | acc_z | roll | pitch | yaw) * #consecutive_steps + label
    The label is calculated as the median of the labels in the time frame.

    Args:
        df (pd.DataFrame): The original dataframe. Expects to have timestamp and datetime removed.
        consecutive_steps (int): The number of consecutive steps that form a window.
        overlapping (bool, optional): Wether windows should overlap or not. Defaults to False.
    """
    var = ['left', 'right', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw',]
    __COLS = []
    for i in range(consecutive_steps):
        __COLS.extend(["{}_{}".format(v,i) for v in var])
    __COLS.append("label")


    # Convert to NumPY
    df_n = df.to_numpy()

    # Create windows
    if overlapping:
        indices = np.array([[k for k in range(i,i+consecutive_steps,1)] for i in range(0,len(df_n) - consecutive_steps,1)] )
    else:
        indices = np.array([[k for k in range(i,i+consecutive_steps,1)] for i in range(0,len(df_n) - consecutive_steps,consecutive_steps)] )

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


def preprocess_dataset(method: str, conscutive_steps : int, overlapping : bool, normalize : bool = True, persons : list = None, days : list = None ) -> None:
    """Preprocesses all raw csv files

    Args:
        method (str): Either "condensed" or "concat", for the used method of forming windows.
        conscutive_steps (int): How many consecutive steps form a window.
        overlapping (bool):  Wether windows should overlap or not. 
        normalize (bool): If the input data should be normalized first. Default is True.
        persons (list): Specify which persons should be included in the dataset. If None, includes all.
        days (list): Specify which days should be included in the dataset. If None, includes all.
       
    """
    
    func = form_window_condensed if method == "condensed" else form_window_concat if method == "concat" else None
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

            df_new = func(df = df, consecutive_steps = conscutive_steps , overlapping = overlapping)
            dfs[file_name] = df_new
            
            _en_t = datetime.now()
            _t = _en_t - _st_t
            print("Finished {}. Took {} minutes, {} seconds".format(file_name,_t.seconds//60, _t.seconds%60)) 
    
    return dfs      
            
            
    
def main():
    data = preprocess_dataset("concat", 5, False, persons=[1,2,3])
    print(len(data))
    print(data.keys())
    
    df = data["p1_d1"]
    print(len(df))
    print(df.head())

    

    
    
    

if __name__ == "__main__":
    main()