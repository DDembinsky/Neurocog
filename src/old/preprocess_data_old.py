import pandas as pd
import numpy as np
from datetime import datetime

import os

def form_window_condensed(df: pd.DataFrame , consecutive_steps: int,  path: str, allow_multilabel : bool = False, overlapping : bool = False,) -> pd.DataFrame:
    """Creates a new dataset of time windows by calculating meat-informaion of the window's features. 
    Saves it directly to a csv. The columns are   (left | right | acc_x | acc_y | acc_z | roll | pitch | yaw) *(mean | std) + label

    Args:
        df (pd.DataFrame): The original dataframe. Expects to have timestamp and datetime removed.
        consecutive_steps (int): The number of consecutive steps that form a window.
        path (str): The path of the csv to save the new dataset to
        allow_multilabel (bool, optional): If True, the label ob a window is determined by majority vote. If False, cuts of the window if switching labels are detected. Defaults to False.
        overlapping (bool, optional): Wether windows should overlap or not. Defaults to False.

   
    """
    
    def __save(_df_buffer, _label):
        _new_df = pd.DataFrame({
        "left mean"  : [np.mean(_df_buffer.loc[:,"left"])],  "left std" :  [np.std(_df_buffer.loc[:,"left"])], 
        "right mean" : [np.mean(_df_buffer.loc[:,"right"])], "right std" : [np.std(_df_buffer.loc[:,"right"])], 
        "acc_x mean" : [np.mean(_df_buffer.loc[:,"acc_x"])], "acc_x std" : [np.std(_df_buffer.loc[:,"acc_x"])], 
        "acc_y mean" : [np.mean(_df_buffer.loc[:,"acc_y"])], "acc_y std" : [np.std(_df_buffer.loc[:,"acc_y"])], 
        "acc_z mean" : [np.mean(_df_buffer.loc[:,"acc_z"])], "acc_z std" : [np.std(_df_buffer.loc[:,"acc_z"])], 
        "roll mean"  : [np.mean(_df_buffer.loc[:,"roll"])], "roll std" :   [np.std(_df_buffer.loc[:,"roll"])], 
        "pitch mean" : [np.mean(_df_buffer.loc[:,"pitch"])], "pitch std" : [np.std(_df_buffer.loc[:,"pitch"])], 
        "yaw mean"   : [np.mean(_df_buffer.loc[:,"yaw"])], "yaw std" :     [np.std(_df_buffer.loc[:,"yaw"])], 
        "label" : [_label]
        })
        
        _new_df.to_csv(path, index=False, header=False, mode="a" )
        return  
    
    
    # Create new df
    _dropped_windows = 0
    new_df = pd.DataFrame({
        "left mean" : [], "left std" : [], 
        "right mean" : [], "right std" : [], 
        "acc_x mean" : [], "acc_x std" : [], 
        "acc_y mean" : [], "acc_y std" : [], 
        "acc_z mean" : [], "acc_z std" : [], 
        "roll mean" : [], "roll std" : [], 
        "pitch mean" : [], "pitch std" : [], 
        "yaw mean" : [], "yaw std" : [], 
        "label" : []
        })
    new_df.to_csv(path, index=False)
    # Iterate over old df and form windows
    
    step_size = 1 if overlapping else consecutive_steps
    for i in range(0,len(df), step_size):
        df_buffer = df.iloc[i:i+consecutive_steps]
        
        # Determine label
        labels = np.array(df_buffer["label"].tolist())
        
        if not allow_multilabel:
            # Check if different label. (Numpy is faster than iterating)
            labels_ = labels - labels[0]
            if np.count_nonzero(labels_) > 0:
                _dropped_windows +=1
                continue
        l = np.argmax(np.bincount(labels))
     
        # Check length
        if len(df_buffer) < consecutive_steps:
            # End of data frame reached
            break 
        
        # Save
        __save(df_buffer,l)
    
    #print("New dataframe has {} windows. {} windows were dropped in the process.".format(len(new_df), _dropped_windows))
    return

def form_window_concat(df: pd.DataFrame , consecutive_steps: int, path: str, allow_multilabel : bool = False, overlapping : bool = False) -> pd.DataFrame:
    """Creates a new dataset of time windows by concatenating all features of the window.
    Saves it directly to csv. 
    The new dataframe contains the columns (left | right | acc_x | acc_y | acc_z | roll | pitch | yaw) * #consecutive_steps + label

    Args:
        df (pd.DataFrame): The original dataframe. Expects to have timestamp and datetime removed.
        consecutive_steps (int): The number of consecutive steps that form a window.
        path (str): The path of the csv to save the new dataset to
        allow_multilabel (bool, optional): If True, the label ob a window is determined by majority vote. If False, cuts of the window if switching labels are detected. Defaults to False.
        overlapping (bool, optional): Wether windows should overlap or not. Defaults to False.
    """
    
    def __save(_df_buffer,_label):
        
        var = list(_df_buffer.columns)
        var.remove("label")
        columns = {}
        
        for i in range(len(_df_buffer)):
            columns.update({"{}_{}".format(v,i) : [_df_buffer.iloc[i][v]] for v in var})
        columns["label"] = _label

        _new_df = pd.DataFrame(columns)
        _new_df.to_csv(path, index=False, header=False, mode="a" )
        return 
            
      
    # Create new dataframe
    var = list(df.columns)
    var.remove("label")
    columns = {}
    for i in range(consecutive_steps):
        columns.update({"{}_{}".format(v,i) : [] for v in var})

    new_df = pd.DataFrame(columns)
    new_df.to_csv(path, index=False)
        
    _dropped_windows = 0
    # Iterate over old of and form windows
    step_size = 1 if overlapping else consecutive_steps
    for i in range(0,len(df), step_size):
        df_buffer = df[i:i+consecutive_steps]
        
        
        # Determine label
        labels = np.array(df_buffer["label"].tolist())
        
        if not allow_multilabel:
            # Check if different label. (Numpy is faster than iterating)
            labels_ = labels - labels[0]
            if np.count_nonzero(labels_) > 0:
                _dropped_windows +=1
                continue
        l = np.argmax(np.bincount(labels))
        
        # Check length
        if len(df_buffer) < consecutive_steps:
            # End of data frame reached
            break 
        # Save
        __save(df_buffer,l)
    
    #print("New dataframe has {} windows. {} windows were dropped in the process.".format(len(new_df), _dropped_windows))
    return


def __get_dirctory_name(method: str, conscutive_steps: int,normalize:bool, **kwargs) -> str:
    """Retruns the name of the directory that matches the given kwargs

        method (str): Either "condensed" or "concat", for the used method of forming windows
        All arguments are expected to match the argumetns of the methods `form_window_condensed`and `form_window_concat`, excluding df and consecutive_steps
    """
    
    if not "allow_multilabel" in kwargs:
        kwargs["allow_multilabel"] = False
    if not "overlapping" in kwargs:
        kwargs["overlapping"] = False
    
    kwarg_list = list(kwargs)
    kwarg_list.sort()
    
    dir_name = ""

    for k in kwarg_list:
        if kwargs[k]:
            dir_name += k + "__"
   
    n_flag = "N__" if normalize else ""
    dir_name = n_flag + dir_name + str(conscutive_steps)

    return dir_name
    
def __normalize_dataframe(df : pd.DataFrame) -> pd.DataFrame:
    """Normalizes the dataset per feature without changing the labels
    """
    cols_to_norm = df.columns.drop("label")

    df_new = df.copy()
    df_new[cols_to_norm] = (df[cols_to_norm]-df[cols_to_norm].mean())/df[cols_to_norm].std()
    return df_new   

### !!!
### !!!
##
## Takes all args and inserts them into the first argument df. Absolutely unwanted behaviour!
## Should isntead unpack the dict like **kwargs would do
## Apply_Async didn't do the trick either. We have to somehow retirve the restults
##
def preprocess_dataset(method: str, conscutive_steps : int, normalize : bool = True, verbose = False, force : bool = False, **kwargs) -> None:
    """Preprocesses all raw csv files according to kwargs

    Args:
        method (str): Either "condensed" or "concat", for the used method of forming windows.
        conscutive_steps (int): How many consecutive steps form a window.
        normalize (bool): If the input data should be normalized first. Default is True.
        verbose (bool): Prints information on which dataset is currently processed. Default is False.
        force (bool): Whether an existing file should be deleted and overridden. Default is False.
        **kwargs (dict): Depend on the used method and are forwarded.
    """
    
    func = form_window_condensed if method == "condensed" else form_window_concat if method == "concat" else None
    if func is None:
        raise ValueError("method has to be either 'condensed' or 'concat', but '''{}''' was given".format(method))
    
    # Get directory_name
    dir_name = "data/" + method + "/" + __get_dirctory_name(method, conscutive_steps,normalize, **kwargs)
    try:
        os.mkdir(dir_name)
    except OSError:
        pass
    
    # Load each raw dataset
    if verbose:
        print("Start preprocessing to folder {}".format(dir_name)) 
    for p in range(1,11,1):
        for d in range(1,3,1):
            _st_t = datetime.now()
            file_name = "p{}_d{}.csv".format(p,d)
            
            df = pd.read_csv("data/raw/" + file_name)
            
            # Remove unnecessary columns
            df = df.drop(labels=["#timestamp","date_time"], axis=1)
            
            if normalize:
                df = __normalize_dataframe(df)
            path =  dir_name + "/" + file_name
            if os.path.exists(path):
                if force:
                    os.remove(path)
                else:
                    if verbose:
                        print("File {} allreaady exists. Use `force=True` if you want to override it.".format(file_name))
                    continue
            
            
            func(df = df, consecutive_steps = conscutive_steps , path =path, **kwargs )
            
            _en_t = datetime.now()
            _t = _en_t - _st_t
            if verbose:
                print("Saved {}. Took {} minutes, {} seconds".format(file_name,_t.seconds//60, _t.seconds%60)) 
            
            

    
    
###
# Joining the dataset probably takes for ever. Would be better, to just write the windows directly into a csv file
# Maybe keep track of indices for later reordering, although it might be unnecessary    

    
def main():
    preprocess_dataset("condensed", 5, verbose=True, force=True)
    

    
    
    

if __name__ == "__main__":
    main()