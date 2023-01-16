import numpy as np
import pandas as pd
import math
import datetime

from src.preprocess_data import preprocess_dataset


class KNN():
    MAX_MATRIX_ENTRIES = 2e8
    def __init__(self, num_cols : int, k : int ) -> None:
        """Creates an empty KNN classifier

        Args:
            num_cols (int): The number of columns (features) the data is going to have, including the label.
            k (int): The number of neighbours that should be considered.
        """
        self.data = np.empty([0,num_cols])
        self.shape = self.data.shape
        self.k = k
        self.__verbose =  False
        return
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __call__(self, *args, **kwds):
        return self.top_k(*args, **kwds)

    def verbose(self, verbose = True) -> None:
        """Sets the model to the verbose mode, where it comments on what it is doing.

        Args:
            verbose (bool, optional): Defaults to True.
        """
        self.__verbose = True
        return None
    
    def store_data(self,df : pd.DataFrame ) -> None:
        """Stores a single dataframe into the internal storage

        Args:
            df (pd.DataFrame): The data to be saved.
        """ 
        self.data = np.concatenate([self.data, df.to_numpy()])
        self.shape = self.data.shape
        
        if self.__verbose:
            print("New shape is {}.".format(self.shape))
        # Reset internal
        self.__dat = None
        self.__lab = None
        return
    
    def set_k(self,k:int) -> None:
        """Set another k hyperparameter

        Args:
            k (int): The new k
        """
        self.k = k
    
    def top_k(self, datapoint : np.ndarray) -> np.ndarray:
        """Calculates the label using the cosine similarity between each datapoint and the stored data points

        Args:
            datapoint (np.narray): An array of shape [N,C], where N is the number of datasamples (rows) and C the number of features (columns), not including the label

        Returns:
            np.ndarray: An array of shape [N], containing the label prediction 
        """
        assert datapoint.shape[1] == self.shape[1]-1
        
        datapoint = datapoint.copy()
        
        # Pepare internal data representation for calculation
        if self.__dat == None:
            
            # Remove label column
            self.__dat = self.data[:,:-1]
            self.__lab = self.data[:,-1].astype(int)
            
            # Normalize each row to length 1
            self.__dat = self.__dat  / np.linalg.norm(self.__dat, axis = 1, keepdims=True)

        datapoint = datapoint  / np.linalg.norm(datapoint, axis = 1, keepdims=True)
        
        # Split input array into smaller ones if N*n is to big
        step_size = max(1,int(KNN.MAX_MATRIX_ENTRIES // len(self)))
        labels_accumulator = []
        
        max_i = math.ceil(len(datapoint)/ step_size)
        
        if self.__verbose:
            print("Starts calculating on self.data ({}) and input ({}).".format(self.shape, datapoint.shape))
            _st_t = datetime.datetime.now()
    
        for i in range(max_i):
            
            if self.__verbose:
                _en_t = datetime.datetime.now()
                _t = _en_t - _st_t
                if i == 0:
                     _t_ex = datetime.timedelta(seconds = 0)
                else:
                    _t_ex = _t *(max_i/i)

            
                print("Finsihed datapoints {}/{} ({}/{})    {}m{}s/{}m{}s".format(
                    i * step_size, len(datapoint), 
                    i,max_i,
                    _t.seconds//60, _t.seconds%60,
                    _t_ex.seconds//60, _t_ex.seconds%60,
                    ),
                      end = "\r")
                
            # Calculate dot product of matrices
            cos_sim = datapoint[i*step_size:(i+1)*step_size] @ self.__dat.T
            
            # Get indices of highest value
            ind = np.argpartition(cos_sim, -self.k, axis = 1)[:,-self.k:]
       
            # Get corresponding labels from internal data
            labels = np.array([self.__lab[ind[i]] for i in range(len(ind))])     
        
            # Do majority vote for each input data point
            labels = np.array([np.argmax(np.bincount(labels[i])) for i in range(len(labels))])

            labels_accumulator.append(labels)
        
        if self.__verbose:
            _en_t = datetime.datetime.now()
            _t = _en_t - _st_t
            _t_ex = _t

        
            print("Finsihed datapoints {}/{} ({}/{})    {}m{}s/{}m{}s".format(
                len(datapoint), len(datapoint), 
                max_i,max_i,
                _t.seconds//60, _t.seconds%60,
                _t_ex.seconds//60, _t_ex.seconds%60,
                ),)

        return np.concatenate(labels_accumulator)
        

def calculate_accuraccy(labels1 : np.ndarray, labels2 : np.ndarray) -> float:
    """Calculates the accuraccy of a prediction in refference to the ground-truth

    Args:
        labels1 (np.ndarray): Ground-truth or prediction of shape [N]
        labels2 (np.ndarray): Ground-truth or prediction of shape [N]

    Returns:
        float: The accuraccy of the prediction
    """
    assert labels1.shape == labels2.shape
    
    diff = labels1 - labels2
    wrong = np.count_nonzero(diff)
    correct = len(labels1) - wrong
    
    return correct/len(labels1) * 100











    


def main():
    dfs = preprocess_dataset("condensed", 100, False, persons=list(range(1,11,1)) )
    
    
if __name__ == "__main__":
    main()