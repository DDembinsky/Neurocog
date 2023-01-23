import torch



class EOGModel(torch.nn.Module):
    """Base class for all models thta work on the EOG Dataset  """
    
    def __init__(self, in_features, out_features, num_hidden) -> None:
        super().__init__()
        self.num_layer = []

    
    def description(self):
        return str(self.num_hidden)
    
    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer,"reset_parameter"):
                layer.reset_parameters()
    def __prepare_timeseries(self,x):
        return x
    def forward(self,x):
        x = self.__prepare_timeseries(x)
        raise NotImplemented("This is an abstract class")

    
class Linear_NN(EOGModel):
    def __init__(self, in_features = 16, out_features = 4, num_hidden = []) -> None:
        """Creates a linear neural network from the arguments provided

        Args:
            in_features (int, optional): The number of features from the dataset. Defaults to 16.
            out_features (int, optional): The number of classes in the dataset. Defaults to 4.
            num_hidden (list, optional): A list containing the number of additional hidden layers. E.g '[10,5]' will create two hidden layers with 10 and 5 neurons respectiveley. Between two layers will always be a ReLU module. Defaults to [].
        """
        super().__init__(in_features, out_features, num_hidden )
  
        # Add input and output to layers
        self.num_hidden = [in_features] + num_hidden + [out_features]
        layers = []
        
        for i in range(len(self.num_hidden)-1):
            layers.append(
                torch.nn.Linear(in_features=self.num_hidden[i], out_features= self.num_hidden[i+1]))
            layers.append(torch.nn.ReLU())
            
        # Remove last ReLu
        del layers[-1]
            
        self.model = torch.nn.Sequential(*layers)
               
    def forward(self,x):
        return self.model(x)

class OneD_Conv(EOGModel):
    def __init__(self,in_features = 16, out_features = 4, num_hidden = [[10],[146,20]]) -> None:
        """Creates a 1D-Conv network fro mthe arguments provided

        Args:
            in_features (int, optional): The number of features from the dataset. Defaults to 16.
            out_features (int, optional): The number of classes in the dataset. Defaults to 4.
            num_hidden (list, optional): A list containing two lists describing the hidden layers. First list describes the features of the conv layers, second the linear layers. Defaults to [[10],[146,20]].
        """
        super().__init__(in_features, out_features, num_hidden)
        k = 100
        cs = 20
        ps = 10
        
        self.num_hidden = [  [in_features] + num_hidden[0], num_hidden[1] + [out_features]  ]
        
        
        conv_layers = []
        for i in range(len(self.num_hidden[0]-1)):
            conv_layers.append(
                torch.nn.Conv1d(in_features= self.num_hidden[0][i], out_features= self.num_hidden[0][i+1],kernel_size=k, stride=cs))
            conv_layers.append(torch.nn.ReLU())
            conv_layers.append(torch.nn.MaxPool1d(ps))
        
        self.conv = torch.nn.Sequential(*conv_layers)
        
        lin_layers = []
        for i in range(len(self.num_hidden[1])-1):
            lin_layers.append(
                torch.nn.Linear(in_features=self.num_hidden[i], out_features= self.num_hidden[i+1]))
            lin_layers.append(torch.nn.ReLU())
            
        # Remove last ReLu
        del lin_layers[-1]
            
        self.linear = torch.nn.Sequential(*lin_layers)
    
    def __prepare_timeseries(self,x,in_features):
        """Input: (N, C*L)  -> Output (N,C,L)"""
        x = torch.transpose(x.reshape([len(x),-1,4],),1,2)
        return x
    
    def forward(self, x):
        x = self.__prepare_timeseries(x,self.num_hidden[0][0])
        x_ = self.conv(x)
        x_ = torch.flatten(x_,start_dim=1)      
        return self.linear(x_)

class Rec_NN(EOGModel):
    def __prepare_timeseries(self,x,in_features):
        """Input: (N, C*L)  -> Output (N,L,C)"""
        x = x.reshape([len(x),-1,in_features])
        return x
        