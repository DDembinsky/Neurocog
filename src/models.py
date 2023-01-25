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
        raise self.model(x)

    
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

class OneD_Conv(EOGModel):
    def __init__(self,in_features = 16, out_features = 4, num_hidden = [[10],[140,20]]) -> None:
        """Creates a 1D-Conv network from the arguments provided

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
        for i in range(len(self.num_hidden[0])-1):
            conv_layers.append(
                torch.nn.Conv1d(in_channels= self.num_hidden[0][i], out_channels= self.num_hidden[0][i+1],kernel_size=k, stride=cs))
            conv_layers.append(torch.nn.ReLU())
            conv_layers.append(torch.nn.MaxPool1d(ps))
        
        self.conv = torch.nn.Sequential(*conv_layers)
        
        lin_layers = []
        for i in range(len(self.num_hidden[1])-1):
            lin_layers.append(
                torch.nn.Linear(in_features=self.num_hidden[1][i], out_features= self.num_hidden[1][i+1]))
            lin_layers.append(torch.nn.ReLU())
            
        # Remove last ReLu
        del lin_layers[-1]
            
        self.linear = torch.nn.Sequential(*lin_layers)
    
    def reset_parameters(self):
        for layer in self.linear.children():
            if hasattr(layer,"reset_parameter"):
                layer.reset_parameters()
        for layer in self.conv.children():
            if hasattr(layer,"reset_parameter"):
                layer.reset_parameters()
                
    def __prepare_timeseries(self,x,in_features):
        """Input: (N, C*L)  -> Output (N,C,L)"""
        x = torch.transpose(x.reshape([len(x),-1,in_features],),1,2)
        return x
    
    def forward(self, x):
        x = self.__prepare_timeseries(x,self.num_hidden[0][0])
        x_ = self.conv(x)
        x_ = torch.flatten(x_,start_dim=1)      
        return self.linear(x_)

class Rec_NN(EOGModel):
    def __init__(self,in_features = 16, out_features = 4, num_hidden = [(20,1,False),[20]]) -> None:
        """Creates a LSTM network from the arguments provided

        Args:
            in_features (int, optional): The number of features from the dataset. Defaults to 16.
            out_features (int, optional): The number of classes in the dataset. Defaults to 4.
            num_hidden (list, optional): A list containing describing the hidden layers. First element is a tuple describing the LSTM (hidden, layers, bidirectional), second the linear layers.
        """
        super().__init__(in_features, out_features, num_hidden)
        
        self.num_hidden = [  in_features, num_hidden[0]] + num_hidden[1] + [out_features] 
        l_ = num_hidden[0]
        hidden_input = l_[0] * l_[1]
        if l_[2]:
            hidden_input *=2
        num_hidden[1] = [hidden_input] + num_hidden[1] + [out_features]
        
        self.lstm = torch.nn.LSTM(input_size = in_features, hidden_size = l_[0], num_layers = l_[1], bidirectional = l_[2], batch_first = True)
        lin_layers = []
        for i in range(len(num_hidden[1])-1):
            lin_layers.append(
                torch.nn.Linear(in_features=num_hidden[1][i], out_features= num_hidden[1][i+1]))
            lin_layers.append(torch.nn.ReLU())
        # Remove last ReLu
        del lin_layers[-1]    
        self.linear = torch.nn.Sequential(*lin_layers)
    
    def reset_parameters(self):
        for layer in self.linear.children():
            if hasattr(layer,"reset_parameter"):
                layer.reset_parameters()
        for layer in self.lstm.children():
            if hasattr(layer,"reset_parameter"):
                layer.reset_parameters()
     
    def __prepare_timeseries(self,x,in_features):
        """Input: (N, C*L)  -> Output (N,L,C)"""
        x = x.reshape([len(x),-1,in_features])
        return x
    
    def forward(self,x):
        x = self.__prepare_timeseries(x,self.num_hidden[0])
        _, (x_,_) = self.lstm(x)
        x_ = x_.transpose(0,1).flatten(start_dim=1)
        return self.linear(x_)


if __name__ == "__main__":
    model =  Rec_NN(in_features = 8, num_hidden = [(40,2,True),[40,40,40]])
    input = torch.rand([256,8*3000])
    print(model(input).shape)
    print(model.description())
    print(model)
