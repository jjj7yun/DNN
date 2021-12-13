import torch
import torch.nn as nn

class categorical(nn.Module):
    def __init__(self, num_categories, num_features, num_classes):
        super(categorical, self).__init__()
        
        '''
        num_categories = number of digits [0,1,2,3,4,5,6,7,8,9] -> 10
        num_features = number of features to represent numbers -> 23 to [0,0,0,2,3]
        num_classes = number of unique labels. [clap, None] -> 2 in 369 clap
        '''
        
        self.num_features = num_features
        self.num_categories = num_categories

        # ----------------------------------- #
        # FILL THIS PART TO COMPLETE THE CODE #
        self.num_features = num_features
        self.num_categories = num_categories

        self.hidden_size1 = 16
        self.hidden_size2 = 64
        self.hidden_size3 = 16

        self.embedding = nn.Embedding(self.num_categories, self.hidden_size1)
        self.layer1 = nn.Linear(self.hidden_size1, self.hidden_size2)  ## embedding lookup sum
        self.layer2 = nn.Linear(self.hidden_size2, self.hidden_size3)
        self.layer_out = nn.Linear(self.hidden_size3, num_classes)
        self.Sigmoid = nn.Sigmoid()
        
        #                                     #
        # ----------------------------------- #


    def forward(self, x):

        '''
        x = Long Tensor size of (batch, num_features)
        e.g., tensor([[0, 0, 3, 8, 2],
                      [1, 0, 4, 1, 0],
                      [1, 8, 4, 4, 8]])
        
        output = Float Tensor size of (batch, num_classes)
        e.g., tensor([[-1.2837, 2.1892],
                      [3.2111, 0.5821],
                      [3.4112, -0.9710]])
        '''

        # ----------------------------------- #
        # FILL THIS PART TO COMPLETE THE CODE #
        x = self.embedding(x)
        x = self.layer1(torch.sum(x, 1))
        x = self.layer2(x)
        x = self.layer_out(x)
        output = self.Sigmoid(x)
        #                                     #
        # ----------------------------------- #
       
        return output