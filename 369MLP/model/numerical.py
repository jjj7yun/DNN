import torch.nn as nn

class numerical(nn.Module):
    def __init__(self, num_features, num_classes):
        super(numerical, self).__init__()
        
        '''
        num_features = number of features to represent numbers -> 23 to 23
        num_classes = number of unique labels. [clap, None] -> 2 in 369 clap
        '''

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
        e.g., tensor([382,
                      10410,
                      18448])
        
        output = Float Tensor size of (batch, num_classes)
        e.g., tensor([[-1.2837, 2.1892],
                      [3.2111, 0.5821],
                      [3.4112, -0.9710]])
        '''

        x = self.embedding(x)
        x = self.layer1(torch.sum(x, 1))
        x = self.layer2(x)
        x = self.layer_out(x)
        output = self.Sigmoid(x)

        #                                     #
        # ----------------------------------- #
        
        return output