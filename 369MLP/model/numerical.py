import torch.nn as nn

class numerical(nn.Module):
    def __init__(self, num_features, num_classes):
        super(numerical, self).__init__()
        
        '''
        num_features = number of features to represent numbers -> 23 to 23
        num_classes = number of unique labels. [clap, None] -> 2 in 369 clap
        '''

        # ----------------------------------- #
        # FILL THIS PART TO COMPLETE THE CODE #
        
        
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

        # ----------------------------------- #
        # FILL THIS PART TO COMPLETE THE CODE #
    
        
        output = None
        #                                     #
        # ----------------------------------- #
        
        return output