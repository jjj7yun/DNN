from collections import OrderedDict
import numpy as np


class ReLU:
    """
    ReLU Function. ReLU(x) = max(0, x)
    Implement forward & backward path of ReLU.

    ReLU(x) = x if x > 0.
              0 otherwise.
    Be careful. It's '>', not '>='.
    """

    def __init__(self):
        # 1 (True) if ReLU input <= 0
        self.zero_mask = None

    def forward(self, z):
        """
        ReLU Forward.
        ReLU(x) = max(0, x)

        z --> (ReLU) --> out

        [Inputs]
            z : ReLU input in any shape.

        [Outputs]
            self.out : Values applied elementwise ReLU function on input 'z'.
        """
        self.out = None

        self.out = z
        self.out[self.out <= 0] = 0
        ###print("selfout="self.out)

        return self.out
        

    def backward(self, d_prev, reg_lambda):
        """
        ReLU Backward.

        z --> (ReLU) --> out
        dz <-- (dReLU) <-- d_prev(dL/dout)

        [Inputs]
            d_prev : Gradients flow from upper layer.
                - d_prev = dL/dk, where k = ReLU(z).
            reg_lambda: L2 regularization weight. (Not used in activation function)
        [Outputs]
            dz : Gradients w.r.t. ReLU input z.
        """
        dz = None

        ###print(dz)
        ###print(d_prev)
        self.out[self.out>0] =1
        
        dz = d_prev * self.out
    

        return dz

    def update(self, learning_rate):
        # NOT USED IN ReLU
        pass

    def summary(self):
        return 'ReLU Activation'


class Sigmoid:
    """
    Sigmoid Function.
    Implement forward & backward path of Sigmoid.
    """

    def __init__(self):
        self.out = None

    def forward(self, z):
        """
        Sigmoid Forward.

        z --> (Sigmoid) --> self.out

        [Inputs]
            z : Sigmoid input in any shape.

        [Outputs]
            self.out : Values applied elementwise sigmoid function on input 'z'.
        """
        self.out = None

        out = 1 / (1 + np.exp(-z))
        self.out = out
        # =========================================
        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        Sigmoid Backward.

        z --> (Sigmoid) --> self.out
        dz <-- (dSigmoid) <-- d_prev(dL/d self.out)

        [Inputs]
            d_prev : Gradients flow from upper layer.
            reg_lambda: L2 regularization weight. (Not used in activation function)

        [Outputs]
            dz : Gradients w.r.t. Sigmoid input z .
        """
        dz = None

        dz = d_prev * (1.0 - self.out) * self.out
        # =========================================
        return dz

    def update(self, learning_rate):
        # NOT USED IN Sigmoid
        pass

    def summary(self):
        return 'Sigmoid Activation'


class Tanh:
    """
    Hyperbolic Tangent Function(Tanh).
    Implement forward & backward path of Tanh.
    """

    def __init__(self):
        self.out = None

    def forward(self, z):

        self.out = None

        out = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        self.out = out
        # =========================================
        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        Hyperbolic Tangent Backward.

        z --> (Tanh) --> self.out
        dz <-- (dTanh) <-- d_prev(dL/d self.out)

        [Inputs]
            d_prev : Gradients flow from upper layer.
            reg_lambda: L2 regularization weight. (Not used in activation function)

        [Outputs]
            dz : Gradients w.r.t. Tanh input z .
            In other words, the derivative of tanh should be reflected on d_prev.
        """
        dz = None

        dz = (1-self.out)*(1+self.out) * d_prev
        # =========================================
        return dz

    def update(self, learning_rate):
        # NOT USED IN Tanh
        pass

    def summary(self):
        return 'Tanh Activation'

"""
    ** Fully-Connected Layer **
    Single Fully-Connected Layer.

    Given input features,
    FC layer linearly transforms the input with weights (self.W) & bias (self.b).

    You need to implement forward and backward pass.
"""

class FCLayer:
    def __init__(self, input_dim, output_dim):
        # Weight Initialization
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(6/(input_dim+output_dim))
        self.b = np.zeros(output_dim)

    def forward(self, x):
        """
        FC Layer Forward.
        Use variables : self.x, self.W, self.b

        [Input]
        x: Input features.
        - Shape : (batch size, In Channel, Height, Width)
        or
        - Shape : (batch size, input_dim)

        [Output]
        self.out : fc result
        - Shape : (batch size, output_dim)

        Tip : you do not need to implement L2 regularization here. already implemented in RegressionModel.forward()
        """

        self.x = x
        self.out = None
        #print('x=',x)
        #print('W=',self.W)
        self.out = np.dot(x,self.W) + self.b
        #print('self.out=',self.out)
        # =========================================================================
        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        FC Layer Backward.
        Use variables : self.x, self.W

        [Input]
        d_prev: Gradients value so far in back-propagation process.
        reg_lambda: L2 regularization weight. (Not used in activation function)

        [Output]
        dx : Gradients w.r.t input x
        - Shape : (batch_size, input_dim) - same shape as input x

        Tip : you should implement backpropagation of L2 regularization as well.
        """
        dx = None           # Gradient w.r.t. input x
        self.dW = None      # Gradient w.r.t. weight (self.W)
        self.db = None      # Gradient w.r.t. bias (self.b)

        dx = np.dot(d_prev,self.W.T)
        self.dW = np.dot(self.x.T, d_prev) + reg_lambda*self.W
        self.db = np.sum(d_prev,axis = 0)
        # =========================================================================
        return dx

    def update(self, learning_rate):
        self.W -= self.dW * learning_rate
        self.b -= self.db * learning_rate

    def summary(self):
        return 'Input -> Hidden : %d -> %d ' % (self.W.shape[0], self.W.shape[1])

"""
    ** MSE Loss Layer **
    Given an score,
    'MSELoss' calculate the loss using true label.

    You need to implement forward and backward pass.
"""

class MSELoss:
    def __init__(self):
        # No parameters
        pass

    def forward(self, y_hat, y):
        """
        Compute Mean Squared Error Loss.
        Check loss function in HW word file.

        [Input]
        y_hat: Predicted value
        - Shape : (batch_size, 1)

        y: True value
        - Shape : (batch_size, 1)

        [Output]
        loss : mse loss
        - float
        """
        loss = None
        self.y_hat = y_hat
        self.y = y
        self.error = y - y_hat

        loss = np.mean(self.error ** 2)
        # =========================================================================
        return loss

    def backward(self, d_prev=1, reg_lambda=0):
        """
        MSEloss Backward.
        Gradients w.r.t score.

        Forward  : L = Mean(Error^2) ; Error = y - y_hat
        Backward : dL / dy_hat

        Check loss function in HW word file.

        [Input]
        d_prev : Gradients flow from upper layer.

        [Output]
        dyhat: Gradients of MSE loss layer input 'y_hat'
        """
        batch_size = self.y.shape[0]
        dyhat = None

        dyhat = (self.y_hat - self.y) / batch_size
        # =========================================================================
        return dyhat


"""
    ** Regression Model **
    This is an class for entire Regression Model.
    All the functions and variables are already implemented.
    Look at the codes below and see how the codes work.

    <<< DO NOT CHANGE ANYTHING HERE >>>
"""


class RegressionModel:
    def __init__(self,):
        self.layers = OrderedDict()
        self.loss_layer = MSELoss()
        self.pred = None

    def predict(self, x):
        # Outputs model score
        for name, layer in self.layers.items():
            x = layer.forward(x)
        return x

    def forward(self, x, y, reg_lambda):
        # Predicts and Compute CE Loss
        reg_loss = 0
        self.pred = self.predict(x)
        loss = self.loss_layer.forward(self.pred, y)

        for name, layer in self.layers.items():
            if isinstance(layer, FCLayer):
                norm = np.linalg.norm(layer.W, 2)
                reg_loss += 0.5 * reg_lambda * norm * norm

        self.loss = loss + reg_loss

        return self.loss

    def backward(self, reg_lambda):
        # Back-propagation
        d_prev = 1
        d_prev = self.loss_layer.backward(d_prev)
        for name, layer in list(self.layers.items())[::-1]:
            d_prev = layer.backward(d_prev, reg_lambda)

    def update(self, learning_rate):
        # Update weights in every layer with dW, db
        for name, layer in self.layers.items():
            layer.update(learning_rate)

    def add_layer(self, name, layer):
        # Add Neural Net layer with name.
        self.layers[name] = layer

    def summary(self):
        # Print model architecture.
        print('======= Model Summary =======')
        for name, layer in self.layers.items():
            print('[%s] ' % name + layer.summary())
        print()