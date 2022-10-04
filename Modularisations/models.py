'''
Created on Apr 25, 2022

@author: deckyal
'''

import torch
import torch.nn as nn

# Whole Class with additions:
class FC(nn.Module):
    def __init__(self,inputNode=561,hiddenNode = 256, outputNode=1):   
        super(FC, self).__init__()     
        #Define Hyperparameters
        self.inputLayerSize = inputNode
        self.outputLayerSize = outputNode
        self.hiddenLayerSize = hiddenNode
        
        # weights
        self.Linear1 = nn.Linear(self.inputLayerSize, self.hiddenLayerSize)
        self.Linear2 = nn.Linear(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        
        self.z2 = self.Linear1(X) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.a2 = self.sigmoid(self.z2) # activation function
        self.z3 = self.Linear2(self.a2)
        return self.z3
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+torch.exp(-z))
    
    def loss(self, yHat, y):
        J = 0.5*sum((y-yHat)**2)