'''
Created on May 1, 2022

@author: deckyal
'''

from _operator import truediv

#expecting 1D np array of predictions and labels. 
def accuracy(predictions,labels):
    total = len(labels)
    correct = (predictions == labels).sum().item()
    return 100*truediv(correct,total)
