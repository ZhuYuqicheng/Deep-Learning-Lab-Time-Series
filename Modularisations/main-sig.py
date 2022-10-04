'''
Created on Apr 25, 2022

@author: deckyal
'''
import torch
import torch.nn as nn
import numpy as np

from dataset import *
from models import *
from utils import * 
from metrics import * 
from config import device


def train(model = None,SavingName=None, train_loader = None, val_loader=None, optimizer = None):
    print('training')
    
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (signals, labels) in enumerate(train_loader):
            #print(signals, labels)
            labels = labels.to(device)
            signals = signals.to(device)

            # Forward pass
            outputs = model(signals)
            
            loss = torch.sqrt(torch.mean((outputs - labels) ** 2))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            
            
            if i%10 == 0:
                with torch.no_grad():
                    model.eval()     
                    
                    pred,gt = [],[]
                    
                    for signalsV, labelsV in val_loader:
                        
                        labelsV = labelsV.to(device)
                        signalsV = signalsV.to(device)
                        
                        outputsV = model(signalsV).round()
                        
                        gt.extend(labelsV.cpu().numpy()[0])
                        pred.extend(outputsV[0].round().cpu().numpy())
                    
                    gt = np.asarray(gt,np.float32)
                    pred = np.asarray(pred)
                        
                    print('Val Accuracy of the model on the {} epoch: {} %'.format(i,accuracy(pred,gt)))
                    
                model.train()
            
    # Save the model checkpoint
    torch.save(model.state_dict(), SavingName)
    # to load : model.load_state_dict(torch.load(save_name_ori))

def test(model = None,SavingName=None, test_loader=None):
    model.load_state_dict(torch.load(SavingName))
    # Test the model
    
    model.eval() 
    with torch.no_grad():
        
        pred,gt = [],[]
        
        for signals, labels in test_loader:
            
            signals = signals.to(device)
            outputs = model(signals).round().cpu().numpy()
            
            #print(labels)
            #print(outputs)
            
            gt.extend(labels.cpu().numpy()[0])
            pred.extend(outputs[0])
        
        gt = np.asarray(gt,np.float32)
        pred = np.asarray(pred)


        print('Test Accuracy of the model test samples: {} %'.format(accuracy(pred,gt)))

if __name__ == '__main__':

    batch_size = 16 
    
    ADLTrain = ADL(dataDir='../../../Data/HAR-ADL/', type="Train",selIndividual=list(range(27)))
    train_loader = torch.utils.data.DataLoader(dataset=ADLTrain,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    
    ADLVal = ADL(dataDir='../../../Data/HAR-ADL/', type="Train",selIndividual=list(range(27,30)))
    val_loader = torch.utils.data.DataLoader(dataset=ADLVal,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    ADLTest = ADL(dataDir='../../../Data/HAR-ADL/', type="Test")
    test_loader = torch.utils.data.DataLoader(dataset=ADLTest,
                                              batch_size=batch_size,
                                              shuffle=True)
    
    FCS = FC().to(device)
    
    learning_rate = .001
    num_epochs = 10
    optimizer = torch.optim.Adam(FCS.parameters(), lr=learning_rate)
    
    
    
    operation = 0
    
    if operation ==0 or operation==2: 
        train(model = FCS,SavingName='./checkpoints/FCS.ckpt', train_loader = train_loader, val_loader=val_loader, optimizer = optimizer)
    if operation ==1 or operation==2: 
        test(model = FCS,SavingName='./checkpoints/FCS.ckpt', test_loader=test_loader)
        
    
        