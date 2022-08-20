#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from DataGeneration import GenerateHARData, GenerateHAPTData
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.metrics import accuracy_score

class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, X, y):
        # store the inputs and outputs
        self.X = X
        self.y = y
 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    
    def get_splits(self, train_rate):
        n_data = len(self.X)
        train_size = int(n_data*train_rate)
        test_size = n_data - train_size
        return random_split(self, [train_size, test_size])

class OneDCNN(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(OneDCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Linear(3904,100),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(100,n_outputs),
            nn.Softmax(dim=1))
       
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

def train_model(train_dl, model, epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # enumerate epochs
    accuracy = list()
    for epoch in range(epoch):
        predictions, actuals = list(), list()
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            # record
            predictions.append(np.argmax(yhat.detach().numpy(), axis=1))
            actuals.append(np.argmax(targets.numpy(), axis=1))
        predictions, actuals = np.concatenate(predictions), np.concatenate(actuals)
        acc = accuracy_score(actuals, predictions)
        accuracy.append(acc)
        print(f"Epoch: {epoch+1}; Accuracy: {acc}")

    # visualization
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(accuracy)
    plt.show()

def model_evaluation(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        predictions.append(np.argmax(yhat.detach().numpy(), axis=1))
        actuals.append(np.argmax(targets.numpy(), axis=1))
    predictions, actuals = np.concatenate(predictions), np.concatenate(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    print(f"Test Accuracy: {acc}")

#%%
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# load data
X, y = GenerateHARData().run()
# trainsform data
XT = torch.from_numpy(X)
XT = XT.transpose(1,2).float() #input is (N, Cin, Lin) = Ntimesteps, Nfeatures, 128
yT = torch.from_numpy(y).float()

data = CSVDataset(XT, yT)
train, test = data.get_splits(train_rate=0.8)
# create a data loader for train and test sets
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=1024, shuffle=False)

n_timesteps =  XT.shape[2]
n_features = XT.shape[1]
n_outputs = yT.shape[1]

#%%
model = OneDCNN(n_timesteps, n_features, n_outputs)
train_model(train_dl, model, epoch=10)
model_evaluation(test_dl, model)

# %%
