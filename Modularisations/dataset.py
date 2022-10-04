'''
Created on Apr 24, 2022

@author: deckyal
'''
import torch
from torchvision import transforms
from PIL import Image
from _operator import truediv

class CatDog(torch.utils.data.Dataset):
    def __init__(self, dataDir='../../../Data/IC-CatDog/train-small/', transform=None, crossNum=None,crossIDs=None):
        # Initialize the data and label list
        self.labels = []
        self.data = []
        temp = []
        tempL = []

        # First load all images data
        import os
        listImage = os.listdir(dataDir)
        listImage = sorted(listImage)

        for x in listImage:

            #print(x)

            # Read the data using PIL
            temp.append(Image.open(dataDir + x).convert('RGB'))

            # Second filter according name for labelling : cat : 1, dog : 0
            if 'dog' in x:
                tempL.append(torch.FloatTensor([0]))
            else:
                tempL.append(torch.FloatTensor([1]))
        
        
        if crossNum is not None: 
            totalLength = len(temp)
            length = int(truediv(totalLength,crossNum))
            
            
            for crossID in crossIDs : 
                lowR = crossID-1
                if crossID == crossNum: 
                    self.data.extend(temp[(lowR)*length:])
                    self.labels.extend(tempL[(lowR)*length:])
                else: 
                    self.data.extend(temp[(lowR)*length:(crossID)*length])
                    self.labels.extend(tempL[(lowR)*length:(crossID)*length])
                
        else: 
            self.data = temp
            self.labels = tempL
            
        self.transform = transform

    def __getitem__(self, index):
        
        data = self.data[index]
        lbl = self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, lbl

        pass

    def __len__(self):
        #print(len(self.labels))
        return len(self.data)



class ADL(torch.utils.data.Dataset):
    def __init__(self, dataDir='../../../Data/HAR-ADL/', type="Train", transform=None, selIndividual=None):
        
        import csv
        import os 
        
        # Initialize the data and label list
        self.labels = []
        self.data = []
        
        #f = open(dataDir+type+"/X_t"+type[1:]+".csv", "r")
        
        print(dataDir+type+"/X_t"+type[1:]+".csv")
        
        csvFiles = dataDir+type+"/X_t"+type[1:]+".csv"
        f = open(dataDir+type+"/y_t"+type[1:]+".txt", "r")
        fId = open(dataDir+type+"/subject_id_t"+type[1:]+".txt", "r")
        
        with open(csvFiles, 'r') as file:
            
            reader = csv.reader(file)
            for row,gt,id in zip(reader,f,fId):
                
                if selIndividual!=None and not int(id) in selIndividual: 
                    pass
                else:  
                    row_c = [float(k) for k in row]
                    
                    self.data.append(row_c)
                    
                    self.labels.append([int(gt.strip())])
                
        
        f.close()
        
        self.transform = transform

    def __getitem__(self, index):
        
        data = self.data[index]
        lbl = self.labels[index]

        if self.transform is not None:
            data = self.transform(data)
            lbl = self.transform(lbl)
        
        return torch.tensor(data), torch.tensor(lbl)

        pass

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    
    
    tr = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
        ])

    
    cd = CatDog(transform=tr,crossNum=5, crossIDs=[5])
    print(len(cd))
    
    images,labels = next(iter(cd))
    print(images,labels)
    print(images.shape)
    exit(0)
    
    '''AL = ADL(selIndividual=list(range(29)))
    print(len(AL))
    z,z2 = next(iter(AL))
    print('f',z,'f2',z2)
    print(z.shape)'''
