from matplotlib.lines import lineStyles
from torch.utils.data import TensorDataset,DataLoader
from traitlets import Long
from numpy import float32
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



df=pd.read_excel("data.xlsx")
X=df[['study_hours','slept_well']]
y=df['label_pass']

x_train,x_temp,y_train,y_temp=train_test_split(X,y,test_size=0.2)
x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.4)

x_train=torch.tensor(x_train.values,dtype=torch.float32)
y_train=torch.tensor(y_train.values,dtype=torch.long)
train_loader=DataLoader(TensorDataset(x_train,y_train),batch_size=32,shuffle=True)

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,64)
        self.batchnormalization=nn.BatchNorm1d(64)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(64,2)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        x=self.fc1(x)
        x=self.batchnormalization(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.softmax(x)
        return x

model = model()

optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()

# for plot:
collect=[]
for epoch in range(7):
    epoch_loss=0
    for inputs,labels in train_loader:
        optimizer.zero_grad()
        output=model(inputs)
        loss=criterion(output,labels)
        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
    collect.append(epoch_loss/len(train_loader))
plt.plot(collect,color='blue',marker='o',linestyle='--')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()