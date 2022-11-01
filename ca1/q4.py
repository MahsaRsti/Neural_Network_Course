import torch  as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

#A
hdata=pd.read_csv("E:\\seventh_sem\\Neural_network\\projects\\Neural_network_course\\ca1\\Q4_Dataset\\houses.csv")
# print(hdata.info())

# #B
#hdata.dropna(axis='columns',inplace=True)
# invalid_rows = [index for index, row in hdata.iterrows() if row.isnull().any()]
# print(invalid_rows)

#C
# corr_matrix = hdata.corr()
# sns.heatmap(corr_matrix, annot=True)
# plt.title("correlation plot")
# plt.show()

#D
# hdata.plot.hist(column=["price"])
# plt.title("price distribution")
# plt.show()

# sns.barplot(x ='price', y ='sqft_living', data = hdata)
# plt.xlabel("price")
# plt.ylabel("sqft_living")
# plt.show()


#E
hdata.drop('id', inplace=True, axis=1) 
hdata['year'], hdata['month1'] = hdata['date'].str[:4], hdata['date'].str[4:]
hdata.drop('date', inplace=True, axis=1)

hdata['month'], hdata['trash']= hdata["month1"].str.split("T", n = 1, expand = True)
hdata.drop('month1', inplace=True, axis=1)
hdata.drop('trash', inplace=True, axis=1)


#F
train, test = train_test_split(hdata, test_size=0.2)

#G
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled=scaler.transform(test)

#H
model=nn.Sequential(
                    nn.Linear(20,100),
                    nn.ReLU(),
                    nn.Linear(100,50),
                    nn.ReLU(),
                    nn.Linear(50,1)
                    ) 

#I,J,K
learning_rate = 0.0005 
epochs = 100

targets = train_scaled[:, 0]
targets = torch.tensor(targets, dtype=torch.float32)
inputs = train_scaled[:, 1:]
inputs = torch.tensor(inputs, dtype=torch.float32)

test_targets = test_scaled[:, 0]
test_targets = torch.tensor(test_targets, dtype=torch.float32)
test_inputs = test_scaled[:, 1:]
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)

for optim_loss in [0,1]:
  if optim_loss==0:
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_func = nn.MSELoss()
  elif optim_loss==1:
    optim=torch.optim.ASGD(model.parameters(), lr = learning_rate)
    loss_func=nn.L1Loss()
    
  train_loss = []
  test_loss = []
  for epoch in range(epochs):
    optim.zero_grad()
    out = model(inputs)
    loss = loss_func(out, targets)
    loss.backward()
    train_loss.append(float(loss.item()))
    optim.step()
    # print(f'epoch',epoch)
    # print(f'train loss:',loss.item())
    with torch.no_grad():
      los=loss_func(model(test_inputs), test_targets)
      # print(f'test loss:',los.item())
      test_loss.append(float(los.item()))

  plt.figure()
  ax1 = plt.axes()
  ax1.plot(train_loss,c='b',label="Train loss")
  ax1.plot(test_loss,c='r',label='Test loss')
  plt.legend()
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.show()

# indice=np.random.choice(test_inputs.tolist(), 5)
# # indice.tolist()
# with torch.no_grad():
#   out_test=model(test_inputs[indice])
# print(test_targets[indice],    out_test)

