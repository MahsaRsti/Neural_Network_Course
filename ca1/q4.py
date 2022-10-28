import torch  as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#A
hdata=pd.read_csv("E:\\seventh_sem\\Neural_network\\projects\\Neural_network_course\\ca1\\Q4_Dataset\\houses.csv",index_col='id')
# print(hdata.info())

# #B
# invalid_rows = [index for index, row in hdata.iterrows() if row.isnull().any()]
# print(invalid_rows)

#C
#corr_matrix = hdata.corr()
# sns.heatmap(corr_matrix, annot=True)
# plt.show()

#D
#hdata.plot.hist(column=["price"])
#plt.show()

# sns.barplot(x ='price', y ='sqft_living', data = hdata)
# plt.show()


#E
hdata['year'], hdata['month1'] = hdata['date'].str[:4], hdata['date'].str[4:]
hdata.drop('date', inplace=True, axis=1)

hdata['month'], hdata['trash']= hdata["month1"].str.split("T", n = 1, expand = True)
hdata.drop('month1', inplace=True, axis=1)
hdata.drop('trash', inplace=True, axis=1)
#print(hdata)

#F
train, test = train_test_split(hdata, test_size=0.2)
# print(train)
# print("//////////////////////////////////////////////////////////")
# print(test)

#G
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled=scaler.transform(test)

# print("//////////////////////////////////////////////////////////......................")
# print(train_scaled)
# print("//////////////////////////////////////////////////////////")
# print(test_scaled)