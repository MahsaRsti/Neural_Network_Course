# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vP6Y5emz6FbfzV9Ibh82HbU4pDtOSKVH

**Import Libraries**
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

"""**Import Data**"""

movies_data = pd.read_csv("movies.csv")
ratings_data = pd.read_csv("ratings.csv")

"""**A.** Show 5 First and Last Data"""

# A
print(movies_data.head())
print(movies_data.tail())
print(ratings_data.head())
print(ratings_data.tail())

print('movies data shape: ', np.shape(movies_data))
print('ratings data shape: ', np.shape(ratings_data))

# A'
print([movies_data[0:5], movies_data[-5:]])
print([ratings_data[0:5], ratings_data[-5:]])

print('movies data shape: ', np.shape(movies_data))
print('ratings data shape: ', np.shape(ratings_data))

"""**B.** Merge Movies and Ratings Data"""

# B
merged_data = movies_data.merge(ratings_data, how = 'inner', on = 'movieId')
merged_data

"""**C.** Delete Extra Columns"""

# C
merged_data.drop('title', inplace = True, axis = 1)
merged_data.drop('genres', inplace = True, axis = 1)
merged_data.drop('timestamp', inplace = True, axis = 1)
merged_data

"""**D.** Group Data with GroupBy Method"""

# D
data = merged_data.groupby('userId', sort = True)
data.first()

print(type(data))
# data['movieId']
merged_data['movieId']

"""**E.** Normalized Users Ratings"""

# E
train_X = preprocessing.normalize([data['rating'].mean()])
train_X = torch.FloatTensor(train_X)

"""**F.** Create Neurons"""

# F
hidden_nueron_number = 20
visible_nueron_number = 

class RBM():
  def __init__(self, hidden_node, visible_node):
    self.weights = np.random.normal(0, 1, [hidden_node, visible_node])
    self.bias_hidden_give_visible = np.random.normal(0, 1, hidden_node)
    self.bias_visible_give_hidden = np.random.normal(0, 1, visible_node)

  def hidden_layer(self, x):
    weights_hidden_node = np.dot(x, np.transpose(self.weights))
    activation_function = weights_hidden_node + self.bias_hidden_give_visible*np.ones([np.shape(weights_hidden_node)[0], np.shape(weights_hidden_node)[1]])
    activation_function = torch.FloatTensor(activation_function)
    return torch.sigmoid(activation_function)

  def visible_layer(self, x):
    weights_visible_node = np.dot(x, self.weights)
    activation_function = weights_visible_node + self.bias_hidden_give_visible*np.ones([np.shape(weights_visible_node)[0], np.shape(weights_visible_node)[1]])
    activation_function = torch.FloatTensor(activation_function)
    return torch.sigmoid(activation_function)

  def train(self, v0, vk, ph0, phk):
    self.weights += np.dot(np.transpose(v0), ph0) - np.dot(np.transpose(vk), phk)
    self.bias_hidden_give_visible += np.sum((ph0 - phk), axis = 0)
    self.bias_visible_give_hidden += np.sum((v0 - vk), axis = 0)

"""**G.** Train Neuron"""

# G
max_epoch = 20
for epoch in range(1, max_epoch):