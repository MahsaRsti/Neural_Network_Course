from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x=pd.read_csv("E:\\seventh_sem\\Neural_network\\projects\\Neural_network_course\\ca1\\MadaLine.csv", usecols = [0])
y=pd.read_csv("E:\\seventh_sem\\Neural_network\\projects\\Neural_network_course\\ca1\\MadaLine.csv", usecols = [1])
z=pd.read_csv("E:\\seventh_sem\\Neural_network\\projects\\Neural_network_course\\ca1\\MadaLine.csv", usecols = [2])
# df=pd.DataFrame(df0,columns=['x','y','t'])

# print(df.info())
#plt.figure()
fig =plt.figure(figsize=(12, 12))
ax =fig.add_subplot(projection='3d')
ax.scatter(x,y,z,c='r')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

