import numpy as np
import matplotlib.pyplot as plt

x1=np.random.normal(1,0.3,100)
y1=np.random.normal(1,0.3,100)

x2=np.random.normal(-1,0.3,100)
y2=np.random.normal(-1,0.3,100)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x1, y1, s=10, c='r', marker="o", label='class1')
ax1.scatter(x2,y2, s=10, c='b', marker="o", label='class2')
plt.legend(loc='upper left')
plt.show()