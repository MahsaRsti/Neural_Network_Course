import numpy as np
import matplotlib.pyplot as plt

def activation_func(net):
    return np.where(net>=0,+1,-1)

class AdaLine:
    def __init__(self,learning_rate=0.1,iterations=100):
        self.learning_rate=learning_rate
        self.iterations=iterations
        self.activation_func=activation_func
        self.weights=None
        self.bias=None
        self.err=[]

    def fit(self,X,Y):
        samples, inputs= np.shape(X)
        # self.weights=np.random.rand(inputs)
        # self.bias=np.random.rand()
        self.weights=np.zeros(inputs)
        self.bias=0
        
        for _ in range(self.iterations):
            for idx, x_i in enumerate(X):
                net = np.dot(x_i, self.weights) + self.bias
                #y_predicted = self.activation_func(net)

                #update rule
                delta = self.learning_rate * (Y[idx] - net)
                self.weights += delta * x_i
                self.bias += delta

            #error
            self.err.append(0.5*((Y[idx] - net)**2))


    def predict(self, X):
        net = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(net)
        return y_predicted

#generating data
x1=np.random.normal(1,0.3,100)
y1=np.random.normal(1,0.3,100)
# print(x1)
# print(y1)
input0=np.column_stack((x1, y1))
t0=np.zeros(100)+1
print(input0)

x2=np.random.normal(-1,0.3,100)
y2=np.random.normal(-1,0.3,100)
input1=np.column_stack((x2, y2))
t1=np.zeros(100)-1
print(input1)

#plot data sets a)
plt.figure()
ax1 = plt.axes()
ax1.scatter(x1, y1, s=10, c='r', marker="o", label='class1')
ax1.scatter(x2,y2, s=10, c='b', marker="o", label='class2')
plt.legend(loc='upper left')
plt.show()


train_X=np.concatenate((input0,input1))
train_Y=np.concatenate((t0,t1))
print(train_X)

#train the model B)
l=AdaLine(0.1,50)
l.fit(train_X,train_Y)
# print(l.weights,l.bias)
# print(l.err)
# print(len(l.err))

#plot results
plt.figure(figsize=(4, 3))
ax2 = plt.axes()
ax2.scatter(x1, y1, s=10, c='r', marker="o", label='class1')
ax2.scatter(x2,y2, s=10, c='b', marker="o", label='class2')
x=np.arange(-2,4,1)
ax2.plot(x,(-l.weights[0]*x-l.bias)/l.weights[1], c='g', label='net')
plt.legend(loc='upper left')
plt.show()

#plot errors
plt.figure()
plt.plot(l.err)
plt.show()






