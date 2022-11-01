import numpy as np
import matplotlib.pyplot as plt

def activation_func(net):
    return np.where(net>=0,+1,-1)

class AdaLine:
    def __init__(self,learning_rate=0.1,epochs=100):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.activation_func=activation_func
        self.weights=None
        self.bias=None
        self.losses=[]

    def fit(self,X,Y):
        samples, inputs= np.shape(X)
        self.weights=np.zeros(inputs)
        self.bias=0
        
        for epoch in range(self.epochs):
            los=0
            for idx, x_i in enumerate(X):
                net = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(net)
                #update rule
                delta = self.learning_rate * (Y[idx] - y_predicted)
                self.weights += delta * x_i
                self.bias += delta

                los+=0.5*((Y[idx] - y_predicted)**2)
            #error
            self.losses.append(los)


    def predict(self, X):
        net = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(net)
        return y_predicted

#question 1: 
for data_set in [0,1]:
    #generating data for part one and two
    if data_set==0:
        x1=np.random.normal(1,0.3,100)
        y1=np.random.normal(1,0.3,100)
        input0=np.column_stack((x1, y1))
        t0=np.zeros(100)+1

        x2=np.random.normal(-1,0.3,100)
        y2=np.random.normal(-1,0.3,100)
        input1=np.column_stack((x2, y2))
        t1=np.zeros(100)-1
    elif data_set==1:
        x1=np.random.normal(0,0.6,100)
        y1=np.random.normal(0,0.6,100)
        input0=np.column_stack((x1, y1))
        t0=np.zeros(100)+1

        x2=np.random.normal(2,0.8,20)
        y2=np.random.normal(2,0.8,20)
        input1=np.column_stack((x2, y2))
        t1=np.zeros(20)-1


    #plotting data sets 
    plt.figure()
    ax1 = plt.axes()
    ax1.scatter(x1, y1, s=10, c='r', marker="o", label='class1')
    ax1.scatter(x2,y2, s=10, c='b', marker="o", label='class2')
    plt.legend(loc='upper left')
    plt.title("Unclassified Data")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    train_X=np.concatenate((input0,input1))
    train_Y=np.concatenate((t0,t1))

    #question 2
    #train the model 
    l=AdaLine(0.1,100)
    l.fit(train_X,train_Y)

    #plotting results
    plt.figure(figsize=(4, 3))
    ax2 = plt.axes()
    ax2.scatter(x1, y1, s=10, c='r', marker="o", label='class1')
    ax2.scatter(x2,y2, s=10, c='b', marker="o", label='class2')
    x=np.arange(-2,4,1)
    ax2.plot(x,(-l.weights[0]*x-l.bias)/l.weights[1], c='g', label='net')
    plt.legend(loc='upper left')
    plt.title("Classified Data")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    #plotting errors
    plt.figure()
    plt.plot(l.losses)
    plt.title("0.5(t-net)^2")
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.show()






