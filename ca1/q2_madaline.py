from cProfile import label 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
 
 
def activation_func(net): 
    result = [] 
    for i in range(len(net)): 
        if net[i] >= 0: 
            result.append(1) 
        else: 
            result.append(-1) 
    return result 
 
class MAdaLine: 
    def __init__(self, learning_rate=0.1, iterations=100): 
        self.learning_rate = learning_rate 
        self.iterations = iterations 
        self.activation_func = activation_func 
        self.hlayer_w = None 
        self.hlayer_b = None 
        self.and_layer_w= None 
        self.and_layer_b= None 
        self.err = [] 
 
    def fit(self, X, Y): 
        for epoch in range(self.iterations): 
            cost=0
            for i in range(len(X)): 
                net_hlayer =np.dot(self.hlayer_w, X[i])+self.hlayer_b 
                net_outlayer=np.dot(self.and_layer_w,self.activation_func(net_hlayer))+self.and_layer_b 
                y_predicted=np.where(net_outlayer >= 0, +1, -1) 
 
                # update rule 
                error=Y[i]-y_predicted   
                if error != 0 :
                    if Y[i]==1:
                        idw=np.argmin(abs(net_hlayer))
                        delta=Y[i]-net_hlayer[idw]
                        update_=np.zeros((len(self.hlayer_w),2))
                        update_[idw,0]=self.learning_rate * X[i][0]*delta
                        update_[idw,1]=self.learning_rate * X[i][1]*delta
                        self.hlayer_w += update_
                        self.hlayer_b[idw] += delta 
                    
                    elif Y[i]==-1:
                        for j in range(len(net_hlayer)):
                            if net_hlayer[j]>0:
                                delta=Y[i]-net_hlayer[j]
                                update_=np.zeros((len(self.hlayer_w),2))
                                update_[j,0]=self.learning_rate * X[j][0]*delta
                                update_[j,1]=self.learning_rate * X[j][1]*delta
                                self.hlayer_w += update_
                                self.hlayer_b[j] += delta 

                cost += 0.5*(error**2)

            self.err.append(cost)
 
    def init_paremeters(self,neuron_num): 
        random_gen1 = np.random.RandomState(10) 
        self. hlayer_w = np.random.normal(0, 0.01, (neuron_num, 2)) 
        random_gen2 = np.random.RandomState(20) 
        self.hlayer_b = np.random.normal(0, 0.01, neuron_num) 
        self.and_layer_w= np.ones((neuron_num))  
        self.and_layer_b= neuron_num-1
 
 
x1 = np.array(pd.read_csv("E:\\seventh_sem\\Neural_network\\projects\\Neural_network_course\\ca1\\MadaLine.csv", usecols = [0])) 
x2 = np.array(pd.read_csv("E:\\seventh_sem\\Neural_network\\projects\\Neural_network_course\\ca1\\MadaLine.csv", usecols = [1])) 
y = np.array(pd.read_csv("E:\\seventh_sem\\Neural_network\\projects\\Neural_network_course\\ca1\\MadaLine.csv", usecols = [2])) 
 
plt.figure()
ax1 = plt.axes()
x11=[]
x22=[]
y1=[]
y2=[]
for i in range(len(y)): 
    if y[i] == 0: 
        y1.append(x2[i])
        x11.append(x1[i])
    else:
        y2.append(x2[i]) 
        x22.append(x1[i])
ax1.scatter(x11, y1, c='g', marker='o', label='-1') 
ax1.scatter(x22, y2, c='r', marker='o', label='1') 
plt.legend(loc='upper left')
plt.title("Unclassified Data")
plt.xlabel('x1') 
plt.ylabel('x2') 
plt.show()

for i in range (len(y)):
    if y[i]== 0:
        y[i]=-1

x=np.column_stack((x1,x2))

for num_of_neurons in [3, 4, 8]:
    model=MAdaLine(0.2,200)
    model.init_paremeters(num_of_neurons)
    model.fit(x,y)

    plt.plot(model.err)
    plt.title(num_of_neurons)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.show()

    domain=np.arange(-2,2,1)
    Line1=[]
    Line2=[]
    Line3=[]
    Line4=[]
    Line5=[]
    Line6=[]
    Line7=[]
    Line8=[]
   
    for i in range (len(domain)):
        Line1.append(-domain[i]*(model.hlayer_w[0][0]/model.hlayer_w[0][1])-model.hlayer_b[0]/model.hlayer_w[0][1])
        Line2.append(-domain[i]*(model.hlayer_w[1][0]/model.hlayer_w[1][1])-model.hlayer_b[1]/model.hlayer_w[1][1])
        Line3.append(-domain[i]*(model.hlayer_w[2][0]/model.hlayer_w[2][1])-model.hlayer_b[2]/model.hlayer_w[2][1])  
        if num_of_neurons>3:
            Line4.append(-domain[i]*(model.hlayer_w[3][0]/model.hlayer_w[3][1])-model.hlayer_b[3]/model.hlayer_w[3][1])  
        if num_of_neurons==8:
            Line5.append(-domain[i]*(model.hlayer_w[4][0]/model.hlayer_w[4][1])-model.hlayer_b[4]/model.hlayer_w[4][1])  
            Line6.append(-domain[i]*(model.hlayer_w[5][0]/model.hlayer_w[5][1])-model.hlayer_b[5]/model.hlayer_w[5][1])  
            Line7.append(-domain[i]*(model.hlayer_w[6][0]/model.hlayer_w[6][1])-model.hlayer_b[6]/model.hlayer_w[6][1])  
            Line8.append(-domain[i]*(model.hlayer_w[7][0]/model.hlayer_w[7][1])-model.hlayer_b[7]/model.hlayer_w[7][1])  

    plt.figure()
    ax2 = plt.axes()
    x11=[]
    x22=[]
    y1=[]
    y2=[]
    for i in range(len(y)): 
        if y[i] == -1: 
            y1.append(x2[i])
            x11.append(x1[i])
        else:
            y2.append(x2[i]) 
            x22.append(x1[i])
    ax2.scatter(x11, y1, c='g', marker='o', label='-1') 
    ax2.scatter(x22, y2, c='r', marker='o', label='1') 
    ax2.plot(domain,Line1,'b')
    ax2.plot(domain,Line2,'b')
    ax2.plot(domain,Line3,'b')
    if num_of_neurons>3:
        ax2.plot(domain,Line4,'b')
    if num_of_neurons==8:
        ax2.plot(domain,Line5,'b')
        ax2.plot(domain,Line6,'b')
        ax2.plot(domain,Line7,'b')
        ax2.plot(domain,Line8,'b')
    plt.legend(loc='upper left')
    plt.title("Classified Data")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.ylim([-2,2])
    plt.xlim([-2,2])
    plt.show()