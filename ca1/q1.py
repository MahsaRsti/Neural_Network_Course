import numpy as np

def MP_neuron(theta, x1, x2, w1, w2, b,x3=0,w3=0):
    y=x1*w1 + x2*w2 + x3*w3 +b
    if(y>=theta):
        return 1
    else:
        return 0


a0=np.random.choice([0,1])
a1=np.random.choice([0,1])
b0=np.random.choice([0,1])
b1=np.random.choice([0,1])
theta=2

and0=MP_neuron(theta,a0,b0,1,1,0)
and1=MP_neuron(theta,a1,b0,1,1,0)
and2=MP_neuron(theta,a0,b1,1,1,0)
and3=MP_neuron(theta,a1,b1,1,1,0)
c0=MP_neuron(theta,and1,and2,1,1,0)
c1=MP_neuron(theta,and1,and2,1,1,0)
sum1=MP_neuron(theta,and1,and2,4,4,0,c0,-8)
sum2=MP_neuron(theta,c0,and3,4,4,0,c1,-8)

y0=and0
y1=sum1
y2=sum2
y3=c1

print(a1,a0)
print(b1,b0)
print(y3,y2,y1,y0)
