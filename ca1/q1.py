import numpy as np

def MP_neuron(theta, x1, x2, w1, w2, b):
    y=x1*w1 + x2*w2 +b
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
z11=MP_neuron(theta,and1,and2,2,-1,0)
z12=MP_neuron(theta,and1,and2,-1,2,0)
y1=MP_neuron(theta,z11,z12,2,2,0)
and4=MP_neuron(theta,and1,and2,1,1,0)
z21=MP_neuron(theta,and4,and3,2,-1,0)
z22=MP_neuron(theta,and4,and3,-1,2,0)
y2=MP_neuron(theta,z21,z22,2,2,0)
y3=MP_neuron(theta,and4,and3,1,1,0)
y0=and0

print(a1,a0)
print(b1,b0)
print(y3,y2,y1,y0)
