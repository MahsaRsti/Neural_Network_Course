import numpy as np

def MP_neuron(theta, x1, x2, w1, w2, b):
    y=x1*w1 + x2*w2 +b
    if(y>=theta):
        return 1
    else:
        return 0


a0=1
a1=1
b0=1
b1=0
theta=2

and0=MP_neuron(theta,a0,b0,1,1,0)
and1=MP_neuron(theta,a1,b0,1,1,0)
and2=MP_neuron(theta,a0,b1,1,1,0)
and3=MP_neuron(theta,a1,b1,1,1,0)
or0=MP_neuron(theta,and1,and2,2,2,0)
cin0=MP_neuron(theta,and1,and2,1,1,0)
or1=MP_neuron(theta,cin0,and3,2,2,0)
and4=MP_neuron(theta,cin0,and3,1,1,0)

y0=and0
y1=or0
y2=or1
y3=and4

