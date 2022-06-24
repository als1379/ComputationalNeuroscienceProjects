from typing import Pattern
import numpy as np
import math
import random
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from models import LIF





def STDP(tin, tout, method) :
    dw = 0
    if tin != None and tout != None : 
        dw = 50 * math.exp(-abs(tin - tout) * 0.01)
        dw = dw if method == "LTP" else -dw * 0.8
    return dw


pattern = [1, 1, 0, 0, None, None, 1, 1, 0, 0]


I = np.arange(0, 30, 0.01)
T = np.arange(0, 30, 0.01)
for i in range(len(I)) : 
        I[i] = math.sin((i)/360) * 3 + 3

# def random_sin_cal_I(x) : 
#     I = np.arange(0, 30, 0.01)
#     for i in range(len(I)) : 
#         I[i] = math.sin((i + x)/360) * 3 + 3
#     return I

def queue_cal_I(x) : 
    I = np.arange(0, 30, 0.01)
    for i in range(3000) : 
        if 300*x<i<(x+1)*300 : 
            I[i] = 15
        else :
            I[i] = 0
    #I[x-300:x] = 5
    return I



neurons = []
out_neurons = []
out_ls = [0, 0]
in_ls = [None for i in range(10)]
weights = [1 for i in range(10)] 
weights_his = [ np.array([.5 for i in range(len(T))]) for j in range(10)]



# uncomment to make it random
for i in range(10) : 
    #neurons.append(LIF(I=random_sin_cal_I(random.randint(-200, 200))))
    neurons.append(LIF(I=queue_cal_I(i)))
for i in range(2) : 
    out_neurons.append(LIF(I=np.array([0 for i in range(len(I))])))


for t in range(len(T)) : 
    for n in range(len(neurons)) : 
        neurons[n].Sim_Step(t)
        if neurons[n].U[t] > neurons[n].theta : 
            if pattern[n] != None : 
                #print("LTD")
                in_ls[n] = t
                out_neurons[pattern[n]].I[t+1:t+11] += int(150 * float(weights[n]))
                weights[n] += STDP(in_ls[n], out_ls[pattern[n]], 'LTD')
                weights[n] = 0 if weights[n] < 0 else weights[n]
                weights_his[n][t:] = weights[n]
    

    for o in range(len(out_neurons)) : 
        out_neurons[o].Sim_Step(t)
        if out_neurons[o].U[t] > out_neurons[o].theta : 
            for n in range(10) : 
                if pattern[n] == o : 
                    #print("LTP")
                    out_ls[o] = t
                    weights[n] += STDP(in_ls[n], out_ls[o], 'LTP')
                    weights[n] = 0 if weights[n] < 0 else weights[n]
                    weights_his[n][t:] = weights[n]


from Plotter import PlotPlease
PlotPlease(pattern, neurons, out_neurons, weights_his)
















