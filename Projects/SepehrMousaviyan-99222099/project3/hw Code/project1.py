from matplotlib.colors import from_levels_and_colors
from numpy.lib.shape_base import kron
from models import LIF
import matplotlib.pyplot as plt
import random
import math 
import numpy as np





def STDP(t1, t2, a1, a2) :
    dw = 0
    if(t1 > t2) : 
        dw = -a2 * math.exp(-abs(t1 - t2) * 0.01)
    elif(t1 < t2) : 
        dw = a1 * math.exp(-abs(t2 - t1) * 0.01)
    print(dw)
    if dw>0 :
        dw = min(.5, dw) 
    else : 
        dw = max(-.5, dw)
    return dw



def plot_spikes(spike_list1, spike_list2, w_list, t_array, title) : 
    plt.style.use("seaborn")
    fig, axs = plt.subplots(3, 1)
    axs[0].set_title(title)
    axs[0].plot(t_array, spike_list1)
    axs[0].legend(['Pre Synaptic Neuron'])
    axs[1].plot(t_array, spike_list2)
    axs[1].legend(['Post Synaptic Neuron'])
    axs[2].plot(t_array, w_list)
    axs[2].legend(['Synaps Weight Change'])
    fig.tight_layout()
    plt.show()



I1 = np.arange(0, 30, 0.01)
I2 = np.arange(0, 30, 0.01)
I2_initial_only = np.arange(0, 30, 0.01)
T = np.arange(0, 30, 0.01)

for i in range(len(I1)) : 
    I1[i] = 10

for i in range(len(I2)) : 
    I2[i] = 0
    I2_initial_only[i] = 0

I1, I2 = np.array(I1), np.array(I2)


n1 = LIF(I=I1)
n2 = LIF(I=I2)
synapse_w = 0.5
synapse_change = np.array([0.5 for i in range(len(I1))])
n1_last_spike = None
n2_last_spike = None
learn = False

for t in range(len(I1)) : 
    pot_next = n1.Sim_Step(t)
    if(pot_next >= n1.theta) : 
        n1_last_spike = t
        n2.I[t+1:t+100] += 20 * synapse_w
        learn = True

    pot_next = n2.Sim_Step(t)
    if(pot_next >= n2.theta) :
        n2_last_spike = t 
        learn = True
    
    if(learn and (n1_last_spike != None and n2_last_spike != None)) :
        synapse_w += STDP(n1_last_spike, n2_last_spike, 50, 50)
        if synapse_w < 0 : 
            synapse_w = 0
        synapse_change[t:] = synapse_w
    learn = False

    


plot_spikes(n1.U, n2.U, synapse_change, T, "Last Spike STDP")
plot_spikes(I1, I2_initial_only, I2, T, "I Pre  &  I Post initial  &  I Post final")