import matplotlib.pyplot as plt
import numpy as np
import random
import math

def PlotPlease(pattern, neurons, out_neurons, weights_his):
    fig, axs = plt.subplots(4, 2, figsize=(12, 8))

    o = 0
    leg_list = []
    for i in range(10) : 
        if pattern[i] == o :
            axs[0, 0].plot(neurons[i].U)
            leg_list.append(str(i) + "th neuron")
    axs[0, 0].legend(leg_list)
    axs[0, 0].set_title('Output Neuron 0')

    axs[1, 0].plot(out_neurons[o].U)
    axs[1, 0].legend(['output ' + str(o)])

    leg_list = []
    for i in range(10) : 
        if pattern[i] == o :
            axs[2, 0].plot(weights_his[i])
            leg_list.append('weight of ' + str(i) + " to " + str(o))
    axs[2, 0].legend(leg_list)

    leg_list = []
    for i in range(10) : 
        if pattern[i] == o :
            axs[3, 0].plot(neurons[i].I)
            leg_list.append(str(i) + "th I")
    axs[3, 0].legend(leg_list)



    o = 1
    leg_list = []
    for i in range(10) : 
        if pattern[i] == o :
            axs[0, 1].plot(neurons[i].U)
            leg_list.append(str(i) + "th neuron")
    axs[0, 1].legend(leg_list)
    axs[0, 1].set_title('Output Neuron 1')

    axs[1, 1].plot(out_neurons[o].U)
    axs[1, 1].legend(['output ' + str(o)])

    leg_list = []
    for i in range(10) : 
        if pattern[i] == o :
            axs[2, 1].plot(weights_his[i])
            leg_list.append('weight of ' + str(i) + " to " + str(o))
    axs[2, 1].legend(leg_list)

    leg_list = []
    for i in range(10) : 
        if pattern[i] == o :
            axs[3, 1].plot(neurons[i].I)
            leg_list.append(str(i) + "th I")
    axs[3, 1].legend(leg_list)

    plt.savefig('/users/sorou/Desktop/new', bbox_inches='tight')
    plt.show()
    