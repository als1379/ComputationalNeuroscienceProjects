import math

from NeuronGroups import NeuronsGroup as nG
import matplotlib.pyplot as plt


class spike_time_dependant_learning:
    def __init__(self, neurons_group: nG, amplify_val_pos, amplify_val_neg, delta_tau_pos, delta_tau_neg):
        self.neurons_gp = neurons_group

        self.amplify_val_pos = amplify_val_pos
        self.amplify_val_neg = amplify_val_neg

        self.delta_tau_pos = delta_tau_pos
        self.delta_tau_neg = delta_tau_neg

        self.w_history = []

        self.neurons_gp.reset()

    def train(self):
        func = self.neurons_gp.start()
        while next(func):
            temp = []
            for w1 in self.neurons_gp.weight_table:
                temp2 = []
                for w2 in w1:
                    temp2.append(w2)
                temp.append(temp2)
            self.w_history.append(temp)

            for i in range(len(self.neurons_gp.neurons)):
                for j in self.neurons_gp.connections[i]:

                    delta_t = self.neurons_gp.neurons[j].last_spike - self.neurons_gp.neurons[i].last_spike
                    delta_w = 0

                    if self.neurons_gp.neurons[j].last_spike != -1 or self.neurons_gp.neurons[i].last_spike != -1:
                        if delta_t < 0:
                            delta_w = self.amplify_val_neg * math.exp(math.fabs(delta_t)/self.delta_tau_neg)
                        elif delta_t > 0:
                            delta_w = self.amplify_val_pos * math.exp(math.fabs(delta_t)/self.delta_tau_pos)

                    self.neurons_gp.weight_table[i][j] += delta_w

                    if self.neurons_gp.weight_table[i][j] <= 0:
                        self.neurons_gp.weight_table[i][j] = 0

    def w_plot(self, i, j):
        w_t = []
        for table in self.w_history:
            w_t.append(table[i][j])
        plt.plot(list(map(lambda k: k * self.neurons_gp.neurons[i].dt, range(len(w_t)))), w_t)
        plt.ylabel('Weight between neurons ' + str(i) + ' and' + str(j))
        plt.xlabel('Time')
        plt.title('W-T plot')
        plt.grid(True)
        plt.show()
