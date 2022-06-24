import random

import matplotlib.pyplot as plt
from LIF import LIF
from LIF import neuron_type


class NeuronsGroup:
    def __init__(self, neurons, connections, weight_table, delay=1, iteration_count=1000):
        self.neurons = neurons
        self.neuron_action = []
        self.connections = connections
        self.weight_table = weight_table

        self.delay = delay

        self.iteration_count = iteration_count

        self.spikes = []

        self.excitatory_spike_times = []
        self.excitatory_spikes = []

        self.inhibitory_spike_times = []
        self.inhibitory_spikes = []

        self.spikes_effect = [[0] * len(self.neurons) for _ in range(self.iteration_count)]

        for neuron in neurons:
            self.neuron_action.append(neuron.start())

    def start(self):
        for time in range(self.iteration_count):
            for i in range(len(self.neuron_action)):
                sigma_delta_func = next(self.neuron_action[i])
                if sigma_delta_func == 1:
                    if self.neurons[i].n_t is neuron_type.excitatory:
                        for j in self.connections[i]:
                            self.excitatory_spikes.append(i + 1)
                            self.excitatory_spike_times.append(time)
                            if time + self.delay < self.iteration_count:
                                self.spikes_effect[time + self.delay][j] += self.weight_table[i][j]
                    else:
                        for j in self.connections[i]:
                            self.inhibitory_spikes.append(i + 1)
                            self.inhibitory_spike_times.append(time)
                            if time + self.delay < self.iteration_count:
                                self.spikes_effect[time + self.delay][j] -= self.weight_table[i][j]

            for i in range(len(self.neurons)):
                self.neurons[i].u += self.spikes_effect[time][i]
            yield True
        yield False

    def u_plot(self, neurons_count):
        legend = []
        for i in range(min(len(self.neurons), neurons_count)):
            plt.plot(list(map(lambda j: j * self.neurons[i].dt, range(len(self.neurons[i].u_t)))), self.neurons[i].u_t)
            legend.append('neuron ' + str(i + 1))
        plt.legend(legend)

    def raster_plot(self):
        plt.scatter(self.excitatory_spike_times, self.excitatory_spikes, color='blue', s=10)
        plt.scatter(self.inhibitory_spike_times, self.inhibitory_spikes, color='red', s=10)
        plt.legend(['blue:Excitatory', 'red:Inhibitory'])

    def reset(self):
        self.spikes = []

        self.excitatory_spike_times = []
        self.excitatory_spikes = []

        self.inhibitory_spike_times = []
        self.inhibitory_spikes = []

        self.spikes_effect = [[0] * len(self.neurons) for _ in range(self.iteration_count)]

        self.neuron_action = []

        for neuron in self.neurons:
            neuron.reset()
            self.neuron_action.append(neuron.start())

    @staticmethod
    def create_neuron_group(neurons_count, excitatory_count, inhibitory_count,
                            excitatory_probability, inhibitory_probability, input_current, **kwargs):
        neurons = []
        connections = []
        exc_neuron_conn_count = int(neurons_count * excitatory_probability)
        inh_neuron_conn_count = int(neurons_count * inhibitory_probability)

        for i in range(excitatory_count):
            args = {}
            if 'Excitatory' in kwargs.keys():
                for arg in kwargs['Excitatory']:
                    args[arg] = kwargs[arg][i]
            neuron = LIF(input_current=input_current[i], n_t=neuron_type.excitatory, **args)
            neurons.append(neuron)
            connections.append(random.sample(range(neurons_count), exc_neuron_conn_count))
        for i in range(inhibitory_count):
            args = {}
            if 'Inhibitory' in kwargs.keys():
                for arg in kwargs['Inhibitory']:
                    args[arg] = kwargs[arg][i]
            neuron = LIF(input_current=input_current[i], n_t=neuron_type.inhibitory, **args)
            neurons.append(neuron)
            connections.append(random.sample(range(neurons_count), inh_neuron_conn_count))

        return NeuronsGroup(neurons, connections, **kwargs)
