import math


class spiking_neural_networks:
    def __init__(self, neurons, connections, dt, weight_table, delay, time):

        self.neurons = neurons

        self.connections = connections

        self.dt = dt

        self.weight_table = weight_table

        self.delay = delay

        self.time = time

        self.neuron_action = []

        for layer in self.neurons:
            t = []
            for neuron in layer:
                t.append(neuron.start())
            self.neuron_action.append(t)

    def reset(self):
        self.neuron_action = []
        for layers in self.neurons:
            t = []
            for neuron in layers:
                neuron.reset()
                t.append(neuron.start())
            self.neuron_action.append(t)

    def hard_reset(self):
        self.neuron_action = []
        for layers in self.neurons:
            t = []
            for neuron in layers:
                neuron.hard_reset()
                t.append(neuron.start())
            self.neuron_action.append(t)

    def start(self):
        time = 0
        while time < self.time:
            for j in range(len(self.neuron_action[0])):
                next(self.neuron_action[0][j])

            next_inputs = self.get_output_of_every_neuron(0, time)

            for layer in range(1, len(self.neurons)):
                for j, l_neurons in enumerate(self.neurons[layer]):
                    inp = 0

                    for weight_t in self.weight_table:
                        for i, weight in enumerate(weight_t):
                            inp += weight[j] * next_inputs[i]

                    l_neurons.input_current = lambda t: inp

                    next((self.neuron_action[layer][j]))
                next_inputs = self.get_output_of_every_neuron(layer, time)

            time += self.dt
            yield True
        yield False

    def get_output_of_every_neuron(self, layer, time):
        output = []
        for neuron in self.neurons[layer]:
            epsilon = 0
            for t in neuron.spikes:
                epsilon += self.snn_time_course(time - t - self.delay)
            output.append(epsilon)

        return output

    @staticmethod
    def snn_time_course(time, sigma=3):
        return 250 * (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(time / sigma) ** 2)
