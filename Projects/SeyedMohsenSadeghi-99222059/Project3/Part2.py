import heapq

import numpy as np
import math


class SNN:

    def __init__(self, network_dim, neurons, tau_tag,
                 tau_dopamine, reward_score):
        self.network_dim = network_dim
        self.network_neurons = neurons
        self.connections = []
        self.tau_dopamine = tau_dopamine
        self.tau_tag = tau_tag
        self.tags = []
        self.reward_score = reward_score
        self.prepare_connections()
        self.dopamine = 0

    def prepare_connections(self):
        for i in range(0, len(self.network_dim) - 1):
            self.connections.append(np.ones((self.network_dim[i], self.network_dim[i + 1])) * 8 + np.random.rand())
            self.tags.append(np.zeros((self.network_dim[i], self.network_dim[i + 1])))

    def fit(self, x, y, dt, epoch_time, dt_minus, dt_plus, a_minus, a_plus, time_course_threshold, iterations,
            learn_time):
        time = 0
        lt = learn_time
        for iteration in range(iterations):
            index = np.random.randint(0, x.shape[0])
            epoch = 0
            while epoch < epoch_time:
                inputs = x[index]
                for layer in range(len(self.network_neurons)):
                    for i, neuron in enumerate(self.network_neurons[layer]):
                        neuron.single_step(inputs[i], time, dt)
                    if layer != len(self.network_neurons) - 1:
                        activities = self.calculate_activity_history(layer, time + dt, dt, time_course_threshold)
                        inputs = activities @ self.connections[layer]

                flag = False
                if time > lt:
                    flag = True
                    lt += learn_time
                self.learn(y, index, time + dt, dt, dt_minus, dt_plus, a_minus, a_plus, learn_time, flag)

                epoch += dt
                time += dt

    def learn(self, y, index, time, dt, dt_minus, dt_plus, a_minus, a_plus, learn_time, flag):
        for layer in range(len(self.connections)):
            for i in range(len(self.network_neurons[layer])):
                for j in range(len(self.network_neurons[layer + 1])):
                    pre_neuron = self.network_neurons[layer][i]
                    post_neuron = self.network_neurons[layer][j]
                    stdp = 0

                    if pre_neuron.spike_count != 0 and post_neuron.spike_count != 0 and post_neuron.spikes[-1] == time:
                        dt = abs(post_neuron.spikes[-1] - pre_neuron.spikes[-1])
                        if post_neuron.spikes[-1] > pre_neuron.spikes[-1]:
                            stdp += a_plus * math.exp(-(dt / dt_plus))
                        if post_neuron.spikes[-1] < pre_neuron.spikes[-1]:
                            stdp += a_minus * math.exp(-(dt / dt_minus))

                    self.tags[layer][i, j] += dt * ((-self.tags[layer][i, j] / self.tau_tag) + stdp)
                    self.connections[layer][i, j] += dt * (self.tags[layer][i, j] * self.dopamine)
                    if self.connections[layer][i, j] < 0:
                        self.connections[layer][i, j] = 0

        reward = 0
        if flag:
            actions = np.zeros_like(self.network_neurons[-1])
            for i, neuron in enumerate(self.network_neurons[-1]):
                amount = 0
                for spike in reversed(neuron.spikes):
                    if time - spike > learn_time:
                        amount += 1
                actions[i] = amount
            if np.argmax(actions) == int(y[index]):
                largest_integers = heapq.nlargest(2, list(actions))
                if largest_integers[0] != 0 and (largest_integers[0] - largest_integers[1]) / largest_integers[0] > 0.1:
                    reward += self.reward_score
                else:
                    reward -= self.reward_score
            else:
                reward -= self.reward_score
        self.dopamine += learn_time * ((-self.dopamine / self.tau_dopamine) + reward)

    def calculate_activity_history(self, layer, time, dt, time_course_threshold):
        activity_list = np.zeros(self.network_dim[layer])
        for idx in range(self.network_dim[layer]):
            activity_list[idx] = self.calculate_activity_history_single(layer, idx, time, dt, time_course_threshold)
        return activity_list

    def calculate_activity_history_single(self, layer, index, time, dt, time_course_threshold):
        neuron = self.network_neurons[layer][index]
        s = 0
        activity = 0
        while self.time_course(s) > time_course_threshold:
            if (time - s) in neuron.spikes:
                activity += self.time_course(s)
            s += dt
        return activity

    def predict(self, x, time_interval, dt, time_course_threshold):
        results = np.zeros((len(x)))
        for i in range(len(x)):
            self.reset()
            time = 0
            while time < time_interval:
                inputs = x[i]
                for layer in range(len(self.network_neurons)):
                    for k, neuron in enumerate(self.network_neurons[layer]):
                        neuron.single_step(current=inputs[k], time=time, dt=dt)
                    if layer != len(self.network_neurons) - 1:
                        activities = self.calculate_activity_history(layer, time + dt, dt, time_course_threshold)
                        inputs = activities @ self.connections[layer]
                time += dt

            if self.network_neurons[-1][0].spike_count > self.network_neurons[-1][1].spike_count:
                results[i] = 0
            else:
                results[i] = 1

        return results

    def reset(self):
        for i in range(len(self.network_neurons)):
            for neuron in self.network_neurons[i]:
                neuron.reset()

    @staticmethod
    def time_course(s):
        sigma = 7.5
        return 250 * (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(s / sigma) ** 2)
