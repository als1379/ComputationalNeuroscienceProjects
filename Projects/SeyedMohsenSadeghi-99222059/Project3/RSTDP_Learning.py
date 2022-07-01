import math
import random


class reward_spike_time_dependant_learning:
    def __init__(self, snn, tau_tag, tau_dope, reward_score, iterations, delta_tau_pos,
                 delta_tau_neg, a_pos, a_neg, train_time, dt):

        self.snn = snn

        self.tau_tag = tau_tag
        self.tau_dope = tau_dope

        self.reward_score = reward_score

        self.iterations = iterations

        self.delta_tau_pos = delta_tau_pos
        self.delta_tau_neg = delta_tau_neg

        self.amplify_val_pos = a_pos
        self.amplify_val_neg = a_neg

        self.dopamine = 1

        self.tt = train_time

        self.dt = dt

        self.tags = []

        for layers in self.snn.connections:
            temp = []

            for level1 in layers:
                temp2 = []

                for _ in level1:
                    temp2.append(0)

                temp.append(temp2)

            self.tags.append(temp)

    def train(self, inputs, winners):

        for _ in range(self.iterations):
            w = self.fix_inputs(inputs, winners)
            lt = self.tt

            self.snn.reset()
            a = self.snn.start()

            while next(a):
                flag = False
                time = self.snn.neurons[0][0].current_time
                if time > lt:
                    lt += self.tt
                    flag = True
                self.learn(lt, flag, time, w)

            print(_)
            print(self.snn.connections)
            print(self.snn.neurons[1][0].spike_count)
            print(self.snn.neurons[1][1].spike_count)
            print(self.dopamine)
            print(self.tags)

    def learn(self, lt, flag, time, w):
        for layer in range(len(self.snn.neurons) - 1):
            for i in range(len(self.snn.neurons[layer])):
                for j in self.snn.connections[layer][i]:
                    k = int(j)
                    delta_t = self.snn.neurons[layer][i].last_spike - self.snn.neurons[layer + 1][k].last_spike
                    delta_w = 0

                    if self.snn.neurons[layer][i].last_spike != -1 and self.snn.neurons[layer + 1][k].last_spike != 0:
                        if delta_t < 0:
                            delta_w = self.amplify_val_neg * math.exp(-math.fabs(delta_t) / self.delta_tau_neg)
                        elif delta_t > 0:
                            delta_w = self.amplify_val_pos * math.exp(-math.fabs(delta_t) / self.delta_tau_pos)

                    if self.snn.neurons[layer][i].sigma_delta_func == 0 and\
                            self.snn.neurons[layer+1][k].sigma_delta_func == 0:
                        delta_w = 0

                    self.tags[layer][i][k] += -self.tags[layer][i][k]/self.tau_tag + delta_w

                    self.snn.connections[layer][i][k] += self.tags[layer][i][k] * self.dopamine

                    if self.snn.connections[layer][i][k] < 0:
                        self.snn.connections[layer][i][k] = 0
        reward = 0

        if flag:
            arr = []
            for neuron in self.snn.neurons[-1]:
                amount = 0
                for spike_t in reversed(neuron.spikes):
                    if time - spike_t >= self.tt:
                        amount += 1
                arr.append(amount)
            # print(arr)
            arr2 = sorted(arr)

            if arr2[-1] != 0 and arr2[-1] == arr[w] and (arr2[-1] - arr2[-2]) / arr2[-1] > 0.1:
                reward += self.reward_score
            else:
                reward -= self.reward_score

            self.dopamine += ((-math.fabs(self.dopamine) / self.tau_dope) + reward)

    def predict(self, inputs, time, dt):
        self.snn.hard_reset()
        for i, neuron in enumerate(self.snn.neurons[0]):
            neuron.input_current = lambda t: inputs[i]
        self.snn.time = time
        self.snn.dt = dt
        a = self.snn.start()
        while next(a):
            pass
        arr = []
        for neuron in self.snn.neurons[-1]:
            arr.append(neuron.spike_count)

        return arr

    def fix_inputs(self, inputs, winner):
        rand = random.randint(0, len(inputs) - 1)
        for i, neuron in enumerate(self.snn.neurons[0]):
            neuron.input_current = lambda x: int(inputs[rand][i])
        return int(winner[rand])
