from typing import List, Optional
from neuron import LIF
from dataclasses import dataclass, field
import math
import random


@dataclass
class Connection:
    origin: LIF
    dest: LIF
    weight: float
    enzyme_importance: float = 0
    c_history : List = field(default_factory=list)
    weight_history: List = field(default_factory=list)
    delta_w_history: List = field(default_factory=list)
    iter_w: int = 0
    iter_c: int = 0


class SNN:
    def __init__(self, layers: List[List[LIF]],
                 default_weight: Optional[float] = 5,
                 dt: Optional[float] = 0.03125,
                 effect_delay: Optional[int] = 1,
                 time_window: Optional[int] = 100,
                 tau_c: Optional[float] = 0.15,
                 dopamine: Optional[float] = 0,
                 reward: Optional[float] = 5,
                 tau_d: Optional[float] = 3.5,
                 a_plus: Optional[float] = 8,
                 a_minus: Optional[float] = 6,
                 tau_plus: Optional[float] = 10,
                 tau_minus: Optional[float] = 6) -> None:
        self.layers = layers
        self.default_weight = default_weight
        self.dt = dt
        self.effect_delay = effect_delay
        self.time_window = time_window
        self.tau_c = tau_c
        self.tau_d = tau_d
        self.a_minus = a_minus
        self.a_plus = a_plus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.dopamine = dopamine
        self.reward = reward
        self.dopamine_history = []
        self.iter = 0
        self._create_fully_connections()

    def _create_fully_connections(self):
        for index in range(1, len(self.layers)):
            for pre_neuron in self.layers[index - 1]:
                for post_neuron in self.layers[index]:
                    connection = Connection(pre_neuron, post_neuron,
                                            weight=random.uniform(self.default_weight - .1, self.default_weight + .1))
                    pre_neuron.connections.append(connection)
                    post_neuron.connections.append(connection)

    def update_weights(self, connection: Connection):
        pre, post  = connection.origin, connection.dest
        if pre.fires and post.fires:
            t_pre, t_post = pre.fires[-1], post.fires[-1]
            if t_pre < t_post:
                stdp = self.a_plus * math.exp(-abs(t_pre - t_post) / self.tau_plus) 
            elif t_pre > t_post:
                stdp = -self.a_minus * math.exp(-abs(t_post - t_pre) / self.tau_minus) 
            else:
                stdp = 0

            delta_c = (-connection.enzyme_importance/self.tau_c + stdp) * self.dt
            for _ in range(connection.iter_c, max(t_pre, t_post) + 1):
                connection.c_history.append(connection.enzyme_importance)
            connection.iter_c = max(t_pre, t_post) + 1
            connection.enzyme_importance = connection.enzyme_importance + delta_c

            delta_w = (connection.enzyme_importance * self.dopamine) * self.dt 
            for _ in range(connection.iter_w, max(t_pre, t_post) + 1):
                    connection.weight_history.append(connection.weight)
            connection.iter_w = max(t_pre, t_post) + 1
            connection.weight = max(0, connection.weight + delta_w)

    def update_dopamine(self, y, max_iter: int):
        max_fires = -1 
        y_hat = -1
        for index, neuron in enumerate(self.layers[-1]):
            fires = 0
            for iter in range(max_iter - int(self.time_window / self.dt), max_iter):
                if iter in neuron.fires:
                    fires += 1
            
            if fires > max_fires:
                y_hat = index
                max_fires = fires

        DA = self.reward
        if y != y_hat:
            DA *= -1

        delta_d = (-self.dopamine / self.tau_d + DA) * (self.time_window * self.dt) * 0.1
        self.dopamine = self.dopamine + delta_d
        self.dopamine_history.append(self.dopamine)

    def fit(self, X, y, epochs: int, n_iter: int):
        for epoch in range(epochs):
            for index_case, case in enumerate(X):
                for index, neuron in enumerate(self.layers[0]):
                    neuron.I_inp = case[index]

                for iter in range(int(n_iter / self.dt)):
                    for layer in self.layers:
                        for neuron in layer:
                            neuron.execute(self.iter)
                            if self.iter in neuron.fires:
                                for connection in neuron.connections:
                                    if neuron == connection.origin:
                                        connection.dest.pre_synapses_effect[self.iter + self.effect_delay] += connection.weight
                                    self.update_weights(connection)
                
                    if self.iter % (int(self.time_window / self.dt)) == 0 and self.iter != 0:
                        self.update_dopamine(y[index_case], self.iter)
                    self.iter += 1

    def predict(self, X):
        res = []

        for case in X:
            for index, neuron in enumerate(self.layers[0]):
                neuron.I_inp = case[index]
    
            for iter in range(int(self.time_window / self.dt)):
                for layer in self.layers:
                    for neuron in layer:
                        neuron.execute(self.iter)
                        if self.iter in neuron.fires:
                            for connection in neuron.connections:
                                if neuron == connection.origin:
                                    connection.dest.pre_synapses_effect[self.iter + self.effect_delay] += connection.weight         
                self.iter += 1

            max_fires = -1
            y_hat = -1
            for index, neuron in enumerate(self.layers[-1]):
                fires = 0
                for iter in range(self.iter - int(self.time_window / self.dt), self.iter + 1):
                    if iter in neuron.fires:
                        fires += 1
                
                if fires > max_fires:
                    y_hat = index
                    max_fires = fires 
            
            res.append(y_hat)
        return res		
		
        