import math
from typing import List, Optional
from neuron import LIF
from dataclasses import dataclass, field


@dataclass
class Connection:
    origin: LIF
    dest: LIF
    weight: float
    weight_history: List = field(default_factory=list)
    delta_w_history: List = field(default_factory=list)
    iter: int = 0


class STDP:
    def __init__(self, neurons: List[LIF],
                 connection_matrix: List[List[float]],
                 a_plus: Optional[float] = 8.0,
                 a_minus: Optional[float] = 6.0,
                 tau_plus: Optional[float] = 45,
                 tau_minus: Optional[float] = 45,
                 effect_delay: Optional[int] = 1,
                 dt: Optional[float] = 0.03125) -> None:
        self.neurons = neurons
        self.connection_matrix = connection_matrix
        self.a_minus = a_minus
        self.a_plus = a_plus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.effect_delay = effect_delay
        self.dt = dt
        self._create_connections()

    def _create_connections(self) -> None:
        for i in range(len(self.neurons)):
            for j in range(len(self.neurons)):
                val = self.connection_matrix[i][j]
                if val != 0:
                    connection = Connection(self.neurons[i], self.neurons[j], val)
                    self.neurons[i].connections.append(connection)
                    self.neurons[j].connections.append(connection)

    def update_weight(self, connection: Connection):
        pre, post  = connection.origin, connection.dest
        if pre.fires and post.fires:
            t_pre, t_post = pre.fires[-1], post.fires[-1]
            if t_pre < t_post:
                delta_w = self.a_plus * math.exp(-abs(t_pre - t_post) / self.tau_plus)
            elif t_pre > t_post:
                delta_w = -self.a_minus * math.exp(-abs(t_post - t_pre) / self.tau_minus) 
            else:
                delta_w = 0

            connection.delta_w_history.append((t_pre - t_post, delta_w))

            for _ in range(connection.iter, max(t_pre, t_post) + 1):
                connection.weight_history.append(connection.weight)
            connection.iter = max(t_pre, t_post) + 1
            connection.weight = max(0, connection.weight + delta_w)
            
    def execute(self, n_iter: int):
        iter = 0
        while iter <= int(n_iter / self.dt):
            for neuron in self.neurons:
                neuron.execute(iter)
                if iter in neuron.fires:
                    for connection in neuron.connections:
                        if neuron == connection.origin:
                            connection.dest.pre_synapses_effect[iter + self.effect_delay] += connection.weight 
                        self.update_weight(connection)
            iter += 1

        for neuron in self.neurons:
            for connection in neuron.connections:
                for _ in range(int(n_iter / self.dt) - len(connection.weight_history)):
                    connection.weight_history.append(connection.weight)


