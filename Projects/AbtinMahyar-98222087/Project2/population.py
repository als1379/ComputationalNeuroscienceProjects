import random
from neuron import LIF
from typing import List, Optional


class Population:
    def __init__(self, neurons: List[LIF],
                 connection_ratio: Optional[float] = 0.9,
                 effect_delay: Optional[int] = 1,
                 effect_weight: Optional[float] = 3.0,
                 connection_weight: Optional[float] = 2.0,
                 connection_spikes_threshold: Optional[int] = 20) -> None:
        self.neurons = neurons
        self.effect_delay = effect_delay
        self.effect_weight = effect_weight
        self.connection_weight = connection_weight
        self.spikes_counter = 0
        self.connection_spikes_threshold = connection_spikes_threshold
        self.connected_populations = []
        self._create_connections(connection_ratio)

    def _create_connections(self, ratio: float) -> None:
        prob = int(ratio * len(self.neurons))
        for neuron in self.neurons:
            post_synapses = random.sample(self.neurons, prob)
            if neuron in post_synapses:
                post_synapses.remove(neuron)
            neuron.post_synapses = post_synapses

    def execute(self, n_iter: int, dt: Optional[float] = 0.03125) -> None:
        for i in range(int(n_iter / dt)):
            for neuron in self.neurons:
                neuron.execute(i)
                if i in neuron.fires:
                    self.spikes_counter += 1
                    for post_neuron in neuron.post_synapses:
                        post_neuron.pre_synapses_effect[i + self.effect_delay] += self.effect_weight * neuron.neuron_type.value
                    
                    if self.spikes_counter % self.connection_spikes_threshold == 0:
                        for pop in self.connected_populations:
                            for n in pop.neurons:
                                n.pre_synapses_effect[i + 1] += self.connection_weight

    def connect(self, population: 'Population') -> None:
        self.connected_populations.append(population)

