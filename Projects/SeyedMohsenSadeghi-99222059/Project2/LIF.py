import math
import matplotlib.pyplot as plt
from enum import Enum


def inp(t):
    """I(t) = 0"""
    return 0 * t


class neuron_type(Enum):
    excitatory = 1,
    inhibitory = 0


class LIF:

    def __init__(self, u_rest=-70, resistance=10, capacitance=0.8, time=100, dt=0.1, threshold=-45, u_spike=5,
                 u_reset=-65, u_start=-80, input_current=inp,
                 is_exponential=False, delta_tt=1, theta_rh=-55,
                 is_adaptive=False, tau_w=1, a=0.01, b=0.5,
                 n_t=neuron_type.excitatory):

        self.u_rest = u_rest
        self.u_reset = u_reset
        self.resistance = resistance
        self.capacitance = capacitance
        self.u_spike = u_spike
        self.time = time
        self.dt = dt
        self.threshold = threshold
        self.input_current = input_current

        self.is_exponential = is_exponential
        self.delta_tt = delta_tt
        self.theta_rh = theta_rh

        self.is_adaptive = is_adaptive
        self.tau_w = tau_w
        self.a = a
        self.b = b

        self.tau = self.resistance * self.capacitance
        self.u_t = []

        self.sigma_delta_func = 0
        self.u = u_start
        self.w = 0

        self.n_t = n_t

    def start(self):
        current_time = 0.0
        while current_time <= self.time:
            self.__fu(current_time)
            yield self.sigma_delta_func
            current_time += self.dt

    def __fu(self, current_time):
        exp_val = 0
        input_value = self.input_current(current_time)
        if self.is_exponential:
            exp_val = self.delta_tt * math.exp((self.u - self.theta_rh) / self.delta_tt)

        du = self.dt * (-(self.u - self.u_rest) + exp_val - self.resistance * self.w +
                        self.resistance * input_value) / self.tau

        if self.is_adaptive:
            dw = (self.dt / self.tau_w) * (self.a * (self.u - self.u_rest)
                                           - self.w + self.b * self.tau_w * self.sigma_delta_func)
            self.w += dw

        self.u += du
        self.sigma_delta_func = 0

        self.check_spike()

        self.u_t.append(self.u)

    def u_t_plot(self):
        plt.plot(list(map(lambda i: i * self.dt, range(len(self.u_t)))), self.u_t)
        plt.ylabel('U')
        plt.xlabel('Time')
        plt.title('U-T plot')
        plt.grid(True)
        plt.show()

    def check_spike(self):
        if self.u >= self.threshold:
            self.u_t.append(self.u + self.u_spike)
            self.u = self.u_reset
            self.sigma_delta_func = 1
