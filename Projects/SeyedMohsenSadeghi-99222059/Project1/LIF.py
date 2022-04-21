import math
import matplotlib.pyplot as plt
from collections import OrderedDict


def inp(t):
    """I(t) = 0"""
    return 0 * t


class LIF:

    def __init__(self, u_rest=0, resistance=1, capacitance=10, time=100, dt=0.125, threshold=5, input_current=inp,
                 is_exponential=False, delta_tt=2, theta_rh=2,
                 is_adaptive=False, tau_w=5, a=2, b=2,
                 counting=False):

        self.counting = counting
        self.u_rest = u_rest
        self.resistance = resistance
        self.capacitance = capacitance
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
        self.w_t = []

        self.sigma_delta_func = 0
        self.u = 0
        self.w = 0
        self.f_i = {}

        self.__one_round = 0

        self.__sup_title = ''

        if is_adaptive and is_exponential:
            self.__sup_title = 'Adaptive Exponential Leaky Integrate and Fire' + '\n' + \
                            'resistance: ' + str(self.resistance) + ' capacitance: ' + str(self.capacitance) + '\n' + \
                            'I: ' + str(self.input_current.__doc__).strip() + '\n' + \
                            'threshold: ' + str(self.threshold) + \
                            'theta_rh: ' + str(self.theta_rh) + \
                            'delta_T: ' + str(self.delta_tt) + \
                            'a: ' + str(self.a) + \
                            'b: ' + str(self.b) + \
                            'tw: ' + str(self.tau_w) + '\n'
        elif is_exponential:
            self.__sup_title = 'Exponential Leaky Integrate and Fire' + '\n' + \
                            'resistance: ' + str(self.resistance) + ' capacitance: ' + str(self.capacitance) + '\n' + \
                            'I: ' + str(self.input_current.__doc__).strip() + '\n' + \
                            'threshold: ' + str(self.threshold) + \
                            'theta_rh: ' + str(self.theta_rh) + \
                            'delta_T: ' + str(self.delta_tt) + '\n'
        elif is_adaptive:
            self.__sup_title = 'Adaptive Leaky Integrate and Fire' + '\n' + \
                            'resistance: ' + str(self.resistance) + ' capacitance: ' + str(self.capacitance) + '\n' + \
                            'I: ' + str(self.input_current.__doc__).strip() + '\n' + \
                            'threshold: ' + str(self.threshold) + \
                            'a: ' + str(self.a) + \
                            'b: ' + str(self.b) + \
                            'tw: ' + str(self.tau_w) + '\n'
        else:
            self.__sup_title = 'Leaky Integrate and Fire' + '\n' + \
                            'resistance: ' + str(self.resistance) + ' capacitance: ' + str(self.capacitance) + '\n' + \
                            'I: ' + str(self.input_current.__doc__).strip() + '\n' + \
                            'threshold: ' + str(self.threshold) + '\n'
        self.__start()

    def __start(self):
        current_time = 0.0
        while current_time <= self.time:
            self.__fu(current_time)
            current_time += self.dt

            if self.__one_round != 0 and self.counting:
                return

        self.f_i = OrderedDict(sorted(self.f_i.items()))

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

        if self.u >= self.threshold:
            self.u = self.u_rest
            self.sigma_delta_func = 1
            self.__one_round = current_time

        if not self.counting and input_value not in self.f_i.keys():
            def current(t=0):
                return input_value + 0 * t

            try:
                self.f_i[input_value] = 1 / LIF(u_rest=self.u_rest, resistance=self.resistance,
                                                capacitance=self.capacitance,
                                                time=self.time, dt=self.dt, threshold=self.threshold,
                                                input_current=current,
                                                is_exponential=self.is_exponential, delta_tt=self.delta_tt,
                                                theta_rh=self.theta_rh,
                                                is_adaptive=self.is_adaptive, tau_w=self.tau_w, a=self.a, b=self.b,
                                                counting=True).__one_round
            except ZeroDivisionError:
                self.f_i[input_value] = 0

        self.u_t.append(self.u)
        self.w_t.append(self.w)

    def u_t_plot(self):
        plt.plot(list(map(lambda i: i * self.dt, range(len(self.u_t)))), self.u_t)
        plt.ylabel('U')
        plt.xlabel('Time')
        plt.suptitle(self.__sup_title, va='bottom')
        plt.title('U-T plot')
        plt.grid(True)
        plt.show()

    def w_t_plot(self):
        if self.is_adaptive:
            plt.plot(list(map(lambda i: i * self.dt, range(len(self.u_t)))), self.w_t)
            plt.suptitle(self.__sup_title, va='bottom')
            plt.ylabel('w')
            plt.xlabel('Time')
            plt.title('W-T plot')
            plt.grid(True)
            plt.show()
        else:
            print("isn't adaptive")

    def f_i_plot(self):
        plt.plot(self.f_i.keys(), self.f_i.values())
        plt.suptitle(self.__sup_title, va='bottom')
        plt.ylabel('f')
        plt.xlabel('I')
        plt.title('F-I plot')
        plt.grid(True)
        plt.show()
