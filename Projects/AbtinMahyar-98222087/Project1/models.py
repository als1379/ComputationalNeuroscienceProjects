from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import math


class Strategy(ABC):
    
    @abstractmethod
    def calculate_delta(self, **kwargs) -> Tuple[float, float]:
        pass


class Default(Strategy):
    
    def calculate_delta(self, **kwargs) -> Tuple[float, float]:
        u = kwargs['u_prev']
        u_rest = kwargs['u_rest']
        R = kwargs['R']
        I = kwargs['I']
        tau = kwargs['tau']
        dt = kwargs['dt']

        return (-(u - u_rest) + (R * I)  ) / (tau) * dt, 0 


class Adaptive(Strategy):

    def calculate_delta(self, **kwargs) -> Tuple[float, float]:
        u = kwargs['u_prev']
        u_rest = kwargs['u_rest']
        R = kwargs['R']
        I = kwargs['I']
        tau = kwargs['tau']
        dt = kwargs['dt']

        w = kwargs['w']
        tau_w = kwargs['tau_w']
        a = kwargs['a']
        b = kwargs['b']
        fiers = kwargs['fiers']

        dw = (((a * (u - u_rest) - w) / tau_w) + b * fiers) * dt
        new_w = w + dw

        delta = (-(u - u_rest) + (R * I) - (R * new_w)  ) / (tau) * dt 

        return delta, new_w


class AdaptiveExponential(Strategy):

    def calculate_delta(self, **kwargs) -> Tuple[float, float]:
        u = kwargs['u_prev']
        u_rest = kwargs['u_rest']
        R = kwargs['R']
        I = kwargs['I']
        tau = kwargs['tau']
        dt = kwargs['dt']

        w = kwargs['w']
        tau_w = kwargs['tau_w']
        a = kwargs['a']
        b = kwargs['b']
        fiers = kwargs['fiers']

        delta_T = kwargs['delta_T']
        theta_rh = kwargs['theta_rh']

        dw = (((a * (u - u_rest) - w) / tau_w) + b * fiers) * dt
        new_w = w + dw

        delta = (-(u - u_rest) + (R * I) - (R * new_w) + (delta_T * (math.exp(u - theta_rh) / delta_T))) / (tau) * dt 

        return delta, new_w


class LIF:

    def __init__(self, I: Callable[[int], float],
                 strategy: Optional[Strategy] = None,
                 R: Optional[int] = 10,
                 dt: Optional[float] = 0.03125,
                 u_rest: Optional[int] = -79,
                 u_reset: Optional[int] = -65,
                 u_spike: Optional[int] = 5,
                 treshold: Optional[int] = -45,
                 tau: Optional[int] = 8,
                 tau_w: Optional[float] = 1,
                 a: Optional[float] = 0.01,
                 b: Optional[float] = 0.5,
                 delta_T: Optional[float] = 1,
                 theta_rh: Optional[int] = -55) -> None:
        
        self.I = I
        self.R = R
        self.dt = dt
        self.u_rest = u_rest
        self.u_reset = u_reset
        self.u_spike = u_spike
        self.treshold = treshold
        self.tau = tau
        self.iter = 0
        self.u = [u_rest]
        self.fires = []

        self.w = [0]
        self.tau_w = tau_w
        self.a = a
        self.b = b

        self.delta_T = delta_T
        self.theta_rh = theta_rh

        if strategy is not None:
            self.strategy = strategy
        else:
            self.strategy = Default()
    
    def execute(self, n_iter: int) -> None:
        n_iter /= self.dt
        while self.iter <= n_iter:
            delta_u, new_w = self.strategy.calculate_delta(
                                                    u_prev=self.u[self.iter],
                                                    u_rest=self.u_rest,
                                                    R=self.R,
                                                    I=self.I(self.iter * self.dt),
                                                    tau=self.tau,
                                                    dt=self.dt,
                                                    w=self.w[self.iter - len(self.fires)],
                                                    tau_w=self.tau_w,
                                                    a=self.a,
                                                    b=self.b,
                                                    fiers=len(self.fires),
                                                    delta_T=self.delta_T,
                                                    theta_rh=self.theta_rh
                                                    )
            self.w.append(new_w)

            u_new = self.u[self.iter] + delta_u
            if u_new >= self.treshold:
                self.u.append(self.treshold + self.u_spike)
                self.fires.append(self.iter)
                self.iter += 1
                u_new = self.u_reset
            
            self.u.append(u_new)    
            self.iter += 1

    def __repr__(self, I: str) -> str:
        extra = ''
        if isinstance(self.strategy, Adaptive):
            extra += f'''
                taw_w: {self.tau_w}     a: {self.a}\n
                b: {self.b}
            '''
        
        if isinstance(self.strategy, AdaptiveExponential):
            extra += f'''
                taw_w: {self.tau_w}     a: {self.a}\n
                b: {self.b}     delta_T: {self.delta_T}\n
                theta_rh: {self.theta_rh}
            '''

        return f'''
            dt: {self.dt}       R: {self.R}\n
            tau: {self.tau}     u_treshold:{self.treshold}\n
            u_rest: {self.u_rest}      u_reset: {self.u_reset}\n
            u_spike: {self.u_spike}     I: {I}\n
        ''' + extra