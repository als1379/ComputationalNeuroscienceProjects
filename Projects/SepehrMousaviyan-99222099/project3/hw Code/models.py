import matplotlib.pyplot as plt
import numpy as np
import random


class LIF  :

    def __init__(self, I, R=10, tau=8, theta=-45, dspike=5, U_rest=-79, U_reset=-65, total_t=30, dt=0.01) : 
        self.I = I
        self.R = R + random.randint(-10, 10)/50
        self.U_rest = U_rest 
        self.U_reset = U_reset 
        self.tau = tau
        self.theta = theta 
        self.dspike = dspike 
        self.total_t = total_t
        self.dt = dt
        self.ref_steps = 0
        self.spike_array = []
        self.Default_U()


    def Default_U(self) : 
        self.U = [self.U_reset for i in range(int(30//0.01 + 1))]
        self.U[0] = self.U_rest
        


    def Sim_Step(self, i) : 
        if(self.ref_steps == 0) :
            delta_term = (-(self.U[i-1] - self.U_rest) + (self.R * self.I[i-1])  ) / (self.tau) * self.dt 
            self.U[i] = self.U[i-1] + delta_term
            self.U[i] = min(self.U[i-1] + delta_term, self.theta)
            
            if self.U[i] >= self.theta : 
                self.spike_array.append(i)
                self.U[i] += self.dspike
                self.ref_steps += 2
        else : 
            self.ref_steps -= 1
        self.U[i] = max(self.U[i], self.U_reset)
       
        return self.U[i]
                

