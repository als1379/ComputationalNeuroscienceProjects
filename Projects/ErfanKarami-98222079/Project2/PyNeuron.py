import numpy as np
import math



class AdaptiveMCH:
    """
    class for handeling adaptive mechansim in neuron model
    """

    def __init__(self,param_dict_list):
        self.param_dict_list=param_dict_list
    def calculate(self,u,u_rest,time,spikes,dt,reset=True):
        if reset:
            self.reset()
        sigma_w=0
        for param_dict in self.param_dict_list:
            if time in spikes:
                dw=dt*(param_dict['a']*(u-u_rest)-0.05*param_dict['w']+param_dict['b']*param_dict['tau'])/param_dict['tau']
            else:
                dw=dt*(param_dict['a']*(u-u_rest)-0.05*param_dict['w'])/param_dict['tau']
            param_dict['w']=param_dict['w']+dw
            sigma_w+=param_dict['w']*15
        return sigma_w,param_dict['w']*15

    def reset(self):
        for param_dict in self.param_dict_list:
            param_dict['w']=0

class NeuronModel():
    """
    Bace classs for neuron models 
    """
    def __init__(self,model_name='bace'):
        self.model_name=model_name
        pass

    def process(self):
        return None
    
    def frequency(self):
        return None
    
    def reset(self):
        pass

    def to_rest(self):
        pass

    def clear_history(self):
        pass

    def single_step(self):
        pass



class LIF(NeuronModel):
    """
    Leaky Integrate and Fire model model for neuron dynamic
    """
    
    def __init__(self,R,tau,u_rest,threshold,u_spike,u_reset,model_name='LIF'):
        self.history=[]
        self.spikes=[]
        self.current=[]
        self.R=R
        self.tau=tau
        self.u=u_rest
        self.u_rest=u_rest
        self.threshold=threshold
        self.u_spike=u_spike
        self.u_reset=u_reset
        super().__init__(model_name)

    def process(self,current_function,timespan,dt,reset=True):
        if reset:
            self.to_rest()
        size=math.ceil(timespan/dt)
        U=np.zeros(shape=(size,2))
        spikes=[]
        time=0
        for index in range(len(U)):
            U[index,1]=self.u
            U[index,0]=time
            # checking for spike
            if self.u>self.threshold:
                spikes.append(time)
                self.reset()
            du=dt*(-1*(self.u-self.u_rest)+1e-3*self.R*current_function(time))/self.tau
            self.u+=du
            time+=dt
        return {'voltage':U,'spikes':spikes}

    def frequency(self,current_range,timespan,dt):
        data=np.zeros(shape=(len(current_range),2))
        for index in range(len(current_range)):
            self.to_rest()
            Func=lambda x: current_range[index]
            result=self.process(Func,timespan=timespan,dt=dt)
            result=result['spikes']
            data[index,0]=current_range[index]
            if len(result)==0:
                data[index,1]=0
            elif len(result)==1:
                data[index,1]=1/timespan
            else:
                data[index,1]=(len(result)-1)/(result[-1]-result[0])
        return data


    def single_step(self,input_current,time,dt):
        du=dt*(-1*(self.u-self.u_rest)+1e-3*self.R*input_current)/self.tau
        self.u+=du
        time+=dt
        self.history.append((self.u,time))
        self.current.append((input_current,time))
        if self.u>self.threshold:
                self.spikes.append(time)
                self.reset()

    def reset(self):
        self.u=self.u_reset
    
    def to_rest(self):
        self.u=self.u_rest

    def clear_history(self):
        self.history=[]
        self.spikes=[]
        self.current=[]
        self.u=self.u_rest
        


class ELIF(NeuronModel):
    """
    Exponentioal Leaky Integrate and Fire model model for neuron dynamic
    """
    def __init__(self,R,tau,u_rest,delta_T,theta_rh,u_spike,u_reset,model_name='ELIF'):
        self.R=R
        self.tau=tau
        self.u=u_rest
        self.u_rest=u_rest
        self.thete_rh=theta_rh
        self.u_spike=u_spike
        self.u_reset=u_reset
        self.delta_T=delta_T
        super().__init__(model_name)

    def process(self,current_function,timespan,dt,reset=True):
        if reset:
            self.to_rest()
        size=math.ceil(timespan/dt)
        U=np.zeros(shape=(size,2))
        spikes=[]
        time=0
        U[0,1]=self.u
        U[0,0]=time
        for index in range(1,len(U)):
            du=dt*(-1*(self.u-self.u_rest)+self.delta_T*math.exp((self.u-self.thete_rh)/self.delta_T)+1e-3*self.R*current_function(time))/self.tau
            self.u+=du
            time+=dt
            # checking for spike
            if self.u>self.u_spike:
                spikes.append(time)
                self.reset()
            U[index,1]=self.u
            U[index,0]=time
        return {'voltage':U,'spikes':spikes}
    
    def frequency(self,current_range,timespan,dt):
        data=np.zeros(shape=(len(current_range),2))
        for index in range(len(current_range)):
            self.to_rest()
            Func=lambda x: current_range[index]
            result=self.process(Func,timespan=timespan,dt=dt)
            result=result['spikes']
            data[index,0]=current_range[index]
            if len(result)==0:
                data[index,1]=0
            elif len(result)==1:
                data[index,1]=1/timespan
            else:
                data[index,1]=(len(result)-1)/(result[-1]-result[0])
        return data
    
    def reset(self):
        self.u=self.u_reset

    def to_rest(self):
        self.u=self.u_rest
    

class QLIF(NeuronModel):
    """
    Quadratic Leaky Integrate and Fire model model for neuron dynamic
    """
    def __init__(self,R,tau,u_rest,alpha_0,u_c,u_spike,u_reset,model_name='LIF'):
        self.R=R
        self.tau=tau
        self.u=u_rest
        self.u_rest=u_rest
        self.alpha_0=alpha_0
        self.u_spike=u_spike
        self.u_reset=u_reset
        self.u_c=u_c
        super().__init__(model_name)

    def process(self,current_function,timespan,dt,reset=True):
        if reset:
            self.to_rest()
        size=math.ceil(timespan/dt)
        U=np.zeros(shape=(size,2))
        spikes=[]
        time=0
        U[0,1]=self.u
        U[0,0]=time
        for index in range(1,len(U)):
            du=dt*(self.alpha_0*(self.u-self.u_rest)*(self.u-self.u_c)+1e-3*self.R*current_function(time))/self.tau
            self.u+=du
            time+=dt
            # checking for spike
            if self.u>self.u_spike:
                spikes.append(time)
                self.reset()
            U[index,1]=self.u
            U[index,0]=time
        return {'voltage':U,'spikes':spikes}

    def frequency(self,current_range,timespan,dt):
        data=np.zeros(shape=(len(current_range),2))
        for index in range(len(current_range)):
            self.to_rest()
            Func=lambda x: current_range[index]
            result=self.process(Func,timespan=timespan,dt=dt)
            result=result['spikes']
            data[index,0]=current_range[index]
            if len(result)==0:
                data[index,1]=0
            elif len(result)==1:
                data[index,1]=1/timespan
            else:
                data[index,1]=(len(result)-1)/(result[-1]-result[0])
        return data

    def reset(self):
        self.u=self.u_reset

    def to_rest(self):
        self.u=self.u_rest

class ALIF(LIF):
    """
    Adaptive Leaky Integrate and Fire model model for neuron dynamic
    """
    def __init__(self,R,tau,u_rest,threshold,u_spike,u_reset,adaptive_mechanism,model_name='ALIF'):
        self.adaptive_mechanism=adaptive_mechanism
        super().__init__(R,tau,u_rest,threshold,u_spike,u_reset,model_name)

    def process(self,current_function,timespan,dt,reset=True):
        if reset:
            self.to_rest()
        size=math.ceil(timespan/dt)
        U=np.zeros(shape=(size,2))
        w_details=np.zeros(shape=(size,2))
        spikes=[]
        time=0
        for index in range(len(U)):
            U[index,1]=self.u
            U[index,0]=time
            # checking for spike
            if self.u>self.threshold:
                spikes.append(time)
                self.reset()
            # here 
            sigma_w,w=self.adaptive_mechanism.calculate(self.u,self.u_rest,time,spikes,dt,False)
            w_details[index,1]=w
            w_details[index,0]=time
            #
            du=dt*(-1*(self.u-self.u_rest)-(1e-3*self.R*sigma_w)+1e-3*self.R*current_function(time))/self.tau
            self.u+=du
            time+=dt
        return {'voltage':U,'spikes':spikes,'w_details':w_details}

    def frequency(self,current_range,timespan,dt):
        data=np.zeros(shape=(len(current_range),2))
        for index in range(len(current_range)):
            self.to_rest()
            Func=lambda x: current_range[index]
            result=self.process(Func,timespan=timespan,dt=dt)
            self.adaptive_mechanism.reset()
            result=result['spikes']
            data[index,0]=current_range[index]
            if len(result)==0:
                data[index,1]=0
            elif len(result)==1:
                data[index,1]=1/timespan
            else:
                data[index,1]=(len(result)-1)/(result[-1]-result[0])
        return data

    def reset(self):
        self.u=self.u_reset

    def to_rest(self):
        self.u=self.u_rest

class AELIF(ELIF):
    """
    Adaptive Exponential Leaky Integrate and Fire model model for neuron dynamic
    """
    def __init__(self, R, tau, u_rest, delta_T, theta_rh, u_spike, u_reset,adaptive_mechanism, model_name='AELIF'):
        self.adaptive_mechanism=adaptive_mechanism
        super().__init__(R, tau, u_rest, delta_T, theta_rh, u_spike, u_reset, model_name)
    
    def process(self, current_function, timespan, dt, reset=True):
        if reset:
            self.to_rest()
        size=math.ceil(timespan/dt)
        U=np.zeros(shape=(size,2))
        w_details=np.zeros(shape=(size,2))
        spikes=[]
        time=0
        U[0,1]=self.u
        U[0,0]=time
        for index in range(1,len(U)):
            sigma_w,w=self.adaptive_mechanism.calculate(self.u,self.u_rest,time,spikes,dt,False)
            w_details[index,1]=w
            w_details[index,0]=time
            du=dt*(-1*(self.u-self.u_rest)+self.delta_T*math.exp((self.u-self.thete_rh)/self.delta_T)-(1e-3*self.R*sigma_w)+1e-3*self.R*current_function(time))/self.tau
            self.u+=du
            time+=dt
            # checking for spike
            if self.u>self.u_spike:
                spikes.append(time)
                self.reset()
            U[index,1]=self.u
            U[index,0]=time
        return {'voltage':U,'spikes':spikes,'w_details':w_details}

    def frequency(self, current_range, timespan, dt):
        data=np.zeros(shape=(len(current_range),2))
        for index in range(len(current_range)):
            self.to_rest()
            Func=lambda x: current_range[index]
            result=self.process(Func,timespan=timespan,dt=dt)
            self.adaptive_mechanism.reset()
            result=result['spikes']
            data[index,0]=current_range[index]
            if len(result)==0:
                data[index,1]=0
            elif len(result)==1:
                data[index,1]=1/timespan
            else:
                data[index,1]=(len(result)-1)/(result[-1]-result[0])
        return data

    def reset(self):
        return super().reset()

    def to_rest(self):
        return super().to_rest()