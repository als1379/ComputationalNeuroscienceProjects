import numpy as np
import math
import random
from copy import deepcopy

class NeuronPopulation:
    """
    this class models a neuron population
    """

    def __init__(self,population_type,connection_details:dict,neuron_list:list,time_course):
        self.population_activity=[]
        self.neuron_list=neuron_list
        self.connections=np.zeros((len(neuron_list),len(neuron_list)))
        self.connection_details=connection_details
        set_connection=eval('self.'+self.connection_details['type'])
        self.population_type=population_type
        set_connection()
        self.time_course=time_course
    
    def fully_connection(self):
        self.connections=np.ones_like(self.connections)*(self.connection_details['j']/len(self.neuron_list))
        self.inhibitory_coeff_mannager()

    def fully_ex_connection(self):
        self.connections=np.ones_like(self.connections)*(self.connection_details['j']/len(self.neuron_list))
        for i in range(len(self.neuron_list)):
            self.connections[i,i]=0
        self.inhibitory_coeff_mannager()

    def fully_gausian_connection(self):
        meu=self.connection_details['j']
        sigma=self.connection_details['sigma']
        for i in range(len(self.neuron_list)):
            for k in range(len(self.neuron_list)):
                self.connections[i,k]=np.random.normal(meu/len(self.neuron_list),sigma/len(self.neuron_list))
        self.inhibitory_coeff_mannager()

    def random_fixed_prob_connection(self):
        j=self.connection_details['j']
        prob=self.connection_details['prob']
        for i in range(len(self.neuron_list)):
            for k in range(len(self.neuron_list)):
                if np.random.rand()<=prob:
                    self.connections[i,k]=j/(prob*len(self.neuron_list))
        self.inhibitory_coeff_mannager()
    
    def random_fixed_number_connection(self):
        j=self.connection_details['j']
        number=self.connection_details['connection_number']
        for i in range(len(self.neuron_list)):
            indeces=np.random.choice(np.arange(0,len(self.neuron_list),1),number,replace=False)
            for k in indeces:
                self.connections[i,k]=j
        self.inhibitory_coeff_mannager()

    def calculate_activity_history(self,time,dt,threshold):
        activity_list=np.zeros((len(self.neuron_list),1))

        for idx in range(len(self.neuron_list)):
            activity_list[idx,0]=self.calculate_activity_history_single(idx,time,dt,threshold)
        return activity_list

    def calculate_activity_history_single(self,idx,time,dt,threshold):
        neuron=self.neuron_list[idx]
        S=0
        activity=0
        while self.time_course(S)>threshold:
            if (time-S) in neuron.spikes:
                activity+=self.time_course(S)
            S+=dt
        return activity

    def single_step(self,input_current,self_activity,time,dt,time_course_threshold):
        inputs=self.connections.dot(self_activity)
        for i,neuron in enumerate(self.neuron_list):
            neuron.single_step(input_current+inputs[i,0],time,dt)
        activity=self.calculate_activity_history(time+dt,dt,time_course_threshold)
        return activity

    def reset(self,reset_connection=False):
        self.population_activity=[]
        if reset_connection:
            set_connection=eval('self.'+self.connection_details['type'])
            set_connection()
        for neuron in self.neuron_list:
            neuron.clear_history()

    def inhibitory_coeff_mannager(self):
        if self.population_type=='inhibitory':
            self.connections=-1*self.connections

    
    def raster_plot_data(self):
        x=[]
        y=[]
        for neuron in self.neuron_list:
            id='#'+str(int(np.floor(np.random.rand()*10000000)))
            for l in neuron.spikes:
                x.append(l)
                y.append(id)
        return x,y


class Populations:
    """
    class for simulate network of neural populations
    """

    def __init__(self,population_list,connections,input_cfunc_list):
        self.population_list=population_list
        self.connections=connections
        self.input_cfunc_list=input_cfunc_list
        self.activities=[]
        self.inputs=[]
        for idx in range(len(self.population_list)):
            self.activities.append(np.zeros((len(self.population_list[idx].neuron_list),1)))

    def single_step(self,time,dt,threshold_list):
        new_activities=[]
        l=[]
        for  i in range(len(self.population_list)):
            input=0
            input+=self.input_cfunc_list[i](time)
            for k in range(len(self.population_list)):
                if k!=i:
                    input+=self.connections[i,k]*self.activities[k].sum()
            l.append(input+self.population_list[i].connections.mean()*self.activities[i].mean())
            activity=self.population_list[i].single_step(input,self.activities[i],time,dt,threshold_list[i])
            new_activities.append(activity)
        self.inputs.append(deepcopy(l))
        self.activities=new_activities
    
    def raster_plot_data(self):
        x=[]
        y=[]
        for population in self.population_list:
            result=population.raster_plot_data()
            x+=result[0]
            y+=result[1]
        return x,y

    def reset(self):
        self.inputs=[]
        self.activities=[]
        for idx in range(len(self.population_list)):
            self.activities.append(np.zeros((len(self.population_list[idx].neuron_list),1)))
        for population in self.population_list:
            population.reset()

        
                

    







