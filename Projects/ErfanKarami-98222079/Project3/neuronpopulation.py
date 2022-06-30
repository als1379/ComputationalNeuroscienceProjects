import numpy as np
import math
from copy import deepcopy
from PyNeuron import LIF
import heapq

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
        self.connection_history=deepcopy(self.connections.ravel())
    
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
    

    def run_and_learn(self,learning_rule,input_current_list,time_interval,dt,dt_minus,dt_plus,a_minus,a_plus,time_course_threshold):
        learn=eval('self.'+learning_rule)
        time=0
        activity=np.zeros((len(self.neuron_list),1))
        while time<=time_interval:
            inputs=self.connections.dot(activity)
            for i,neuron in enumerate(self.neuron_list):
                neuron.single_step(input_current_list[i](time)+inputs[i,0],time,dt)
            activity=self.calculate_activity_history(time+dt,dt,time_course_threshold)
            learn(time+dt,dt_minus,dt_plus,a_minus,a_plus)
            time+=dt

    def STDP(self,time,dt_minus,dt_plus,a_minus,a_plus):
        new_connections=deepcopy(self.connections)
        for i in range(len(self.neuron_list)):
            for j in range(len(self.neuron_list)):
                if i!=j:
                    pre_neuron=self.neuron_list[j]
                    post_neuron=self.neuron_list[i]
                    # LTP
                    if len(pre_neuron.spikes)!=0 and len(post_neuron.spikes)!=0 and post_neuron.spikes[len(post_neuron.spikes)-1]==time:
                        if post_neuron.spikes[len(post_neuron.spikes)-1]> pre_neuron.spikes[len(pre_neuron.spikes)-1]:
                            dt=abs(post_neuron.spikes[len(post_neuron.spikes)-1]-pre_neuron.spikes[len(pre_neuron.spikes)-1])
                            new_connections[i,j]+=a_plus*math.exp(-(dt/dt_plus))
                    # LTD
                    if len(pre_neuron.spikes)!=0 and len(post_neuron.spikes)!=0 and pre_neuron.spikes[len(pre_neuron.spikes)-1]==time:
                        if post_neuron.spikes[len(post_neuron.spikes)-1]< pre_neuron.spikes[len(pre_neuron.spikes)-1]:
                            dt=abs(post_neuron.spikes[len(post_neuron.spikes)-1]-pre_neuron.spikes[len(pre_neuron.spikes)-1])
                            new_connections[i,j]+=a_minus*math.exp(-(dt/dt_minus))
                            if new_connections[i,j]<0:
                                new_connections[i,j]=0
        self.connection_history=np.vstack((self.connection_history,new_connections.ravel()))
        self.connections=deepcopy(new_connections)


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





class SNN:
    "class for spike neural network"

    def __init__(self,network_dim:tuple,lif_neuron_parameters:dict,time_course,connection_details:dict,tau_tag,tau_dopamine,reward_score):
        self.network_dim=network_dim
        self.lif_neuron_parameters=lif_neuron_parameters
        self.connection_details=connection_details
        self.connectivity=eval('self._'+connection_details['connection_type'])
        self.network_neurons=[]
        self.connections=[]
        self.tau_dopamine=tau_dopamine
        self.tau_tag=tau_tag
        self.tags=[]
        self.reward_score=reward_score
        self.connection_history=[]
        self.time_course=time_course
        self._create_network_neurons()
        self.connectivity()
        self.dopamine=0

    def _fully_connect(self):
        for i in range(1,len(self.network_dim)):
            self.connections.append(np.ones((self.network_dim[i-1],self.network_dim[i]))*self.connection_details['j']+np.random.rand())
            self.connection_history.append(deepcopy(self.connections[-1].ravel()))
            self.tags.append(np.zeros((self.network_dim[i-1],self.network_dim[i])))

    def _create_network_neurons(self):
        for i in range(len(self.network_dim)):
            neuron_list=[]
            for j in range(self.network_dim[i]):
                if self.lif_neuron_parameters['random']==True:
                    neuron_list.append(
                        LIF(
                            self.lif_neuron_parameters['R']+np.random.rand()*self.lif_neuron_parameters['rand_change'],
                            self.lif_neuron_parameters['tau']+np.random.rand()*self.lif_neuron_parameters['rand_change'],
                            self.lif_neuron_parameters['u_rest']+np.random.rand()*self.lif_neuron_parameters['rand_change'],
                            self.lif_neuron_parameters['threshold']+np.random.rand()*self.lif_neuron_parameters['rand_change'],
                            self.lif_neuron_parameters['u_spike']+np.random.rand()*self.lif_neuron_parameters['rand_change'],
                            self.lif_neuron_parameters['u_reset']+np.random.rand()*self.lif_neuron_parameters['rand_change']
                        )
                    )
                else:
                    neuron_list.append(
                        LIF(
                            self.lif_neuron_parameters['R'],
                            self.lif_neuron_parameters['tau'],
                            self.lif_neuron_parameters['u_rest'],
                            self.lif_neuron_parameters['threshold'],
                            self.lif_neuron_parameters['u_spike'],
                            self.lif_neuron_parameters['u_reset']
                        )
                    )
            self.network_neurons.append(deepcopy(neuron_list))


    def fit(self,x,y,dt,epoch_time,dt_minus,dt_plus,a_minus,a_plus,time_course_threshold,iterations,learn_time):
        time=0
        lt=learn_time
        for iter in range(iterations):
            index=np.random.randint(low=0,high=x.shape[0])
            epoch=0
            while epoch<epoch_time:
                inputs=x[index]
                for layer in range(len(self.network_neurons)):
                    for i,neuron in enumerate(self.network_neurons[layer]):
                        neuron.single_step(inputs[i],time,dt)
                    if layer!=len(self.network_neurons)-1:
                        activities=self.calculate_activity_history(layer,time+dt,dt,time_course_threshold)
                        inputs=activities@self.connections[layer]
                # learning
                flag=False
                if time>lt:
                    flag=True
                    lt+=learn_time
                self.learn(y,index,time+dt,dt,dt_minus,dt_plus,a_minus,a_plus,learn_time,flag)
                if flag:
                    flag=False
                epoch+=dt
                time+=dt

    def learn(self,y,index,time,dt,dt_minus,dt_plus,a_minus,a_plus,learn_time,flag):
        for layer in range(len(self.connections)):
            for i in range(len(self.network_neurons[layer])):
                for j in range(len(self.network_neurons[layer+1])):
                    pre_neuron=self.network_neurons[layer][i]
                    post_neuron=self.network_neurons[layer][j]
                    stdp=0
                    # LTP
                    if len(pre_neuron.spikes)!=0 and len(post_neuron.spikes)!=0 and post_neuron.spikes[len(post_neuron.spikes)-1]==time:
                        if post_neuron.spikes[len(post_neuron.spikes)-1]> pre_neuron.spikes[len(pre_neuron.spikes)-1]:
                            dt=abs(post_neuron.spikes[len(post_neuron.spikes)-1]-pre_neuron.spikes[len(pre_neuron.spikes)-1])
                            stdp+=a_plus*math.exp(-(dt/dt_plus))
                    # LTD
                    if len(pre_neuron.spikes)!=0 and len(post_neuron.spikes)!=0 and pre_neuron.spikes[len(pre_neuron.spikes)-1]==time:
                        if post_neuron.spikes[len(post_neuron.spikes)-1]< pre_neuron.spikes[len(pre_neuron.spikes)-1]:
                            dt=abs(post_neuron.spikes[len(post_neuron.spikes)-1]-pre_neuron.spikes[len(pre_neuron.spikes)-1])
                            stdp+=a_minus*math.exp(-(dt/dt_minus))

                    # print(f'{stdp*self.dopamine}')
                    # change c/ tags
                    self.tags[layer][i,j]+=dt*((-self.tags[layer][i,j]/self.tau_tag)+stdp)
                    self.connections[layer][i,j]+=dt*(self.tags[layer][i,j]*self.dopamine)
                    if self.connections[layer][i,j]<0:
                        self.connections[layer][i,j]=0
        
        # check for reward and update dopamine
        reward=0
        if flag:
            actions=np.zeros_like(self.network_neurons[-1])
            for i,neuron in enumerate(self.network_neurons[-1]):
                amount=0
                for spike in reversed(neuron.spikes):
                    if time-spike>learn_time:
                        amount+=1
                actions[i]=amount
            if np.argmax(actions)==int(y[index]):
                largest_integers = heapq.nlargest(2, list(actions))
                if largest_integers[0]!=0 and (largest_integers[0]-largest_integers[1])/largest_integers[0]>0.1:
                    reward+=self.reward_score
                else:
                    reward-=self.reward_score
            else :
                reward-=self.reward_score
        self.dopamine+=learn_time*((-self.dopamine/self.tau_dopamine)+reward)

        # update history
        for i in range(len(self.connection_history)):
            self.connection_history[i]=np.vstack((self.connection_history[i],self.connections[i].ravel()))


    def calculate_activity_history(self,layer,time,dt,time_course_threshold):
        activity_list=np.zeros(self.network_dim[layer])
        for idx in range(self.network_dim[layer]):
            activity_list[idx]=self.calculate_activity_history_single(layer,idx,time,dt,time_course_threshold)
        return activity_list

    def calculate_activity_history_single(self,layer,index,time,dt,time_course_threshold):
        neuron=self.network_neurons[layer][index]
        S=0
        activity=0
        while self.time_course(S)>time_course_threshold:
            if (time-S) in neuron.spikes:
                activity+=self.time_course(S)
            S+=dt
        return activity 

    def predict(self,x,time_interval,dt,time_course_threshold):
        results=np.zeros((len(x)))
        for i in range(len(x)):
            self.reset()
            time=0
            while time<time_interval:
                inputs=x[i]
                for layer in range(len(self.network_neurons)):
                    for k,neuron in enumerate(self.network_neurons[layer]):
                        neuron.single_step(inputs[k],time,dt)
                    if layer!=len(self.network_neurons)-1:
                        activities=self.calculate_activity_history(layer,time+dt,dt,time_course_threshold)
                        inputs=activities@self.connections[layer]
                time+=dt
            end_layer=self.network_neurons[-1]
            end_layer_result=np.zeros((len(end_layer)))
            for j in range(len(end_layer_result)):
                end_layer_result[j]=len(end_layer[j].spikes)
            results[i]=np.argmax(end_layer_result)
        
        return results


    def reset(self):
        for i in range(len(self.network_neurons)):
            for neuron in self.network_neurons[i]:
                neuron.clear_history()
        pass

