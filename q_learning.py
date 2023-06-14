# function to initialise 
import math
import random
from distutils.util import strtobool
import pandas as pd
import gymnasium as gym
from gym.spaces import Discrete, Box
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


possible_actions = ["increase by level 1","increase by level 2","increase by level 3","maintain",
           "decrease by level 1","decrease by level 2","decrease by level 3"]

gamma = pd.read_csv("Datasets/gamma_modified.csv")
print("hello")
pue = pd.read_csv("Datasets/pue_modified.csv")
gpu = pd.read_csv("Datasets/gpu_modified.csv")


def actions2(dataframe,input_value,column_name,level,info_name):
    current_index = dataframe.index[dataframe[column_name] == input_value][0]
    info_value = None
    new_index = current_index + level
    if new_index <=0:
        new_index = 0
    elif new_index >= dataframe.shape[0]:
        new_index = dataframe.shape[0] - 1 
    value = dataframe.loc[new_index,column_name]
    info_value = dataframe.loc[new_index,info_name]
    return [info_value,value]

# testing
# print(actions2(gpu,30,"tdp_watts",-3,"name"))


# info_action = (0,4,5,6,3) -> 7 possible actions
# component -> state component's index in the state_space
# action -> action space index
# state -> state list
# value -> is the actual value of the state_space index
# 300 -> 
# state space = [1,2,34,4]
# number of chips : 50
# config: 20
def actions1(component,action,state):
    dict_act = {0:1,1:2,2:3,3:0,4:-1,5:-2,6:-3}
    input_value = state[component]
    if component == 0:
        info_value, output_value = actions2(gamma,input_value,"Gamma",dict_act[action],"Regions")
    elif component == 1:
        info_value, output_value = actions2(pue,input_value,"PUE",dict_act[action],"Providers")
    elif component == 2: 
        info_value, output_value = actions2(gpu,input_value,"tdp_watts",dict_act[action],"name") 
    elif component == 3:
        output_value = input_value + dict_act[action]
        if output_value > 20:
            output_value = 20
        elif output_value <= 0:
            output_value = 1  # maybe change it: 1 grid search
        info_value = str(output_value)
    elif component == 4:
        # 100 -> 5,000
        # actions -> level 1 -> 100 to 101
        # 101 -> 5,050
        output_value = input_value + dict_act[action]
        if output_value > 300:
            output_value = 300
        elif output_value <= 0:
            output_value = 1  # maybe change it : 10 gpus
        info_value = str(output_value)
        # 15,000/50 
    # elif component == 5:  # batch of 5 hours-> 96 batches in total
    #     output_value = input_value + dict_act[action]
    #     if output_value > 96:
    #         output_value = 96
    #     elif output_value <= 0:
    #         output_value = 1  # maybe change it : 10 gpus
    #     info_value = str(output_value)
    return [info_value,output_value]
            
# testing
# print(actions1(3,2,[23.916,1.25,125,21,0])) 
    
# Environment Implementation 
class env():
    # start from state and then take an action to return next state and the reward in the next state
    def __init__(self, curr_state,termination_co2):
        # 7 actions can be taken 
        # self.action_space = Discrete(7)          
        self.curr_state = curr_state
        self.termination_co2 = termination_co2
        # self.info_action = (0,0,0,0,0)
        super().__init__()

    
    def carbon_emissions(self,curr_state):
        product = 1
        # state_space1 = (gamma, pue, tdp_watts, config, chips) 
        # 100 -> divided by 5
        # 15,000 -> divided by 100
        for i in range(len(curr_state)):
            product *= curr_state[i]
        product = product/(1000*1000*1000)
        return product*250*5
    
       
    def step(self,info_action):
        # state_space1 = (gamma, pue, tdp_watts, config, chips) 
        # info_action = (0,4,5,6,3)
        # state_list = [23.916,1.25,125,21,0]
        reward = 0
        prev_co2 = self.carbon_emissions(self.curr_state)
        info_list = []
        # each state component takes an action 
        state_list = list(self.curr_state)
        actions_list = list(info_action)
        # actions1(component,action,state) -> output_value, info_value
        for i in range(len(state_list)):
            
            result = actions1(i,actions_list[i],state_list)
            state_list[i] = result[1]
            info_list.append(result[0])
        
        self.curr_state = state_list
        new_state = self.curr_state
        
        new_co2 = self.carbon_emissions(new_state)
        
        old_diff = abs(self.termination_co2 - prev_co2)
        new_diff = abs(self.termination_co2 - new_co2)
        
        if new_diff <= 0.1*self.termination_co2:
            done = True
            reward += 5
        else:
            # 100 - 500 = 400 -> old diff
            # 100 - 200 = 200 -> new diff 
            # 200 - 400  = - 200 
            if new_diff - old_diff >0: 
                reward -= 1
            else:
                reward += 1
            done = False
        # info could be actions_list ?????????
        info = {0:"The recommended country is "+str(info_list[0]),
               1: "The recommended provider is "+str(info_list[1]),
               2: "The recommended GPU is "+str(info_list[2]),
               3: "The recommended number of Grid Configurations is "+str(info_list[3])+" (in batches of 5)",
               4: "The recommended number of chips is "+str(info_list[4])+ " (in batches of 50)"}
            #    5: "The recommended number of hours is "+str(info_list[5])+ " (in batches of 5)"}
        return self.curr_state, reward, done, info
        
    # difference between reset and init
    def reset(self, curr_state,termination_co2):
        self.action_space = Discrete(7)   
        self.curr_state = curr_state
        self.termination_co2 = termination_co2
#         self.info_action = (0,0,0,0,0)
        return curr_state
    




# state_space1 = (country, datacenter, gpu, config, chips) -> 5
# state_space1 = (gamma, pue, tdp_watts, config, chips) -> 5
# initialised 
#input: outouts grid configurations = 100 -> 100/5
# ouput chips = 10,000 -> 10,000/50
# 
state = [120.776,1.25,125,20,250]

# 3 tonnes -> carbon dicoxide termination state
env1 = env(state,3)

# testing
# prod = env1.step([0,4,5,6,3])
# print(prod)


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 35),
        )

    def forward(self, x):
        return self.network(x)



q_network = QNetwork()
optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
target_network = QNetwork()
state = torch.tensor(state,dtype= torch.float32)
qval = q_network(state)


# # Agent Implementation

# Values used for parameter in dqn.py via cleanRL website vs the values we used for parameters
# python dqn.py --total-timesteps 500000 \ 300
#     --learning-rate 2.5e-4 \
#     --buffer-size 10000 \
#     --gamma 0.99 \ 0.9
#     --target-network-frequency 500 \ 30
#     --max-grad-norm 0.5 \
#     --batch-size 128 \
#     --start-e 1 \
#     --end-e 0.05 \
#     --exploration-fraction 0.5 \
#     --learning-starts 10000 \ 15
#     --train-frequency 1 
#     --tau / 0.9




def concise(states):
    # tensor of 35 length
    tensor_35 = torch.tensor(states)
    # Reshape the tensor into a 2D array of size 7x5
    tensor_2d = tensor_35.reshape(5, 7)
    # Convert the tensor into a NumPy array
    array_2d = tensor_2d.numpy()
    max_indices = np.argmax(array_2d, axis=1)

    return max_indices


def concise_target(states):
    # tensor of 35 length
    tensor_35 = torch.tensor(states)
    # Reshape the tensor into a 2D array of size 7x5
    tensor_2d = tensor_35.reshape(5, 7)
    # Convert the tensor into a NumPy array
    array_2d = tensor_2d.numpy()
    max_arr = []
    for i in range(len(array_2d)):
        max_val =np.amax(array_2d[i])
        max_arr.append(max_val)
    return max_arr


# testing
# print(concise(qval))

# 5 rows and 7 columns
def rand_action():
    rand_indices = []
    for i in range(5):
        random_num = random.randint(0,6)
        rand_indices.append(random_num)
    return rand_indices

# testing
# print(rand_action())



def i_to_val(tensor_array, actions):
    # [0,1,2,3,4]
    # tensor_array 35
    # actions -> index
    tensor_35 = torch.tensor(tensor_array)
    # Reshape the tensor into a 2D array of size 7x5
    tensor_2d = tensor_35.reshape(5, 7)
    # Convert the tensor into a NumPy array
    array_2d = tensor_2d.numpy()
    old_val = []
    for i in range(len(actions)):
        old_val.append(array_2d[i][actions[i]])
    return torch.tensor(old_val,requires_grad = True)
    

#testing
# print(i_to_val(qval,[0,1,6,5,4]))




def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# Reset the environment 
def training(env,current_state):
    for global_step in range(500):
        epsilon = linear_schedule(1,0.05,0.2*500,global_step)
        # prob is more than epsilon -> best action
        q_values = None
        if epsilon < random.random():
            q_values = q_network(torch.Tensor(current_state))
            actions = concise(q_values)

        else:
            actions = rand_action()
        
        next_state, rewards, termination, infos = env.step(actions)
# Plotting, maybe????
        # training 
        # next state
        with torch.no_grad():
            target_values = target_network(torch.Tensor(next_state))
            target_actions = torch.tensor(concise_target(target_values))
            
            # concise function
            # gamma -> 0.9
            td_target = rewards + 0.9 * target_actions * (1 - float(termination))
            td_target = torch.tensor(td_target, requires_grad=True)
        
        # old_val : actions [0,1,6,5,3]
        old_val = torch.tensor(i_to_val(q_network(torch.Tensor(current_state)),actions),requires_grad=True)
#         old_val = q_network(current_state)
        
        loss = F.mse_loss(td_target, old_val)


        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # target_network_frequency - 30
        # tau - 0.9
        
        if global_step % 30 == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        0.9 * q_network_param.data + (0.1) * target_network_param.data
                    )
                
        current_state = next_state
        
    # Human Language conversion
    string_return = "The Recommendations are as follows: \n"
    for value in infos.values():
        string_return += f"""{value}\n"""
    
    # string_return += "The new carbon estimation is "+(str((env.carbon_emissions(current_state))))
    curr_carbon_emission = str((env.carbon_emissions(current_state))) + " tonnes, multiplied by the number of hours trained."
    string_return += f"""Current State: {current_state}, \nCurrent Carbon Emissions: {curr_carbon_emission}"""

    return string_return

        
#Testing for our RL agent
print(training(env1,state))