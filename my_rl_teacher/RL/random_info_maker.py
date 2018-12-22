import numpy as np

def make_random_info(vec_obs_size:int, act_size:int, stack_num:int):
    observation = [2*np.random.rand(vec_obs_size)-1 for x in range(stack_num)]
    action = [2*np.random.rand(act_size)-1 for x in range(stack_num)]
    reward = [2*np.random.rand()-1 for x in range(stack_num)]
    return {
        'observation':observation,
        'action':action,
        'reward':reward
        }

def make_random_info_list(data_num:int, vec_obs_size:int, act_size:int, stack_num:int):
    return [make_random_info(vec_obs_size, act_size, stack_num) for x in range(data_num)]
