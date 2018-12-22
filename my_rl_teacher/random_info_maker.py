import numpy as np

def make_random_info(param_size:int, stack_num:int):
    parameter = [2*np.random.rand()-1 for x in range(stack_num)]
    return {'parameter':parameter}

def make_random_info_list(data_num:int, param_size:int, stack_num:int):
    return [make_random_info(param_size, stack_num) for x in range(data_num)]
