import numpy as np

def make_comparison(left_info:dict, right_info:dict, label_func):
    return {
        'left':left_info,
        'right':right_info,
        'label':label_func(left_info,right_info)
    }

def choice_left_right_info(info_list:list):
    choice_info = np.random.choice(info_list, 2, False)
    return {
        'left_info':choice_info[0],
        'right_info':choice_info[1]
    }

def make_comparison_list(comparison_num:int ,info_list:list, label_func):
    comparison_list = [None]*comparison_num
    for i in range(comparison_num):
        choice_info = choice_left_right_info(info_list)
        comparison_list[i] = make_comparison(choice_info['left_info'], choice_info['right_info'], label_func)

    return comparison_list

    

