def label_more_reward(left_info:dict, right_info:dict):
    left_more_reward = left_info['reward'] >= right_info['reward']
    return 0 if left_more_reward else 1

def label_more_action(left_info:dict, right_info:dict):
    left_more_action = left_info['action'][0][0] >= right_info['action'][0][0]
    return 0 if left_more_action else 1
