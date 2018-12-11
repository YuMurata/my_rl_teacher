from math import exp

def score_obs(left_info:dict, right_info:dict):
    left_score = exp(left_info['observation'][0])
    right_score = exp(right_info['observation'][0])

    return [left_score, right_score]
