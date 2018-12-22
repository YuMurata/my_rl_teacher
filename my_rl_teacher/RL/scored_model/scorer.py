import numpy as np

def score_obs(left_info:dict, right_info:dict):
    left_score = left_info['observation'][0][0]
    right_score = right_info['observation'][0][0]

    return [left_score, right_score]
