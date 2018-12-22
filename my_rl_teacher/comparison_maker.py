import numpy as np
from .exception import ComparisonMakerException

class ComparisonMaker:
    def __init__(self, comparison_num:int, info_list:list, batch_size:int, func):
        self.comparison_num = comparison_num
        self.batch_size = batch_size

        assert self.comparison_num >= self.batch_size, 'comparison_num is lesser than batch_size'

        self.comparison_list = self.make_comparison_list(comparison_num, info_list, func)


        choice_info = np.random.choice(info_list, 2, False)
        return {
            'left_info':choice_info[0],
            'right_info':choice_info[1]
        }

    def make_comparison_list(self, comparison_num:int ,info_list:list, func)->list:
        comparison_list = [None]*comparison_num
        for i in range(comparison_num):
            choice_info = self.choice_left_right_info(info_list)
            comparison_list[i] = self.make_comparison(choice_info['left_info'], choice_info['right_info'], func)

        return comparison_list

    def make_comparison_batch(self)->list:
        return np.random.choice(self.comparison_list, self.batch_size, False)




