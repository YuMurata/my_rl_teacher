import numpy as np

class ComparisonMaker:
    def __init__(self, comparison_num:int, info_list:list, use_score:bool, batch_size:int, label_func):
        self.comparison_num = comparison_num
        self.batch_size = batch_size

        assert self.comparison_num >= self.batch_size, 'comparison_num is lesser than batch_size'

        self.comparison_list = self.make_comparison_list(comparison_num, info_list, use_score, label_func)

    def make_comparison(self, left_info:dict, right_info:dict, use_score:bool, score_func):
        return {
            'left':left_info,
            'right':right_info,
            'score' if use_score else 'label':score_func(left_info,right_info)
        }

    def choice_left_right_info(self, info_list:list):
        choice_info = np.random.choice(info_list, 2, False)
        return {
            'left_info':choice_info[0],
            'right_info':choice_info[1]
        }

    def make_comparison_list(self, comparison_num:int ,info_list:list, use_score:bool, score_func):
        comparison_list = [None]*comparison_num
        for i in range(comparison_num):
            choice_info = self.choice_left_right_info(info_list)
            comparison_list[i] = self.make_comparison(choice_info['left_info'], choice_info['right_info'], use_score, score_func)

        return comparison_list

    def make_comparison_batch(self):
        return np.random.choice(self.comparison_list, self.batch_size, False)




