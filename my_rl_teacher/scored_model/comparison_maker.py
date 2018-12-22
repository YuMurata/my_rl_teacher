from my_rl_teacher.comparison_maker import ComparisonMaker

class ScoredComparisonMaker(ComparisonMaker):
    def __init__(self, comparison_num:int, info_list:list, batch_size:int, score_func):
        super(ScoredComparisonMaker, self).__init__(comparison_num, info_list, batch_size, score_func)

    def make_comparison(self, left_info:dict, right_info:dict, score_func):
        return {
            'left':left_info,
            'right':right_info,
            'score':score_func(left_info,right_info)
        }
