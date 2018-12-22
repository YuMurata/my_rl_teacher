from my_rl_teacher.comparison_maker import ComparisonMaker

class LabeledComparisonMaker(ComparisonMaker):
    def __init__(self, comparison_num:int, info_list:list, batch_size:int, label_func):
        super(LabeledComparisonMaker, self).__init__(comparison_num, info_list, batch_size, label_func)

    def make_comparison(self, left_info:dict, right_info:dict, label_func):
        return {
            'left':left_info,
            'right':right_info,
            'label':label_func(left_info,right_info)
        }
