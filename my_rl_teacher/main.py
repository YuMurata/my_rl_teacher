import tensorflow as tf
import numpy as np

from my_rl_teacher.predict_model_rl import PredictRLRewardModel
from my_rl_teacher.random_info_maker import make_random_info_list
from my_rl_teacher.comparison_maker import ComparisonMaker
from my_rl_teacher.labeler import left_more_action
from my_rl_teacher.scorer import score_obs
from my_deep_learning.verify.cross_validation_maker import CrossValidationMaker, Trainer
from pprint import pprint
from datetime import datetime
import os

class RLTeacherTrainer(Trainer):
    def __init__(self,train_num:int, predict_model:PredictRLRewardModel, summary_dir:str):
        super(RLTeacherTrainer, self).__init__(train_num)
        self.predict_model = predict_model
        self.summary_root_dir = summary_dir

    def train(self, train_list:list, test_index:int):
        self.predict_model.initialize_variable()
        new_summary_dir = os.path.join(self.summary_root_dir, 'train_{0}'.format(test_index))
        self.predict_model.summary_writer = tf.summary.FileWriter(new_summary_dir)
        for i in range(self.train_num):
            batch_size = 64
            mini_batch = np.random.choice(train_list, batch_size, False)
            self.predict_model.update_model(mini_batch)

    def verify(self, test_list:list)->float:
        test_length = len(test_list)
        correct_label_count = 0

        for i, test in enumerate(test_list):
            # import pdb;pdb.set_trace()
            left_predict = self.predict_model.predict_reward([test['left']])[0]
            right_predict = self.predict_model.predict_reward([test['right']])[0]
            if i == 0:
                print('left info: {0}           right info:{1} ...'.format(test['left'], test['right']))
                print('left reward: {0:.3f}     right reward: {1:.3f} ...'.format(left_predict, right_predict))

            predict_left_better = left_predict >= right_predict
            if self.predict_model.use_score:
                truth_left_better = test['score'][0] >= test['score'][1]
            else:
                truth_left_better = left_more_action(test['left'], test['right']) == 0

            if truth_left_better == predict_left_better:
                correct_label_count += 1

        accuracy = correct_label_count / test_length

        return accuracy


def main():
    vec_obs_size=10
    act_size=1
    stack_num = 1
    use_score=False
    compare_func = left_more_action
    batch_size = 100

    now_date=datetime.now()
    summary_dir='summary/{0:%Y}_{0:%m%d}/{0:%H}_{0:%M}'.format(now_date)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    summary_writer = tf.summary.FileWriter(summary_dir)

    info_list = make_random_info_list(1000, vec_obs_size, act_size, stack_num)
    comparison_maker = ComparisonMaker(1000, info_list, use_score,batch_size, compare_func)
    comparison_list = comparison_maker.comparison_list
    predict_model = PredictRLRewardModel(vec_obs_size, act_size, stack_num, scope='', layer_num=3, use_score=use_score, summary_writer=summary_writer)
    return
    trainer = RLTeacherTrainer(1000, predict_model, summary_dir)
    cross_validation = CrossValidationMaker(10, comparison_list, trainer)

    # tf.Session(graph=predict_model.graph).run(tf.global_variables_initializer())
    accuracy = cross_validation.cross_validation()
    pprint(accuracy)


if __name__ =='__main__':
    main()
