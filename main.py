import tensorflow as tf
import numpy as np

from predict_model import PredictModel
from comparison_collectors import HumanComparisonCollector
from random_info_maker import make_random_info_list
from comparison_maker import make_comparison_list
from labeler import label_more_action
from my_deep_learning.verify.cross_validation_maker import CrossValidationMaker, Trainer
from pprint import pprint

class RLTeacherTrainer(Trainer):
    def __init__(self,train_num:int, predict_model:PredictModel):
        super(RLTeacherTrainer, self).__init__(train_num)
        self.predict_model = predict_model

    def train(self, train_list:list)->list:
        loss_list = [None]*self.train_num
        for i in range(self.train_num):
            batch_size = 64
            mini_batch = np.random.choice(train_list, batch_size, False)
            loss_list[i] = self.predict_model.update_model(mini_batch)
        print(np.mean(loss_list))
        return loss_list

    def verify(self, test_list:list)->float:
        test_length = len(test_list)
        correct_label_count = 0

        for test in test_list:
            # import pdb;pdb.set_trace()
            left_predict = self.predict_model.predict_reward([test['left']])[0]
            right_predict = self.predict_model.predict_reward([test['right']])[0]
            
            predict_label = 0 if left_predict >= right_predict else 1

            if test['label'] == predict_label:
                correct_label_count += 1

        accuracy = correct_label_count / test_length

        return accuracy


def main():
    vec_obs_size=1
    act_size=1
    stack_num = 1
    info_list = make_random_info_list(1000, vec_obs_size, act_size, stack_num)
    comparison_list = make_comparison_list(1000, info_list, label_more_action)

    sess = tf.Session()
    predict_model = PredictModel(sess, vec_obs_size, act_size, stack_num, scope='', layer_num=3)
    trainer = RLTeacherTrainer(1000, predict_model)

    cross_validation = CrossValidationMaker(10, comparison_list, trainer)

    sess.run(tf.global_variables_initializer())
    
    loss, accuracy = cross_validation.cross_validation()
    pprint(accuracy)

    # comparison_collector = HumanComparisonCollector()


if __name__ =='__main__':
    main()
