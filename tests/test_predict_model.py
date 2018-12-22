import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from my_rl_teacher.predict_model import PredictEvaluationModel
import tensorflow as tf

class TestPartialModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.parameter_size = 10
        self.stack_num = 1
        self.layer_num = 3
        self.summary_writer = tf.summary.FileWriter('summaries')
        self.comparison_size = 10

    def setUp(self):
        self.score_predict_model = PredictEvaluationModel(self.parameter_size,self.stack_num, 'score', self.layer_num, True, self.summary_writer)
        self.label_predict_model = PredictEvaluationModel(self.parameter_size,self.stack_num, 'label', self.layer_num, False, self.summary_writer)

    def generate_parameter(self):
        return [x for x in range(self.parameter_size)]

    def test_generate_parameter(self):
        self.assertEqual([x for x in range(self.parameter_size)], self.generate_parameter())

    def generate_stacked_parameter(self):
        return [self.generate_parameter() for x in range(self.stack_num)]

    def test_generate_stacked_parameter(self):
        self.assertEqual([self.generate_parameter() for x in range(self.stack_num)], self.generate_stacked_parameter())

    def test_predict(self):
        info = {'parameter':self.generate_stacked_parameter()}
        self.score_predict_model.predict_reward([info])
        self.label_predict_model.predict_reward([info])

    def generate_comparison_list(self):
        comparison_list =[
            {
                'left':{'parameter':self.generate_stacked_parameter()},
                'right':{'parameter':self.generate_stacked_parameter()},
            }
            for x in range(self.comparison_size)
        ]
        return comparison_list

    def generate_scored_comparison_list(self):
        scored_comparison_list = self.generate_comparison_list()
        for i,scored_comparison in enumerate(scored_comparison_list):
            scored_comparison['score'] = [i, i]
        return scored_comparison_list

    def generate_labeled_comparison_list(self):
        labeled_comparison_list = self.generate_comparison_list()
        for i,labeled_comparison in enumerate(labeled_comparison_list):
            labeled_comparison['label'] = i%2
        return labeled_comparison_list

    def test_update(self):
        scored_comparison_list = self.generate_scored_comparison_list()
        labeled_comparison_list = self.generate_labeled_comparison_list()
        self.score_predict_model.update_model(scored_comparison_list)
        self.label_predict_model.update_model(labeled_comparison_list)


if __name__ == '__main__':
    unittest.main()