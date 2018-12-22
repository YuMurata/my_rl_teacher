import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import tests.path_printer

import unittest
from my_rl_teacher.labeled_model.predict_model import LabeledPredictEvaluationModel
import tensorflow as tf

class TestPartialModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.parameter_size = 10
        self.stack_num = 1
        self.layer_num = 3
        self.summary_writer = tf.summary.FileWriter('summaries')
        self.comparison_size = 10
        self.predict_model = LabeledPredictEvaluationModel(self.parameter_size,self.stack_num, '', self.layer_num, self.summary_writer)

    def generate_parameter(self):
        return [x for x in range(self.parameter_size)]

    def test_generate_parameter(self):
        self.assertEqual([x for x in range(self.parameter_size)], self.generate_parameter())

    def generate_stacked_parameter(self):
        return [self.generate_parameter() for x in range(self.stack_num)]

    def test_generate_stacked_parameter(self):
        self.assertEqual([self.generate_parameter() for x in range(self.stack_num)], self.generate_stacked_parameter())

    def generate_info(self):
        return {'parameter':self.generate_stacked_parameter()}

    def test_predict(self):
        info = self.generate_info()
        self.predict_model.predict_reward([info])

    def generate_comparison_list(self):
        comparison_list =[
            {
                'left':{'parameter':self.generate_stacked_parameter()},
                'right':{'parameter':self.generate_stacked_parameter()},
                'label':0
            }
            for x in range(self.comparison_size)
        ]
        return comparison_list

    def test_update(self):
        comparison_list = self.generate_comparison_list()
        self.predict_model.update_model(comparison_list)


if __name__ == '__main__':
    unittest.main()