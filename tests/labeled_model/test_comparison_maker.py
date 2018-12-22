import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from my_rl_teacher.labeled_model.comparison_maker import LabeledComparisonMaker
import tensorflow as tf

import tests.path_printer

class TestPartialModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.comparison_num = 10
        self.info_list_size = 10
        self.parameter_size = 10
        self.batch_size = 10

    def label_func(self, left, right):
        return 0

    def test_comparison_maker(self):
        self.comparison_maker = LabeledComparisonMaker(self.comparison_num, [0,1], self.batch_size, self.label_func)

if __name__ == '__main__':
    unittest.main()