import sys
import os
from pprint import pprint
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

pprint({'path':sys.path})

import unittest
from my_rl_teacher.scored_model.comparison_maker import ScoredComparisonMaker
import tensorflow as tf

class TestPartialModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.comparison_num = 10
        self.info_list_size = 10
        self.parameter_size = 10
        self.batch_size = 10

    def score_func(self, left, right):
        return [0, 1]

    def setUp(self):
        self.comparison_maker = ScoredComparisonMaker(self.comparison_num, [0,1], self.batch_size, self.score_func)

    def test_make_comparison(self):
        self.comparison_maker.make_comparison(dict(),dict(), self.score_func)


if __name__ == '__main__':
    unittest.main()