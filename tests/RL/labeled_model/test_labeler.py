import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import tests.path_printer
import unittest
from my_rl_teacher.labeled_model.RL.labeler import left_more_action, left_more_reward

class TestLabeler(unittest.TestCase):
    def test_more_action(self):
        test_patterns=[
            ({'action':[[0]]}, {'action':[[0]]}, 0),
            ({'action':[[0]]}, {'action':[[1]]}, 1),
            ({'action':[[1]]}, {'action':[[0]]}, 0),
            ({'action':[[1]]}, {'action':[[1]]}, 0),
        ]

        for left, right, exept in test_patterns:
            with self.subTest(left=left, right=right):
                self.assertEqual(left_more_action(left, right), exept)

    def test_more_reward(self):
        test_patterns=[
            ({'reward':[[0]]}, {'reward':[[0]]}, 0),
            ({'reward':[[0]]}, {'reward':[[1]]}, 1),
            ({'reward':[[1]]}, {'reward':[[0]]}, 0),
            ({'reward':[[1]]}, {'reward':[[1]]}, 0),
        ]

        for left, right, exept in test_patterns:
            with self.subTest(left=left, right=right):
                self.assertEqual(left_more_reward(left, right), exept)
if __name__ == '__main__':
    unittest.main()