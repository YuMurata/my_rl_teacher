import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import tests.path_printer
import unittest
from my_rl_teacher.scored_model.RL.scorer import score_obs
class TestScorer(unittest.TestCase):
    def test_score_observation(self):
        test_patterns=[
            ({'observation':[[0]]}, {'observation':[[0]]}, [0, 0]),
            ({'observation':[[0]]}, {'observation':[[1]]}, [0, 1]),
            ({'observation':[[1]]}, {'observation':[[0]]}, [1, 0]),
            ({'observation':[[1]]}, {'observation':[[1]]}, [1, 1]),
        ]

        for left, right, exept in test_patterns:
            with self.subTest(left=left, right=right):
                self.assertEqual(score_obs(left, right), exept)

if __name__ == '__main__':
    unittest.main()