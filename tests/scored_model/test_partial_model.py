import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import tests.path_printer
import unittest
from my_rl_teacher.scored_model.partial_model import ScoredPartialPredictModel
import tensorflow as tf

class TestPartialPredictModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.partial_model = ScoredPartialPredictModel('tests/test.json', '', 3, tf.summary.FileWriter('summaries'))

    def predict_reward(self, segment_list):
        reward_list = []
        for partial_name, index_list in self.structure['partial'].items():
            reward_list.append(self.predict_models[partial_name].predict_reward(segment_list[index_list]))

        return sum(reward_list)

    def update_model(self,labeled_comparisons_batch):
        loss_list = []
        for partial_name, index_list in self.structure['partial'].items():
            partial_comparison_batch =[{
                'left':{'parameter':comp['left']['parameter'][index_list]},
                'right':{'parameter':comp['right']['parameter'][index_list]},
                'score':comp['score'][partial_name]
                }
                for comp in labeled_comparisons_batch
            ]

            loss_list.append(self.predict_models[partial_name].update_model(partial_comparison_batch))

        return np.mean(loss_list)
