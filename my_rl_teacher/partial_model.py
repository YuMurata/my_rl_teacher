import yaml
import json
from pprint import pprint
from my_rl_teacher.predict_model import PredictEvaluationModel
import tensorflow as tf
import numpy as np
class PartialPredictModel:
    def __init__(self, structure_file:str, use_score:bool,scope:str, summary_writer:tf.summary.FileWriter):
        with open(structure_file,'r') as f:
            self.structure = json.load(f)
        pprint(self.structure)
        print('partial: {0}'.format(len(self.structure['partial'].items())))
        self.predict_models = {}

        with tf.variable_scope(scope):
            for partial_name, index_list in self.structure['partial'].items():
                self.predict_models[partial_name]=PredictEvaluationModel(len(index_list), self.structure['stack_num'], partial_name, 3, use_score, summary_writer)
                import pdb;pdb.set_trace()

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

if __name__ == "__main__":
    partial_model = PartialPredictModel('tests/test.json',True, tf.summary.FileWriter('summary'))
