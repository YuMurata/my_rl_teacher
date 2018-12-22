import tensorflow as tf
import tensorflow.contrib.layers as c_layers
import numpy as np
from collections import deque

from my_rl_teacher.predict_model import PredictEvaluationModel

import os

class PredictRLRewardModel(PredictEvaluationModel):
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, vec_obs_size:int, act_size:int,stack_num:int, scope:str, layer_num:int, use_score:bool, summary_writer:tf.summary.FileWriter):
        # Build and initialize our predictor model

        self.act_size = act_size
        self.vec_obs_size = vec_obs_size
        super(PredictRLRewardModel, self).__init__(vec_obs_size+act_size, stack_num, scope, layer_num, use_score, summary_writer)

    def predict_reward(self, segment_list):
        predict_reward = self.sess.run(self.left_predict_reward, feed_dict={
            self.left_param_placeholder: np.asarray([np.concatenate(segment['observation'], segment['action']) for segment in segment_list]),
        })
        return predict_reward[0]

    def update_model(self,labeled_comparisons_batch):

        left_param = np.asarray(
            [np.concatenate(comp['left']['observation'], comp['left']['action']) for comp in labeled_comparisons_batch])
        right_param = np.asarray(
            [np.concatenate(comp['right']['observation'], comp['right']['action']) for comp in labeled_comparisons_batch])

        feed_dict={
            self.left_param_placeholder: left_param,
            self.right_param_placeholder: right_param,
        }

        if self.use_score:
            scores = np.asarray([comp['score'] for comp in labeled_comparisons_batch])
            feed_dict[self.scores] = scores

        else:
            labels = np.asarray([comp['label'] for comp in labeled_comparisons_batch])
            feed_dict[self.labels] = labels

        _, loss, summary = self.sess.run([self.train_op, self.loss_op, self.loss_summary], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, self.sess.run(self.network_input_creator.global_step))
        self.increment_step()
        return loss




