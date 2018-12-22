import tensorflow as tf
import tensorflow.contrib.layers as c_layers
import numpy as np

from my_rl_teacher.predict_model import PredictEvaluationModel


class ScoredPredictEvaluationModel(PredictEvaluationModel):
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, param_size:int,stack_num:int, scope:str, layer_num:int, summary_writer:tf.summary.FileWriter):
        super(ScoredPredictEvaluationModel, self).__init__(param_size, stack_num, scope, layer_num, True, summary_writer)

    def override_build_loss(self, reward_logits)->tf.Tensor:
        self.scores = tf.placeholder(dtype=tf.float32, shape=(None,2), name="comparison_scores")
        self.soft_max_scores = tf.nn.softmax(self.scores, name='softmax_scores')
        return tf.losses.softmax_cross_entropy(logits=reward_logits, onehot_labels=self.soft_max_scores)

    def override_add_feed_dict(self, feed_dict:dict, labeled_comparisons_batch:list)->dict:
        scores = np.asarray([comp['score'] for comp in labeled_comparisons_batch])
        feed_dict[self.scores] = scores
        return feed_dict

