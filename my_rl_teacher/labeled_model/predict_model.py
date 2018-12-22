import tensorflow as tf
import tensorflow.contrib.layers as c_layers
import numpy as np

from my_rl_teacher.predict_model import PredictEvaluationModel


class LabeledPredictEvaluationModel(PredictEvaluationModel):
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, param_size:int,stack_num:int, scope:str, layer_num:int, summary_writer:tf.summary.FileWriter):
        super(LabeledPredictEvaluationModel, self).__init__(param_size, stack_num, scope, layer_num, True, summary_writer)

    def override_build_loss(self, reward_logits)->tf.Tensor:
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="comparison_labels")
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reward_logits, labels=self.labels)

    def override_add_feed_dict(self, feed_dict:dict, labeled_comparisons_batch:list)->dict:
        labels = np.asarray([comp['label'] for comp in labeled_comparisons_batch])
        feed_dict[self.labels] = labels
        return feed_dict

