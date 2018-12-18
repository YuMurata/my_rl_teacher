import tensorflow as tf
import tensorflow.contrib.layers as c_layers
import numpy as np

from my_deep_learning.model.network_input import NetworkInputCreator


class PredictEvaluationModel():
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, param_size:int,stack_num:int, scope:str, layer_num:int, use_score:bool, summary_writer:tf.summary.FileWriter):
        # Build and initialize our predictor model

        self.param_size = param_size
        self.layer_num = layer_num
        self.param_shape = (self.param_size,)
        self.use_score = use_score
        self.build_model(scope)
        self.initialize_variable()

        self.summary_writer = summary_writer
        self.summary_writer.add_graph(self.sess.graph)

    def initialize_variable(self):
        self.sess.run(self.variables_initializer)

    def build_hidden_layer(self, input_layer, layer_num):
        h_size = 64
        with tf.variable_scope('hidden_layer'):
            hidden = input_layer
            for i in range(layer_num):
                hidden = tf.layers.dense(hidden, h_size, activation=tf.nn.leaky_relu, reuse=False,
                                        name="hidden_{}".format(i),
                                        kernel_initializer=c_layers.variance_scaling_initializer(
                                            1.0))
        return hidden

    def build_reward_network(self, param_placeholder, scope):
        with tf.variable_scope(scope):
            input_layer = param_placeholder
            hidden_layer = self.build_hidden_layer(input_layer, layer_num=3)

            with tf.variable_scope('output_layer'):
                    output_layer = tf.layers.dense(hidden_layer, units=1, activation=None)

            batchsize = tf.shape(param_placeholder)[0]
            segment_length = tf.shape(param_placeholder)[1]
            predict_reward = tf.reshape(output_layer, (batchsize, segment_length), name='predict_reward')
        return predict_reward

    def build_loss_func(self, left_predict_reward, right_predict_reward):
        with tf.variable_scope('loss_function'):
            left_segment_predict_reward = tf.reduce_sum(left_predict_reward, axis=1)
            right_segment_predict_reward = tf.reduce_sum(right_predict_reward, axis=1)
            reward_logits = tf.stack([left_segment_predict_reward, right_segment_predict_reward], axis=1)  # (batch_size, 2)

            if self.use_score:
                self.scores = tf.placeholder(dtype=tf.float32, shape=(None,2), name="comparison_scores")
                self.labels = tf.nn.softmax(self.scores, name='softmax_scores')
                data_loss = tf.losses.softmax_cross_entropy(logits=reward_logits, onehot_labels=self.labels)
            else:
                self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="comparison_labels")
                data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reward_logits, labels=self.labels)

            self.loss_op = tf.reduce_mean(data_loss)
            self.loss_summary = tf.summary.scalar('loss', self.loss_op)
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op)


    def build_model(self, scope):
        with tf.Graph().as_default() as graph:
            with tf.variable_scope(scope):
                with tf.variable_scope('predict_model'):
                    param_mean_shape = tuple([1, self.param_size])

                    batch_param_shape = (None, None) + self.param_shape

                    self.network_input_creator = NetworkInputCreator()

                    with tf.variable_scope('placeholder'):
                        self.left_param_placeholder = \
                            self.network_input_creator.create_vector_input(True, 'left_parameter', batch_param_shape, param_mean_shape)
                        self.right_param_placeholder = \
                            self.network_input_creator.create_vector_input(True, 'right_parameter', batch_param_shape, param_mean_shape)

                    with tf.variable_scope('reward_network'):
                        self.left_predict_reward = self.build_reward_network(self.left_param_placeholder, 'left')
                        right_predict_reward = self.build_reward_network(self.right_param_placeholder, 'right')

                    self.build_loss_func(self.left_predict_reward, right_predict_reward)

                    self.merged_summary = tf.summary.merge([self.loss_summary])
                    self.variables_initializer = tf.global_variables_initializer()

                self.sess = tf.Session(graph=graph)

    def increment_step(self):
        self.sess.run(self.network_input_creator.increment_step)

    def predict_reward(self, segment_list):
        """Predict the reward for each step in a given path"""
        predict_reward = self.sess.run(self.left_predict_reward, feed_dict={
            self.left_param_placeholder: np.asarray([segment['parameter'] for segment in segment_list]),
        })
        return predict_reward[0]

    def update_model(self,labeled_comparisons_batch):

        left_param = np.asarray([comp['left']['parameter'] for comp in labeled_comparisons_batch])
        right_param = np.asarray([comp['right']['parameter'] for comp in labeled_comparisons_batch])

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
