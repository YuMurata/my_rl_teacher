import tensorflow as tf
import tensorflow.contrib.layers as c_layers
import numpy as np
from collections import deque

from my_deep_learning.model.network_input import NetworkInputCreator
from mlagents.trainers.ppo.rl_teacher.nn import FullyConnectedMLP

import os

class PredictModel():
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, vec_obs_size:int, act_size:int,stack_num:int, scope:str, layer_num:int):
        # Build and initialize our predictor model

        self.act_size = act_size
        self.vec_obs_size = vec_obs_size
        self.layer_num = layer_num
        self.obs_shape = (self.vec_obs_size,)
        self.act_shape = (self.act_size,)
        
        self.graph = self.build_model()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(tf.global_variables_initializer())


        summary_dir = 'summary\\data_log'
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        summary_writer = tf.summary.FileWriter(summary_dir)
        summary_writer.add_graph(self.graph)

    @staticmethod
    def swish(input_activation):
        """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
        return tf.multiply(input_activation, tf.nn.sigmoid(input_activation))

    def build_input_layer(self, observation, action):
        with tf.variable_scope('input_layer'): 
            input_layer = tf.concat([observation, action], axis=2)
        return input_layer

    def build_hidden_layer(self, input_layer, layer_num):
        h_size = 64
        with tf.variable_scope('hidden_layer'):
            hidden = input_layer
            for i in range(layer_num):
                hidden = tf.layers.dense(hidden, h_size, activation=self.swish, reuse=False,
                                        name="hidden_{}".format(i),
                                        kernel_initializer=c_layers.variance_scaling_initializer(
                                            1.0))
        return hidden

    def build_reward_network(self, obs_placeholder, act_placeholder, scope):
        with tf.variable_scope(scope):
            input_layer = self.build_input_layer(obs_placeholder, act_placeholder)        
            hidden_layer = self.build_hidden_layer(input_layer, layer_num=3)
            
            with tf.variable_scope('output_layer'):
                    output_layer = tf.layers.dense(hidden_layer, units=1, activation=None)

            batchsize = tf.shape(obs_placeholder)[0]
            segment_length = tf.shape(obs_placeholder)[1]
            predict_reward = tf.reshape(output_layer, (batchsize, segment_length), name='predict_reward')
        return predict_reward



    def build_loss_func(self, left_predict_reward, right_predict_reward):
        with tf.variable_scope('loss_function'):
            left_segment_predict_reward = tf.reduce_sum(left_predict_reward, axis=1)
            right_segment_predict_reward = tf.reduce_sum(right_predict_reward, axis=1)
            reward_logits = tf.stack([left_segment_predict_reward, right_segment_predict_reward], axis=1)  # (batch_size, 2)

            self.scores = tf.placeholder(dtype=tf.float32, shape=(None,2), name="comparison_scores")
            data_loss = tf.losses.softmax_cross_entropy(logits=reward_logits, onehot_labels=self.scores)
            self.loss_op = tf.reduce_mean(data_loss)

            self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op)

    def build_model(self):
        graph = tf.Graph()
        with graph.as_default():
            obs_mean_shape = tuple([1, self.vec_obs_size])
            act_mean_shape = tuple([1, self.act_size])

            batch_obs_shape = (None, None) + self.obs_shape
            batch_act_shape = (None, None) + self.act_shape

            with tf.variable_scope('input_creator'):
                self.left_obs_input_creator = NetworkInputCreator(True, 'left_observation', batch_obs_shape, obs_mean_shape)
                self.right_obs_input_creator = NetworkInputCreator(True, 'right_observation', batch_obs_shape, obs_mean_shape)
                self.left_act_input_creator = NetworkInputCreator(True, 'left_action', batch_act_shape, act_mean_shape)
                self.right_act_input_creator = NetworkInputCreator(True, 'right_action', batch_act_shape, act_mean_shape)

            with tf.variable_scope('placeholder'):
                self.left_obs_placeholder = self.left_obs_input_creator.vector_in
                self.right_obs_placeholder = self.right_obs_input_creator.vector_in
                self.left_act_placeholder = self.left_act_input_creator.vector_in
                self.right_act_placeholder = self.right_act_input_creator.vector_in

            with tf.variable_scope('reward_network'):
                self.left_predict_reward = self.build_reward_network(self.left_obs_placeholder, self.left_act_placeholder, 'left')
                right_predict_reward = self.build_reward_network(self.right_obs_placeholder, self.right_act_placeholder, 'right')
            
            self.build_loss_func(self.left_predict_reward, right_predict_reward)
                    
        return graph

    def increment_step(self):
        self.sess.run(self.left_obs_input_creator.increment_step)
        self.sess.run(self.right_obs_input_creator.increment_step)
        self.sess.run(self.left_act_input_creator.increment_step)
        self.sess.run(self.right_act_input_creator.increment_step)

    def predict_reward(self, segment_list):
        """Predict the reward for each step in a given path"""
        
        predict_reward = self.sess.run(self.left_predict_reward, feed_dict={
            self.left_obs_placeholder: np.asarray([segment['observation'] for segment in segment_list]),
            self.left_act_placeholder: np.asarray([segment['action'] for segment in segment_list]),
        })
        return predict_reward[0]

    def update_model(self,labeled_comparisons_batch):
    
        left_obs = np.asarray([comp['left']['observation'] for comp in labeled_comparisons_batch])
        left_acts = np.asarray([comp['left']['action'] for comp in labeled_comparisons_batch])
        right_obs = np.asarray([comp['right']['observation'] for comp in labeled_comparisons_batch])
        right_acts = np.asarray([comp['right']['action'] for comp in labeled_comparisons_batch])
        scores = np.asarray([comp['score'] for comp in labeled_comparisons_batch])

        _, loss = self.sess.run([self.train_op, self.loss_op], feed_dict={
            self.left_obs_placeholder: left_obs,
            self.left_act_placeholder: left_acts,
            self.right_obs_placeholder: right_obs,
            self.right_act_placeholder: right_acts,
            self.scores: scores
        })

        return loss
