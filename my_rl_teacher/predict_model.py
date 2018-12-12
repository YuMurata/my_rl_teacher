import tensorflow as tf
import tensorflow.contrib.layers as c_layers
import numpy as np
from collections import deque

from my_deep_learning.model.network_input import NetworkInputCreator

import os

class PredictModel():
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, vec_obs_size:int, act_size:int,stack_num:int, scope:str, layer_num:int, use_score:bool, summary_dir='summary/predict_model'):
        # Build and initialize our predictor model

        self.act_size = act_size
        self.vec_obs_size = vec_obs_size
        self.layer_num = layer_num
        self.obs_shape = (self.vec_obs_size,)
        self.act_shape = (self.act_size,)
        self.use_score = use_score
        self.build_model()
        self.initialize_variable()

        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

    def initialize_variable(self):
        self.sess.run(self.variables_initializer)

    def build_input_layer(self, observation, action):
        with tf.variable_scope('input_layer'): 
            input_layer = tf.concat([observation, action], axis=2)
        return input_layer

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
            

    def build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            obs_mean_shape = tuple([1, self.vec_obs_size])
            act_mean_shape = tuple([1, self.act_size])

            batch_obs_shape = (None, None) + self.obs_shape
            batch_act_shape = (None, None) + self.act_shape

            self.network_input_creator = NetworkInputCreator()

            with tf.variable_scope('placeholder'):
                self.left_obs_placeholder = self.network_input_creator.create_vector_input(True, 'left_observation', batch_obs_shape, obs_mean_shape)
                self.right_obs_placeholder = self.network_input_creator.create_vector_input(True, 'right_observation', batch_obs_shape, obs_mean_shape)
                self.left_act_placeholder = self.network_input_creator.create_vector_input(True, 'left_action', batch_act_shape, act_mean_shape)
                self.right_act_placeholder = self.network_input_creator.create_vector_input(True, 'right_action', batch_act_shape, act_mean_shape)

            with tf.variable_scope('reward_network'):
                self.left_predict_reward = self.build_reward_network(self.left_obs_placeholder, self.left_act_placeholder, 'left')
                right_predict_reward = self.build_reward_network(self.right_obs_placeholder, self.right_act_placeholder, 'right')
            
            self.build_loss_func(self.left_predict_reward, right_predict_reward)
            
            self.merged_summary = tf.summary.merge([self.loss_summary])

            self.sess = tf.Session(graph=self.graph)
            self.variables_initializer = tf.global_variables_initializer()

    def increment_step(self):
        self.sess.run(self.network_input_creator.increment_step)

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

        if self.use_score:
            scores = np.asarray([comp['score'] for comp in labeled_comparisons_batch])
            feed_dict={
                 self.left_obs_placeholder: left_obs,
                self.left_act_placeholder: left_acts,
                self.right_obs_placeholder: right_obs,
                self.right_act_placeholder: right_acts,
                self.scores: scores}
            
        else:
            labels = np.asarray([comp['label'] for comp in labeled_comparisons_batch])
            feed_dict={
                 self.left_obs_placeholder: left_obs,
                self.left_act_placeholder: left_acts,
                self.right_obs_placeholder: right_obs,
                self.right_act_placeholder: right_acts,
                self.labels: labels}

        _, loss, summary = self.sess.run([self.train_op, self.loss_op, self.loss_summary], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, self.sess.run(self.network_input_creator.global_step))
        self.increment_step()
        return loss
