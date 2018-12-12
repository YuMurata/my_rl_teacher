import tensorflow as tf
import numpy as np
import random

from mlagents.envs.brain import BrainInfo, BrainParameters

from mlagents.trainers.ppo.rl_teacher.predict_model import PredictModel
from mlagents.trainers.network_input import NetworkInputCreator
from mlagents.trainers.ppo.rl_teacher.comparison_collectors import HumanComparisonCollector

from mlagents.trainers.ppo.rl_teacher.nn import FullyConnectedMLP
from keras import backend as K

class ComparisonRewardPredictor():
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self,sess:tf.Session, brain_parameter:BrainParameters, comparison_collector:HumanComparisonCollector, scope:str, layer_num:int):
        self.comparison_collector = comparison_collector
        self._elapsed_predictor_training_iters = 0
        self.require_comparisons_length=100

        # Build and initialize our predictor model
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = sess

        self.act_size = brain_parameter.vector_action_space_size[0]
        self.vec_obs_size = brain_parameter.vector_observation_space_size * \
                            brain_parameter.num_stacked_vector_observations

        self.layer_num=layer_num
        self.predict_model = PredictModel(sess,self.vec_obs_size,self.act_size,self.comparison_collector.stack_num,scope,layer_num)

    def predict_reward(self, new_info:BrainInfo):
        """Predict the reward for each step in a given path"""

        obs_list,act_list = [],[]
        agent_num = len(new_info.agents)
        base_shape = (1,agent_num)

        for i in range(agent_num):
            observation= np.array(new_info.vector_observations[i])
            action= np.array(new_info.previous_vector_actions[i])
            
            obs_list.append(observation)
            act_list.append(action)

        segment = {'observation':obs_list,'action':act_list}
        return self.predict_model.predict_reward([segment])
        
    def train_predictor(self):
        minibatch_size = min(64, self.comparison_collector.stacked_comparisons_length)
        labeled_comparisons = np.random.choice(self.comparison_collector.comparisons, minibatch_size, False)
        
        loss = self.predict_model.update_model(labeled_comparisons)        
        return loss

    @property
    def is_ready_train(self):
        return self.comparison_collector.stacked_comparisons_length >= self.require_comparisons_length
