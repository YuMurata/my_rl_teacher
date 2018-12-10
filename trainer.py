from logging import getLogger
import os
from collections import deque
from time import time, sleep

import numpy as np
import tensorflow as tf

from mlagents.envs.brain import AllBrainInfo, BrainInfo, BrainParameters
from mlagents.envs.environment import UnityEnvironment

from mlagents.trainers.buffer import Buffer
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.trainer import UnityTrainerException, Trainer

from mlagents.trainers.ppo.rl_teacher.comparison_collectors import HumanComparisonCollector
from mlagents.trainers.ppo.rl_teacher.predictor import ComparisonRewardPredictor

class PredictTrainer(PPOTrainer):
    """The PPOTrainer is an implementation of the PPO algorithm."""

    def __init__(self, sess:tf.Session, brain_parameter:BrainParameters, reward_buff_cap:int, 
        trainer_parameters:dict, training:bool, seed:float, run_id:str, env:UnityEnvironment):
        """
        Responsible for collecting experiences and training PPO model.
        :param sess: Tensorflow session.
        :param env: The UnityEnvironment.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        """

        self.predict_way = trainer_parameters['predict_way']

        self.predict_cumulative_rewards={}

        self.comparison_collector = HumanComparisonCollector()        
        self.predictor = ComparisonRewardPredictor(sess, brain_parameter, self.comparison_collector, trainer_parameters['graph_scope'],layer_num = 1)

        self.left_agent_info = env.left_agent_info
        self.right_agent_info = env.right_agent_info
        self.label = env.label
        self.logger = getLogger('rl_teacher.trainer')
        self.pretrain_param = {
            'labels_length':trainer_parameters['pretrain_labels_length'],
            'pretrain_iters':trainer_parameters['pretrain_iters'],
            'is_verbose':True,
            'summary_freq':1000,
            'require_loss':1e-1
        }

        self.steps_per_predictor_training=trainer_parameters['steps_per_predictor_training']
        self.verbose_predict=True
        self.verbose_label = False
        self.verbose_loss = True

        super(PredictTrainer, self).__init__(sess, brain_parameter,reward_buff_cap, trainer_parameters, training, seed, run_id)
        self.stats['predict_cumulative_rewards']=[]
        self.stats['predict_loss']=[]

    def __str__(self):
        return '''Hyperparameters for the Predict Trainer of brain {0}: \n{1}'''.format(
            self.brain_name, '\n'.join(['\t{0}:\t{1}'.format(x, self.trainer_parameters[x]) for x in self.param_keys]))

    def update_predictor(self):
        loss=self.predictor.train_predictor()
        self.stats['predict_loss'].append(loss)

        if self.verbose_loss:
            self.logger.info('loss: {0}'.format(loss))

    def increment_step_and_update_last_reward(self):
        self.predictor.predict_model.increment_step()
        super(PredictTrainer, self).increment_step_and_update_last_reward()

    def add_experiences(self, curr_all_info: AllBrainInfo, next_all_info: AllBrainInfo, take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param curr_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param next_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param take_action_outputs: The outputs of the take action method.
        """

        next_info = next_all_info[self.brain_name]
        predict_reward=self.predictor.predict_reward(next_info)
        if self.verbose_predict:
            self.logger.info('predict: {0}'.format(predict_reward))

        if self.predict_way != 'synth':
            next_info.rewards=predict_reward

        for agent_id in next_info.agents:
            stored_info = self.training_buffer[agent_id].last_brain_info
            if stored_info is not None:
                idx = stored_info.agents.index(agent_id)
                next_idx = next_info.agents.index(agent_id)
                if not stored_info.local_done[idx]:
                    for agent_id in next_info.agents:
                        if agent_id not in self.predict_cumulative_rewards:
                            self.predict_cumulative_rewards[agent_id] = 0
                        self.predict_cumulative_rewards[agent_id] += next_info.rewards[next_idx]

        super(PredictTrainer,self).add_experiences(curr_all_info,next_all_info,take_action_outputs)

    def process_experiences(self, current_info: AllBrainInfo, new_info: AllBrainInfo):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Dictionary of all current brains and corresponding BrainInfo.
        :param new_info: Dictionary of all next brains and corresponding BrainInfo.
        """

        info = new_info[self.brain_name]
        for l in range(len(info.agents)):
            agent_actions = self.training_buffer[info.agents[l]]['actions']
            if ((info.local_done[l] or len(agent_actions) > self.trainer_parameters['time_horizon'])
                and len(agent_actions) > 0):
                agent_id = info.agents[l]
                if info.local_done[l]:
                    # self.stats['predict_cumulative_rewards'].append(
                    #     self.predict_cumulative_rewards.get(agent_id, 0))
                    self.predict_cumulative_rewards[agent_id]=0

        super(PredictTrainer,self).process_experiences(current_info,new_info)

    @property
    def is_collect_pretrain_labels(self):
        return self.comparison_collector.stacked_comparisons_length >= self.pretrain_param['labels_length']

    @property
    def is_predictor_training_steps(self):
        return self.get_step % self.steps_per_predictor_training == 0

    @property
    def is_ready_update_predictor(self):
        return self.is_predictor_training_steps and self.predictor.is_ready_train

    def add_comparison(self):
        left_info=self.left_agent_info[self.brain_name]
        right_info=self.right_agent_info[self.brain_name]

        self.comparison_collector.stack_segment(left_info,right_info)

        if self.comparison_collector.is_ready_add_segment_pair:
            label=int(self.label[self.brain_name])
            if self.predict_way == 'synth':
                left_rewards=self.comparison_collector.left_segment_dict["reward"]
                right_rewards=self.comparison_collector.right_segment_dict['reward']

                left_has_more_rew = np.sum(left_rewards) > np.sum(right_rewards)
                label = 1 if left_has_more_rew else 2
                
                if np.random.choice([True,False]):
                    left_has_less_rew = not left_has_more_rew
                    label = -1 if left_has_less_rew else -2
            if self.predict_way == 'test':
                label = self.predict_test()
                epsilon = 0.9
                if np.random.choice([True,False],p=[epsilon,1-epsilon]):
                    label=0

            if label != 0:
                self.comparison_collector.add_segment_pair(label)

                if self.verbose_label:
                    if label > 0:
                        self.logger.info('better: '+('left' if label==1 else 'right'))
                    if label < 0:
                        self.logger.info('worse: '+('left' if label==1 else 'right'))

    def pretrain(self):
        loss_list=[]

        self.logger.info('predictor pretrain now ...')
        while len(loss_list) > 0 and \
            np.mean(loss_list) < self.pretrain_param['require_loss']:

            loss_list.append(self.predictor.train_predictor())
            train_count += 1
            if train_count%self.pretrain_param['summary_freq']==0:
                self.logger.info('''{0} predictor pretraining iters \
                         {1} loss ...'''\
                         .format(train_count,np.mean(loss_list)))

        self.stats['predict_loss'].append(np.mean(loss_list))
        if self.verbose_loss:
            self.logger.info('loss: {0}'.format(np.mean(loss_list)))

    def predict_test(self):
        left_obs=self.comparison_collector.left_segment_dict['observation']
        right_obs=self.comparison_collector.right_segment_dict['observation']

        left_acs=self.comparison_collector.left_segment_dict['action']
        right_acs=self.comparison_collector.right_segment_dict['action']

        left_rewards=self.comparison_collector.left_segment_dict['reward']
        right_rewards=self.comparison_collector.right_segment_dict['reward']

        left_has_more_rew = np.sum(left_rewards) > np.sum(right_rewards)

        left_clock_wise = \
        np.sum([a[-1] for a in left_obs]) > \
        np.sum([a[-1] for a in right_obs])

        left_no_speed = \
        np.sum(np.abs([a[-1] for a in left_obs])) < \
        np.sum(np.abs([a[-1] for a in right_obs]))

        left_no_torque = \
        np.sum(np.abs(left_acs)) < \
        np.sum(np.abs(right_acs))

        left_no_angle = \
        np.abs(left_obs[0][2]) < np.abs(right_obs[0][2])

        left_less_obs = left_obs[0][0] < right_obs[0][0]
        test_flag=left_less_obs

        label = 1 if test_flag else 2
        
        if np.random.choice([True,False]):
            inv_test_flag = not test_flag
            label = -1 if inv_test_flag else -2

        return label

    def random_action(self, curr_all_info:AllBrainInfo):
        curr_info = curr_all_info[self.brain_name]
        return np.random.rand(len(curr_info.agents), self.predictor.act_size)