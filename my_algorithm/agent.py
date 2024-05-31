# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
import os
import torch
from torch.autograd import Variable

import numpy as np
import parl
from parl import layers
from paddle import fluid


class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # print("self.obs_dim:", self.obs_dim)
        # print("self.act_dim:", self.act_dim)
        super(Agent, self).__init__(algorithm)

        # 注意：最开始先同步self.model和self.target_model的参数.
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()
        # self.getq_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

        # with fluid.program_guard(self.pred_program):
        #     obs = layers.data(
        #         name='obs', shape=[self.obs_dim], dtype='float32')
        #     act = layers.data(
        #         name='act', shape=[self.act_dim], dtype='float32')
        #     print('obs:', obs)
        #     print('obs:', act)
        #
        #     self.q = self.alg.get_q(obs, act)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        # print('===obs===:', obs)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        # print(act)
        act = np.squeeze(act)
        return act

    # def predict_v_act(self, obs):
    #     obs = np.expand_dims(obs, axis=0)
    #     act = self.fluid_executor.run(
    #         self.pred_program, feed={'obs': obs},
    #         fetch_list=[self.pred_act], return_numpy = False)[0]
    #     #print(act)
    #     #act = np.squeeze(act)
    #     return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        # print("self.critic_cost_shape:",self.critic_cost)
        critic_cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost

    def normalizer(self, obs, rpm):
        # TODO：状态归一化--根据经验池中的均值和梯度实现当前状态的归一化，此步骤在状态输入神经网络时进行
        if len(rpm) <= 1:
            return np.ones(rpm.state_dim)

        mean, std = rpm.countRes(rpm.getStates())
        obs = np.array(obs)
        # print("mean:", mean)
        # print("std:", std)
        # print("obs:", obs)
        return (obs - mean) / std

    def normalizerBatch(self, batch_obs, rpm):
        # TODO：状态归一化--根据经验池中的均值和梯度实现当前状态的归一化，此步骤在状态输入神经网络时进行
        mean, std = rpm.countRes(rpm.getStates())
        res_obs = []
        for obs in batch_obs:
            obs = np.array(obs)
            obs = (obs - mean) / std
            res_obs.append(obs)
        # print("mean:", mean)
        # print("std:", std)
        # print("obs:", obs)
        return np.array(res_obs)

    # def get_q(self, obs, act):
    #     obs = np.expand_dims(obs, axis=0)
    #     q = self.fluid_executor.run(
    #         self.pred_program, feed={'obs': obs, 'act': act},
    #         fetch_list=[self.q])[0]
    #     q = np.squeeze(q)
    #     return q

class TestAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(TestAgent, self).__init__(algorithm)

        mean = np.zeros(obs_dim)
        var = np.zeros(obs_dim)
        # self.normalizer = Normalizer(mean, var)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)
        self.learn_it = 0
        self.policy_freq = self.alg.policy_freq

    def build_program(self):
        self.pred_program = fluid.Program()
        self.actor_learn_program = fluid.Program()
        self.critic_learn_program = fluid.Program()
        self.cal_td_err_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.actor_learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.actor_cost = self.alg.actor_learn(obs)

        with fluid.program_guard(self.critic_learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.critic_cost = self.alg.critic_learn(obs, act, reward,
                                                     next_obs, terminal)
            
        with fluid.program_guard(self.cal_td_err_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.td_error = self.alg.cal_td_error(obs, act, reward,
                                                     next_obs, terminal)
        

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        # print('===obs===:', obs)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        #print(act)
        act = np.squeeze(act)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        self.learn_it += 1
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(
            self.critic_learn_program,
            feed=feed,
            fetch_list=[self.critic_cost])[0]
        # print('critic loss = ', critic_cost[0])

        actor_cost = None
        if self.learn_it % self.policy_freq == 0:
            actor_cost = self.fluid_executor.run(
                self.actor_learn_program,
                feed={'obs': obs},
                fetch_list=[self.actor_cost])[0]
            self.alg.sync_target()
        return actor_cost, critic_cost
    
    def cal_td_error(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        td_error = self.fluid_executor.run(
        self.cal_td_err_program,
        feed=feed,
        fetch_list=[self.td_error])[0]
        print("******************************************")
        print("**                                      **")
        print("**   agent td error = ", td_error)
        print("**                                      **")
        print("******************************************")

        return td_error

    def save(self, save_path, mode, program=None):
        """Save parameters.
        Args:
            save_path(str): where to save the parameters.
            program(fluid.Program): program that describes the neural network structure. If None, will use self.learn_program.
        Raises:
            ValueError: if program is None and self.learn_program does not exist.
        Example:
        .. code-block:: python
            agent = AtariAgent()
            agent.save('./model.ckpt')
        """
        if mode == "train_actor":
            save_program = self.actor_learn_program
        elif mode == "train_critic":
            save_program = self.critic_learn_program
        elif mode == "predict":
            save_program = self.pred_program
        else:
            save_program = self.pred_program
        if program is None:
            program = save_program
        dirname = os.sep.join(save_path.split(os.sep)[:-1])
        filename = save_path.split(os.sep)[-1]
        fluid.io.save_params(
            executor=self.fluid_executor,
            dirname=dirname,
            main_program=program,
            filename=filename)

    def restore(self, save_path, mode, program=None):
        """Restore previously saved parameters.
        This method requires a program that describes the network structure.
        The save_path argument is typically a value previously passed to ``save_params()``.
        Args:
            save_path(str): path where parameters were previously saved.
            program(fluid.Program): program that describes the neural network structure. If None, will use self.learn_program.
        Raises:
            ValueError: if program is None and self.learn_program does not exist.
        Example:
        .. code-block:: python
            agent = AtariAgent()
            agent.save('./model.ckpt')
            agent.restore('./model.ckpt')
        """
        if mode == "train_actor":
            save_program = self.actor_learn_program
        elif mode == "train_critic":
            save_program = self.critic_learn_program
        elif mode == "predict":
            save_program = self.pred_program
        else:
            save_program = self.pred_program
        if program is None:
            program = save_program
        if type(program) is fluid.compiler.CompiledProgram:
            program = program._init_program
        dirname = os.sep.join(save_path.split(os.sep)[:-1])
        filename = save_path.split(os.sep)[-1]
        fluid.io.load_params(
            executor=self.fluid_executor,
            dirname=dirname,
            main_program=program,
            filename=filename)

    # TODO：normalizer到底应该放在哪？？
    def normalizer(self, obs, rpm):
        # TODO：状态归一化--根据经验池中的均值和梯度实现当前状态的归一化，此步骤在状态输入神经网络时进行
        if len(rpm) <= 1:
            return np.ones(rpm.state_dim)

        mean, std = rpm.countRes(rpm.getStates())
        obs = np.array(obs)
        # print("mean:", mean)
        # print("std:", std)
        # print("obs:", obs)
        return (obs - mean) / std

    def normalizerBatch(self, batch_obs, rpm):
        # TODO：状态归一化--根据经验池中的均值和梯度实现当前状态的归一化，此步骤在状态输入神经网络时进行
        mean, std = rpm.countRes(rpm.getStates())
        res_obs = []
        for obs in batch_obs:
            obs = np.array(obs)
            obs = (obs - mean) / std
            res_obs.append(obs)
        # print("mean:", mean)
        # print("std:", std)
        # print("obs:", obs)
        return np.array(res_obs)







class SACAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(SACAgent, self).__init__(algorithm)

        mean = np.zeros(obs_dim)
        var = np.zeros(obs_dim)
        # self.normalizer = Normalizer(mean, var)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)
        self.learn_it = 0
        # self.policy_freq = self.alg.policy_freq

    def build_program(self):
        self.pred_program = fluid.Program()
        # self.actor_learn_program = fluid.Program()
        # self.critic_learn_program = fluid.Program()
        self.cal_td_err_program = fluid.Program()
        self.learn_program = fluid.Program()
        self.sample_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)
            
        with fluid.program_guard(self.sample_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.sample_act, _ = self.alg.sample(obs)

        # with fluid.program_guard(self.actor_learn_program):
        #     obs = layers.data(
        #         name='obs', shape=[self.obs_dim], dtype='float32')
        #     self.actor_cost = self.alg.actor_learn(obs)

        # with fluid.program_guard(self.critic_learn_program):
        #     obs = layers.data(
        #         name='obs', shape=[self.obs_dim], dtype='float32')
        #     act = layers.data(
        #         name='act', shape=[self.act_dim], dtype='float32')
        #     reward = layers.data(name='reward', shape=[], dtype='float32')
        #     next_obs = layers.data(
        #         name='next_obs', shape=[self.obs_dim], dtype='float32')
        #     terminal = layers.data(name='terminal', shape=[], dtype='bool')
        #     self.critic_cost = self.alg.critic_learn(obs, act, reward,
        #                                              next_obs, terminal)
            
        with fluid.program_guard(self.cal_td_err_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.td_error = self.alg.cal_td_error(obs, act, reward,
                                                     next_obs, terminal)
        
        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.critic_cost, self.actor_cost = self.alg.learn(obs, act, reward,
                                                     next_obs, terminal)
        
        
        

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        # print('===obs===:', obs)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        #print(act)
        act = np.squeeze(act)
        return act


    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)
        # print('===obs===:', obs)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        #print(act)
        act = np.squeeze(act)
        return act


    def learn(self, obs, act, reward, next_obs, terminal):
        # self.learn_it += 1
        # feed = {
        #     'obs': obs,
        #     'act': act,
        #     'reward': reward,
        #     'next_obs': next_obs,
        #     'terminal': terminal
        # }
        # critic_cost = self.fluid_executor.run(
        #     self.critic_learn_program,
        #     feed=feed,
        #     fetch_list=[self.critic_cost])[0]
        # # print('critic loss = ', critic_cost[0])

        # actor_cost = None
        # if self.learn_it % self.policy_freq == 0:
        #     actor_cost = self.fluid_executor.run(
        #         self.actor_learn_program,
        #         feed={'obs': obs},
        #         fetch_list=[self.actor_cost])[0]
        #     self.alg.sync_target()
        # return actor_cost, critic_cost
        
        
        self.learn_it += 1
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        
        [critic_cost, actor_cost] = self.fluid_executor.run(
            self.learn_program,
            feed=feed,
            fetch_list=[self.critic_cost, self.actor_cost])
        # print('critic loss = ', critic_cost[0])
        
        self.alg.sync_target()
        return critic_cost[0], actor_cost[0]
        
        
        
        
        
    
    def cal_td_error(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        td_error = self.fluid_executor.run(
        self.cal_td_err_program,
        feed=feed,
        fetch_list=[self.td_error])[0]
        print("******************************************")
        print("**                                      **")
        print("**   agent td error = ", td_error)
        print("**                                      **")
        print("******************************************")

        return td_error

    def save(self, save_path, mode, program=None):
        """Save parameters.
        Args:
            save_path(str): where to save the parameters.
            program(fluid.Program): program that describes the neural network structure. If None, will use self.learn_program.
        Raises:
            ValueError: if program is None and self.learn_program does not exist.
        Example:
        .. code-block:: python
            agent = AtariAgent()
            agent.save('./model.ckpt')
        """
        # if mode == "train_actor":
        #     save_program = self.actor_learn_program
        # elif mode == "train_critic":
        #     save_program = self.critic_learn_program
        if mode == "predict":
            save_program = self.pred_program
        elif mode == "learn":
            save_program = self.learn_program
        else:
            save_program = self.pred_program
        if program is None:
            program = save_program
        dirname = os.sep.join(save_path.split(os.sep)[:-1])
        filename = save_path.split(os.sep)[-1]
        fluid.io.save_params(
            executor=self.fluid_executor,
            dirname=dirname,
            main_program=program,
            filename=filename)

    def restore(self, save_path, mode, program=None):
        """Restore previously saved parameters.
        This method requires a program that describes the network structure.
        The save_path argument is typically a value previously passed to ``save_params()``.
        Args:
            save_path(str): path where parameters were previously saved.
            program(fluid.Program): program that describes the neural network structure. If None, will use self.learn_program.
        Raises:
            ValueError: if program is None and self.learn_program does not exist.
        Example:
        .. code-block:: python
            agent = AtariAgent()
            agent.save('./model.ckpt')
            agent.restore('./model.ckpt')
        """
        # if mode == "train_actor":
        #     save_program = self.actor_learn_program
        # elif mode == "train_critic":
            # save_program = self.critic_learn_program
        if mode == "predict":
            save_program = self.pred_program
        elif mode == "learn":
            save_program = self.learn_program
        else:
            save_program = self.pred_program
        if program is None:
            program = save_program
        if type(program) is fluid.compiler.CompiledProgram:
            program = program._init_program
        dirname = os.sep.join(save_path.split(os.sep)[:-1])
        filename = save_path.split(os.sep)[-1]
        fluid.io.load_params(
            executor=self.fluid_executor,
            dirname=dirname,
            main_program=program,
            filename=filename)

    # TODO：normalizer到底应该放在哪？？
    def normalizer(self, obs, rpm):
        # TODO：状态归一化--根据经验池中的均值和梯度实现当前状态的归一化，此步骤在状态输入神经网络时进行
        if len(rpm) <= 1:
            return np.ones(rpm.state_dim)

        mean, std = rpm.countRes(rpm.getStates())
        obs = np.array(obs)
        # print("mean:", mean)
        # print("std:", std)
        # print("obs:", obs)
        return (obs - mean) / std

    def normalizerBatch(self, batch_obs, rpm):
        # TODO：状态归一化--根据经验池中的均值和梯度实现当前状态的归一化，此步骤在状态输入神经网络时进行
        mean, std = rpm.countRes(rpm.getStates())
        res_obs = []
        for obs in batch_obs:
            obs = np.array(obs)
            obs = (obs - mean) / std
            res_obs.append(obs)
        # print("mean:", mean)
        # print("std:", std)
        # print("obs:", obs)
        return np.array(res_obs)
