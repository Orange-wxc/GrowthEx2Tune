#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import parl
from parl import layers
from copy import deepcopy
from paddle import fluid
import paddle

import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        """  DDPG algorithm

        Args:
            model (parl.Model): actor and critic 的前向网络.
                                model 必须实现 get_actor_params() 方法.
            gamma (float): reward的衰减因子.
            tau (float): self.target_model 跟 self.model 同步参数 的 软更新参数
            actor_lr (float): actor 的学习率
            critic_lr (float): critic 的学习率
        """
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(model)

    def predict(self, obs):
        """ 使用 self.model 的 actor model 来预测动作
        """
        action = self.model.policy(obs)
        # if obs.shape[0]!=-1:
        #     obs.reshape(1, -1)
        q = self.model.value(obs, action)
        # paddle.fluid.layers.Print(obs, message='_actor_learn obs')
        # paddle.fluid.layers.Print(action, message='_actor_learn action')
        # paddle.fluid.layers.Print(q, message='q value')

        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 用DDPG算法更新 actor 和 critic
        """
        actor_cost = self._actor_learn(obs)
        critic_cost = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        return actor_cost, critic_cost


    # def get_q(self, obs, act):
    #     return self.model.value(obs, act)

    # Actor网络(策略网络)更新
    def _actor_learn(self, obs):
        action = self.model.policy(obs)
        Q = self.model.value(obs, action)
        #paddle.fluid.layers.Print(concat, message='obs concat act')
        # paddle.fluid.layers.Print(obs, message='_actor_learn obs')
        # paddle.fluid.layers.Print(action, message='_actor_learn action')
        # paddle.fluid.layers.Print(Q, message='Q value')
        # print('obs: ', obs)
        # print('action: ', action)
        # print('Q: ', np.array(Q))
        # 计算 loss = -Q
        # 目标是最大化Q，但是优化器是最小化loss，所以取-Q，最大化Q
        # 用梯度下降来完成梯度上升
        cost = layers.reduce_mean(-1.0 * Q)
        optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
        # 在这里只更新actor网络的参数，不期望改变critic网络的参数
        # 所以给出parameter_list，使网络更新时自动识别actor的网络参数并进行更新
        optimizer.minimize(cost, parameter_list=self.model.get_actor_params())
        return cost

    # Critic网络(Q网络)更新
    # 输入为ReplayMemory中一个batch的数据
    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        next_action = self.target_model.policy(next_obs)
        next_Q = self.target_model.value(next_obs, next_action)
        #paddle.fluid.layers.Print(obs, message='_critic_learn obs')
        #paddle.fluid.layers.Print(next_action, message='_critic_learn next_action')
        #paddle.fluid.layers.Print(action,  message='_actor_learn action')
        terminal = layers.cast(terminal, dtype='float32')
        # 在这里用terminal实现if else的逻辑
        # if 终止状态 则 Q = reward
        # else 非终止状态 则 Q = reward + gamma * next_Q
        target_Q = reward + (1.0 - terminal) * self.gamma * next_Q
        #print('target_q: ', target_Q)
        #paddle.fluid.layers.Print(target_Q, message='_critic_learn target_Q')
        # 阻止修改target_Q网络
        # target_Q网络定时复制，所以在执行learn函数时不需要更新
        target_Q.stop_gradient = True

        Q = self.model.value(obs, action)
        # print('obs : ', obs, 'action : ', action, 'Q : ', Q)
        #paddle.fluid.layers.Print(Q, message='_critic_learn Q')

        # 计算损失函数
        cost = layers.square_error_cost(Q, target_Q)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(self.critic_lr)
        # 优化损失函数，使Q接近target_Q
        optimizer.minimize(cost)
        return cost

    # Target network参数软更新
    # 设置 tau，来控制更新的幅度
    def sync_target(self, decay=None, share_vars_parallel_executor=None):
        """ self.target_model从self.model复制参数过来，若decay不为None,则是软更新
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(
            self.target_model,
            decay=decay,
            share_vars_parallel_executor=share_vars_parallel_executor)


class TD3(parl.Algorithm):
    def __init__(
            self,
            model,
            max_action,
            gamma=None,
            tau=None,
            actor_lr=None,
            critic_lr=None,
            policy_noise=0.2,  # Noise added to target policy during critic update
            noise_clip=0.5,  # Range to clip target policy noise
            policy_freq=2):  # Frequency of delayed policy updates
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.model = model.to(device)
        self.target_model = deepcopy(model).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.get_critic_params(), lr=critic_lr)

        self.total_it = 0

    def predict(self, obs):
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        self.total_it += 1
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)

            next_action = (self.target_model.policy(next_obs) + noise).clamp(
                -self.max_action, self.max_action)

            target_Q1, target_Q2 = self.target_model.value(
                next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - terminal) * self.gamma * target_Q

        current_Q1, current_Q2 = self.model.value(obs, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:

            actor_loss = -self.model.Q1(obs, self.model.policy(obs)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.sync_target()

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        for param, target_param in zip(self.model.parameters(),
                                       self.target_model.parameters()):
            target_param.data.copy_((1 - decay) * param.data +
                                    decay * target_param.data)
