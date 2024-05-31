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

# Modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size, action_dim, state_dim):
        self.buffer = collections.deque(maxlen=max_size)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.DQN_action = []

    def append(self, exp):
        self.buffer.append(exp)
        # if len(self.buffer) < 20 or len(self.buffer) % 10 == 0:
        #     self.countRes(alls)

    def append_DQN_action(self, act):
        self.DQN_action.append(act)

    def pop(self):
        return self.buffer.popleft()

    # 从经验池中选取N条经验出来
    def DQN_sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        cnt = 0
        for experience in mini_batch:
            s, a, r, s_p, done = experience
            DQN_a = self.DQN_action[cnt]
            cnt += 1
            obs_batch.append(s)
            action_batch.append(DQN_a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
               np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'), \
               np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
            # np.array(action_batch).reshape((128, 4))

        # print('reward_batch', reward_batch)
        # print('action_batch', action_batch)
        # print('batch_size = ', batch_size)
        # print('action_dim = ', action_dim)
        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32').reshape(batch_size, self.action_dim), \
            np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

    def getStates(self):
        obs_all = []
        for experience in self.buffer:
            s, a, r, s_p, done = experience
            obs_all.append(s)
        # 获得state矩阵：len 行， state_dim 列
        # print('rpm state_dim = ', self.state_dim)
        obs_all = np.array(obs_all).astype('float32')
        # print('obs_all = ', obs_all)
        return obs_all

    def getActions(self):
        act_all = []
        for experience in self.buffer:
            s, a, r, s_p, done = experience
            act_all.append(a)
        # 获得state矩阵：len 行， state_dim 列
        act_all = np.array(act_all).astype('float32')
        return act_all

    def countRes(self, alls):
        # 分别计算alls矩阵的mean和std
        mean = np.mean(alls, axis=0)
        std = np.std(alls, axis=0)
        # print('mean:', mean)
        # print('std:', std)

        # avoid std[i] == 0
        for i in range(len(std)):
            # self.mean[i] = np.mean(obs_all[:, i])
            # self.std[i] = np.std(obs_all[:, i])
            if std[i] == 0:
                std[i] = 1

        return mean, std