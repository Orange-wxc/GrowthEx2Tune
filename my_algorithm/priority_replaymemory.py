# -*- coding: utf-8 -*-
"""

Prioritized Replay Memory
"""
import random
import pickle
import numpy as np


class SumTree(object):
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.num_entries = 0

    def _propagate(self, idx, change):
        parent = int((idx - 1) / 2)
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        print('tree add idx:', idx)

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.num_entries < self.capacity:
            self.num_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return [idx, self.tree[idx], self.data[data_idx]]
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


class PrioritizedReplayMemory(object):
    def __init__(self, max_size, action_dim, state_dim):
        self.buffer = collections.deque(maxlen=max_size)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.DQN_action = []
        self.tree = SumTree(max_size)
        self.capacity = max_size
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        self.buffer.append(sample)
        # (s, a, r, s, t)
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def __len__(self):
        return self.tree.num_entries

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            print('get idx:',idx)
            s, a, r, s_p, done = data
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)


        return idxs, np.array(obs_batch).astype('float32'), \
               np.array(action_batch).astype('float32').reshape(n, self.action_dim), \
               np.array(reward_batch).astype('float32'), \
               np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump({"tree": self.tree}, f)
        f.close()

    def load_memory(self, path):
        with open(path, 'rb') as f:
            _memory = pickle.load(f)
        self.tree = _memory['tree']



    def pop(self):
        return self.buffer.popleft()

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
