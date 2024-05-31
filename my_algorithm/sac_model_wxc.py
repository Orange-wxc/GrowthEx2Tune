import parl
from parl import layers
import numpy as np
import paddle


LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0


class SACActorModel(parl.Model):
    # def __init__(self, act_dim, max_action):
    #     # hid_size = 64
    #     #
    #     # self.fc1 = layers.fc(size=hid_size, act='relu')
    #     # self.fc2 = layers.fc(size=act_dim, act='tanh')

    #     hid1_size = 128
    #     hid2_size = 128

    #     self.fc1 = layers.fc(size=hid1_size, act='relu')
    #     self.fc2 = layers.fc(size=hid2_size, act='relu')
    #     self.fc3 = layers.fc(size=act_dim, act='tanh')

    #     self.max_action = max_action

    # def policy(self, obs):
    #     # hid = self.fc1(obs)
    #     # means = self.fc2(hid)
    #     # # print('means')
    #     # # paddle.fluid.layers.Print(means)
    #     # return means
    #     hid1 = self.fc1(obs)
    #     hid2 = self.fc2(hid1)
    #     means = self.fc3(hid2)
    #     means = means * self.max_action
    #     return means
    
    
    def __init__(self, act_dim):
        hid1_size = 400
        hid2_size = 300

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.mean_linear = layers.fc(size=act_dim)
        self.log_std_linear = layers.fc(size=act_dim)
    #这里是一个网络,共享了hid2的参数,再用2个FC层分别输出means和log_std
    def policy(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        means = self.mean_linear(hid2)
        log_std = self.log_std_linear(hid2)
        log_std = layers.clip(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return means, log_std


class SACCriticModel(parl.Model):
    # def __init__(self):
    #     # hid_size = 100
    #     #
    #     # self.fc1 = layers.fc(size=hid_size, act='relu')
    #     # self.fc2 = layers.fc(size=1, act=None)
    #     hid1_size = 256
    #     hid2_size = 256

    #     self.fc1 = layers.fc(size=hid1_size, act='relu')
    #     self.fc2 = layers.fc(size=hid2_size, act='relu')
    #     self.fc3 = layers.fc(size=1, act=None)

    #     self.fc4 = layers.fc(size=hid1_size, act='relu')
    #     self.fc5 = layers.fc(size=hid2_size, act='relu')
    #     self.fc6 = layers.fc(size=1, act=None)

    # def value(self, obs, act):
    #     # 拼接obs和act

    #     # concat = layers.concat([obs, act], axis=1)
    #     # # print("concat_shape:",concat.shape)
    #     # hid = self.fc1(concat)
    #     # # print("hid_shape:",hid.shape)
    #     # Q = self.fc2(hid)
    #     # # print("1Q_shape", Q.shape)
    #     # #print("QqqqQ = ",np.array(Q[0]))
    #     #
    #     # Q = layers.squeeze(Q, axes=[1])
    #     # # print("2Q_shape",Q.shape)
    #     # #print("Q_squeeze = ", Q.)
    #     # # print('Q = ')
    #     # # paddle.fluid.layers.Print(concat, message='obs concat act')
    #     # # paddle.fluid.layers.Print(Q, message='Q value')
    #     #
    #     # return Q
    #     hid1 = self.fc1(obs)
    #     concat1 = layers.concat([hid1, act], axis=1)
    #     Q1 = self.fc2(concat1)
    #     Q1 = self.fc3(Q1)
    #     Q1 = layers.squeeze(Q1, axes=[1])

    #     hid2 = self.fc4(obs)
    #     concat2 = layers.concat([hid2, act], axis=1)
    #     Q2 = self.fc5(concat2)
    #     Q2 = self.fc6(Q2)
    #     Q2 = layers.squeeze(Q2, axes=[1])
    #     return Q1, Q2

    # def Q1(self, obs, act):
    #     hid1 = self.fc1(obs)
    #     concat1 = layers.concat([hid1, act], axis=1)
    #     Q1 = self.fc2(concat1)
    #     Q1 = self.fc3(Q1)
    #     Q1 = layers.squeeze(Q1, axes=[1])

    #     return Q1
    
    
    def __init__(self):
        hid1_size = 400
        hid2_size = 300

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=1, act=None)

        self.fc4 = layers.fc(size=hid1_size, act='relu')
        self.fc5 = layers.fc(size=hid2_size, act='relu')
        self.fc6 = layers.fc(size=1, act=None)
#这里也是输出2个Q值，Q1 Q2与TD3 相同
    def value(self, obs, act):
        hid1 = self.fc1(obs)
        concat1 = layers.concat([hid1, act], axis=1)
        Q1 = self.fc2(concat1)
        Q1 = self.fc3(Q1)
        Q1 = layers.squeeze(Q1, axes=[1])

        hid2 = self.fc4(obs)
        concat2 = layers.concat([hid2, act], axis=1)
        Q2 = self.fc5(concat2)
        Q2 = self.fc6(Q2)
        Q2 = layers.squeeze(Q2, axes=[1])

        return Q1, Q2
    
    

class SACModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = SACActorModel(act_dim)
        self.critic_model = SACCriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    # def Q1(self, obs, act):
    #     return self.critic_model.Q1(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()
