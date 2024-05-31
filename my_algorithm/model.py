import parl
from parl import layers
import numpy as np
import paddle


class Model(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        # 返回一个list，包含模型所有参数的名称
        # print("actor_model.parameters(): ", self.actor_model.parameters())
        return self.actor_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid_size = 64

        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=act_dim, act='tanh')

    def policy(self, obs):
        hid = self.fc1(obs)
        means = self.fc2(hid)
        # print('means')
        # paddle.fluid.layers.Print(means)
        return means


class CriticModel(parl.Model):
    def __init__(self):
        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        # 拼接obs和act

        concat = layers.concat([obs, act], axis=1)
        # print("concat_shape:",concat.shape)
        hid = self.fc1(concat)
        # print("hid_shape:",hid.shape)
        Q = self.fc2(hid)
        # print("1Q_shape", Q.shape)
        #print("QqqqQ = ",np.array(Q[0]))

        Q = layers.squeeze(Q, axes=[1])
        # print("2Q_shape",Q.shape)
        #print("Q_squeeze = ", Q.)
        # print('Q = ')
        # paddle.fluid.layers.Print(concat, message='obs concat act')
        # paddle.fluid.layers.Print(Q, message='Q value')

        return Q
