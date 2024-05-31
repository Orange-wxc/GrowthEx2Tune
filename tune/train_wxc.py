#import gym
import csv
import os
import math
from math import sqrt

import gym
import pandas as pd
import numpy as np
import parl
from parl.utils import logger, action_mapping
import scikitplot as skplt
# from factor_analyzer.factor_analyzer import calculate_kmo
#from rlschool import make_env
from parl import layers
import random

from torch import Tensor

from maEnv.utils import get_timestamp, time_to_str
from my_algorithm.agent import Agent
from my_algorithm.model import Model
# from my_algorithm.algorithm import DDPG  # from parl.algorithms import DDPG
from parl.algorithms import DDPG  # from parl.algorithms import TD3
#from parl.algorithms import TD3  # from parl.algorithms import TD3
from my_algorithm.td3_wxc import TD3
from my_algorithm.sac_wxc import SAC 
# from parl.algorithms import DQN
# from my_algorithm.algorithm import TD3  # from parl.algorithms import TD3
# from my_algorithm.TD3 import TD3

# 用于降维的库
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from my_algorithm.td3_model import TestModel
from my_algorithm.sac_model_wxc import SACModel
from my_algorithm.agent import TestAgent
from my_algorithm.agent import SACAgent
from my_algorithm.DQN_model import Model as DQNModel
from my_algorithm.DQN_model import Agent as DQNAgent
from my_algorithm.DQN_model import DQN

from my_algorithm.replay_memory import ReplayMemory
from my_algorithm.priority_replay_memory import PrioritizedReplayMemory
import pickle
import time
import globalValue
from maEnv.env import SEEnv
from maEnv.env import CEEnv
from maEnv.env import NodesEnv
from maEnv import utils
from maEnv import datautils
from maEnv import utils


import shap


# you can customise the Hyperparameters here

# h_params of TD3:
LEARNING_RATE = 0.0001
ACTOR_LR = 0.0001  # Actor网络的 learning rate
CRITIC_LR = 0.0001 # Critic网络的 learning rate
GAMMA = 0.99  # reward 的衰减因子
TAU = 0.001  # 软更新的系数


# h_params of SAC:
H_SAC_ACTOR_LR = 0.0001     # Actor网络的 learning rate
H_SAC_CRITIC_LR = 0.0001    # Critic网络的 learning rate
H_SAC_GAMMA = 0.99          # reward 的衰减因子
H_SAC_TAU = 0.001           # 软更新的系数
H_SAC_ALPHA = 0.2          # 温度参数，决定了熵相对于奖励的相对重要性



MEMORY_SIZE = 100000  # 经验池大小
MEMORY_WARMUP_SIZE = 120 #120  # 预存一部分经验之后再开始训练
BATCH_SIZE = 16
WARMUP_MOVE_STEPS = 40
MOVE_STEPS = 20 #20
REWARD_SCALE = 1  # reward 缩放系数
# TRAIN_EPISODE = 100  # 训练的总episode数
TRAIN_EPISODE = 15 #15  # 训练的总episode数
EVAL_INTERVAL = 5   # 评估的间隔
EXPL_NOISE = 0.1   # 动作噪声方差
EXPL_NOISE_WARMUP = 0.1
POLICY_NOISE = 0.1  # Noise added to target policy during critic update
NOISE_CLIP = 0.1  # Range to clip target policy noise
POLICY_FREQ = 2
ENV_METHOD = 'TD3'
#ENV_METHOD = 'SAC'

ACTION_TREND_P = 0.95
BEST_NOW_P = 0.001
RANDOM_FOREST_REMAIN = 1
PCA_REMAIN = 10
TWO_PHASE = True

USE_SHAP = False

# 参数降维
USE_KNOBS_DR = True
# 状态向量降维
USE_STATUS_DR = True

USE_EXPERT_EXP = False

# 经验池优先经验回放开关
USE_PRIORITY_RPM = False


# 手动限定使用的经验池
USE_FIXED_RPM = False
FIXED_RPM_PATH = "./test_wxc/fixed_rpm_rw_l.txt"
FIXED_AR_PATH = "./test_wxc/fixed_ar_rw_l.csv"


# 模拟退火算法相关参数
SA_INIT_TEMPERTURE = 100
SA_ALPHA = 0.95
SA_MAX_ITER = 10
SA_BASH_SLEEP_TIME = 20
SA_BASH_BUFFER_TIME = 10    
SA_BASH_TYPE = 'sysbench'
SA_NEW_RANDOM_ACTION_P = 0.5
SA_OLD_ACTION_BIAS = 0.5





# knob tunning using simulated_annealing
def SA_train():
    
    # init db env
    env = NodesEnv()
    env.init_nodes()
    print('\n')
    
    best_qps_list = []
    best_hit_r_list = []
    best_bps_list = []
    
    # Step 1: Initialize temperature and cooling coefficient
    current_temp = SA_INIT_TEMPERTURE

    # Step 2: Choose a random solution in the parameter space
    init_params = []
    
    for se in env.se_info:
        tmp_params = []
        for key, val in se.tune_action.items():
            tmp_params.append(val[0])
        init_params.append(tmp_params)
    for ce in env.ce_info:
        tmp_params = []
        for key, val in ce.tune_action.items():
            tmp_params.append(val[0])
        init_params.append(tmp_params)
    print("[SA LOG] init params = {0}".format(init_params))
    
    
    
    
    
    # evluate init params's score
    init_reward, qps_list, hit_r_list, bps_list = SA_step(env, init_params, 1)
    
    
    current_params = best_params = init_params
    current_reward = best_reward = init_reward
    best_qps_list = qps_list
    best_bps_list = bps_list
    best_hit_r_list = hit_r_list
    
    for iter in range(SA_MAX_ITER):
        print("\n===================[SA LOG] iter {0}=======================".format(iter))
        new_params = []
        tmp_cnt = 0
        for se in env.se_info:
            tmp_params = []
            tmp_key_cnt = 0
            for key, val in se.tune_action.items():
                tmp_r = random.random()
                if tmp_r < SA_NEW_RANDOM_ACTION_P:
                    tmp_params.append(round(random.uniform((val[1]+0.00001), val[2]*0.99)))
                else:
                    tmp_params.append(round(random.uniform(max((val[1]+0.00001), current_params[tmp_cnt][tmp_key_cnt]*(1-SA_OLD_ACTION_BIAS)), min(val[1]*0.99, current_params[tmp_cnt][tmp_key_cnt]*(1+SA_OLD_ACTION_BIAS)))))
                tmp_key_cnt += 1           
            new_params.append(tmp_params)
            tmp_cnt += 1
        for ce in env.ce_info:
            tmp_params = []
            tmp_key_cnt = 0
            for key, val in ce.tune_action.items():
                tmp_r = random.random()
                if tmp_r < SA_NEW_RANDOM_ACTION_P:
                    tmp_params.append(round(random.uniform((val[1]+0.00001), val[2]*0.99)))
                else:
                    tmp_params.append(round(random.uniform(max((val[1]+0.00001), current_params[tmp_cnt][tmp_key_cnt]*(1-SA_OLD_ACTION_BIAS)), min(val[1]*0.99, current_params[tmp_cnt][tmp_key_cnt]*(1+SA_OLD_ACTION_BIAS)))))
                tmp_key_cnt += 1      
            new_params.append(tmp_params)
            tmp_cnt += 1
        print("[SA LOG] new params = {0}".format(new_params))
          
        new_reward, qps_list, hit_r_list, bps_list = SA_step(env, new_params, 0)  
        
        delta_reward = - new_reward + best_reward
        
        if delta_reward < 0:
            best_params = new_params
            best_reward = new_reward
            best_qps_list = qps_list
            best_hit_r_list = hit_r_list
            best_bps_list = bps_list
        
        
        if delta_reward < 0 or random.random() < math.exp(-delta_reward / (current_temp * 1000000)):
            current_params = new_params
            current_reward = new_reward
            print("[SA LOG] accept little worse, new_reward = {0}, new params = {1}".format(new_reward, new_params))
            
            
        tmp_cnt = 0
        for se in env.se_info:
            print("\tnode se {0}: cur_hit_r = {1}, cur_bps = {2}".format(se.uuid, hit_r_list[tmp_cnt], bps_list[tmp_cnt]))
            tmp_cnt += 1
        for ce in env.ce_info:
            if ce.is_primary == True:
                print("\tnode ce {0}: curqps = {1}, cur_hit_r = {2}, curbps = {3}".format(ce.uuid, qps_list[0], hit_r_list[tmp_cnt], bps_list[tmp_cnt]))   
            else:
                print("\tnode ce {0}: cur_hit_r = {1}, cur_bps = {2}".format(ce.uuid, hit_r_list[tmp_cnt], bps_list[tmp_cnt]))
            tmp_cnt += 1
        
        print("[SA LOG] best reward = {0}, best_param = {1}".format(best_reward, best_params))
        tmp_cnt = 0
        for se in env.se_info:
            print("\tnode se {0}: best_hit_r = {1}, best_bps = {2}".format(se.uuid, best_hit_r_list[tmp_cnt], best_bps_list[tmp_cnt]))
            tmp_cnt += 1
        for ce in env.ce_info:
            if ce.is_primary == True:
                print("\tnode ce {0}: best_qps = {1}, best_hit_r = {2}, best_bps = {3}".format(ce.uuid, best_qps_list[0], best_hit_r_list[tmp_cnt], best_bps_list[tmp_cnt]))   
            else:
                print("\tnode ce {0}: best_hit_r = {1}, best_bps = {2}".format(ce.uuid, best_hit_r_list[tmp_cnt], best_bps_list[tmp_cnt]))
            tmp_cnt += 1
        current_temp *= SA_ALPHA
        print("==============================================================\n")     

def SA_step(env, params, init_flag):

    print("[SA LOG] init_flag = {0}, apply params = {1}".format(init_flag, params))
    ces_send = []
    ses_send = []
    # 依次启动所有se和ce
    cnt = 0
    start_flag = env.start_all_nodes(params, ses_send, ces_send)
    if start_flag == False:
        print("[SA ERROR] start nodes failed!!!")
        return -1

    print("[SA LOG] loading {0} bash ... for {1} s".format(SA_BASH_TYPE, SA_BASH_SLEEP_TIME))
    sleep_time = SA_BASH_SLEEP_TIME
    buffer_time = SA_BASH_BUFFER_TIME
    utils.load_bash_remote(SA_BASH_TYPE)
    time.sleep(sleep_time)

    # 收集应用参数之后的性能信息,10s
    # 分别收集各个节点的信息并计算reward 先se再ce
    reward_qps = []
    reward_hit_r = []
    # reward_bp_size = []
    bps_all = []
    # 获取当前节点状态 计算奖励 更新节点状态
    try:
        cnt = 0
        for se in env.se_info:
            # q_after_se = utils.get_node_qps(se)
            h_after_se = utils.get_se_hr(se)
            bps_se = utils.get_bps(se.ip, se.port)
            bps_all.append(bps_se)
            if init_flag:
                se.hit_before = h_after_se
            reward_hit_r.append(h_after_se)
            cnt += 1

        for ce in env.ce_info:
            
            h_after_ce = utils.get_node_hr(ce)
            if ce.is_primary == True:
                q_after_ce = utils.get_node_qps(ce)
                if init_flag:
                    ce.hit_before = h_after_ce
                    ce.qps_before = q_after_ce
                reward_qps.append(q_after_ce)
                
            bps_ce = utils.get_bps(ce.ip, ce.port)
            bps_all.append(bps_ce)
            reward_hit_r.append(h_after_ce)
            cnt += 1
            
    except Exception as e:
        print("[SA ERROR] get node status failed!!! err = {0}".format(e))
        return -1,[],[],[]

    present_bps_all = np.sum(bps_all)
    present_qps_all = np.sum(reward_qps)
    present_hit_r_all = np.sum(reward_hit_r)
    
    if init_flag == 1:
        env.init_bps_all = present_bps_all
        env.init_qps_all = present_qps_all
        env.init_hit_r_all = present_hit_r_all
        reward = 0
        print("[SA LOG] init params: init_bps_all = {0}, init_qps_all = {1}, init_hit_r_all = {2}".format(present_bps_all, present_qps_all, present_hit_r_all))
        # env.close_all_nodes()
        return reward, reward_qps, reward_hit_r, bps_all
    else:
        delta_b = (env.init_bps_all - present_bps_all) * 1.0 / env.init_bps_all
        delta_h = (present_hit_r_all - env.init_hit_r_all) * 1.0 / env.init_hit_r_all
        delta_q = (present_qps_all - env.init_qps_all) * 1.0 / env.init_qps_all
    
        wq = 0.85
        wb = 0.05
        wh = 0.1
        
        reward = delta_q * wq + delta_b * wb + delta_h * wh
        reward = reward * 1000000
        
        print("[SA LOG] step reward = {0}, delta_q = {1}, delta_b = {2}, delta_h = {3}".format(reward, delta_q, delta_b, delta_h))
        # env.close_all_nodes()
        
        return reward, reward_qps, reward_hit_r, bps_all
            
def record_data(type, qps_list, bps_list, hit_r_list, params, env, init_flag, iter):
    
    
    
    if (init_flag):
        f_qps = open("./test_sa/qps_sa.txt", "w+", encoding="utf-8")
        f_qps.close()
        f_qps_b = open("./test_sa/qps_sa_best.txt", "w+", encoding="utf-8")
        f_qps_b.close()
        
        f_action = open("./test_sa/action_sa.txt", "w+", encoding="utf-8")
        f_action.close()
        f_action_b = open("./test_sa/action_sa_best.txt", "w+", encoding="utf-8")
        f_action_b.close()
        
        f_bps = open("./test_sa/bps_sa.txt", "w+", encoding="utf-8")
        f_bps.close()
        f_bps_b = open("./test_sa/bps_sa_best.txt", "w+", encoding="utf-8")
        f_bps_b.close()
        
        f_hit_r = open("./test_sa/hit_r_sa.txt", "w+", encoding="utf-8")
        f_hit_r.close()
        f_hit_r_b = open("./test_sa/hit_r_sa_best.txt", "w+", encoding="utf-8")
        f_hit_r_b.close()
        
    
    
    suffix = ""
    if type == "current":
        suffix = ""
    elif type == "best":
        suffix = "_best"
        
    
    with open("./test_model/qps_sa" + suffix + ".txt", "a+", encoding="utf-8") as f_qps:
        f_qps.write("[iter {0}]\t".format(iter))
        for ce in env.ce_info:
            if ce.is_primary == True:
	    # $node1@qps=xxx$node2@qps=xxx
                f_qps.write("node" + str(ce.uuid) + "@qps=" + str(qps_list[0]) + "\t")
        f_qps.write("\n")
        f_qps.close()
    
    with open("./test_model/bps_sa" + suffix + ".txt", "a+", encoding="utf-8") as f_bps:
        f_bps.write("[iter {0}]\t".format(iter))
        tmp_cnt = 0
        for se in env.se_info:
            f_bps.write("node" + str(se.uuid) + "@bps=" + str(bps_list[tmp_cnt]) + "\t")
            tmp_cnt += 1
        for ce in env.ce_info:
            f_bps.write("node" + str(ce.uuid) + "@bps=" + str(bps_list[tmp_cnt]) + "\t")
            tmp_cnt += 1
        f_bps.write("\n")
        f_bps.close()
        
    with open("./test_model/hit_r_sa" + suffix + ".txt", "a+", encoding="utf-8") as f_hit_r:
        f_hit_r.write("[iter {0}]\t".format(iter))
        tmp_cnt = 0
        for se in env.se_info:
            f_hit_r.write("node" + str(se.uuid) + "@hit_r=" + str(hit_r_list[tmp_cnt]) + "\t")
            tmp_cnt += 1
        for ce in env.ce_info:
            f_hit_r.write("node" + str(ce.uuid) + "@hit_r=" + str(hit_r_list[tmp_cnt]) + "\t")
            tmp_cnt += 1
        f_hit_r.write("\n")
        f_hit_r.close()
        
    with open("./test_model/action_sa" + suffix + ".txt", "a+", encoding="utf-8") as f_action:
        f_action.write("[iter {0}]\t".format(iter))
        tmp_cnt = 0
        for se in env.se_info:
            f_action.write("node" + str(se.uuid) + "@action=" + str(params[tmp_cnt]) + "\t")
            tmp_cnt += 1
        for ce in env.ce_info:
            f_action.write("node" + str(ce.uuid) + "@action=" + str(params[tmp_cnt]) + "\t")
            tmp_cnt += 1
        f_action.write("\n")
        f_action.close()
    





# 改造
# 目标：将原来以函数为主体的训练方法（面向过程）改造成以模型的类为主体（面向对象）
class MATuner:
    
    
    
    def __init__(self, model_type, model_name):
        self.model_type = model_type            # model的类型：目前支持两种，强化学习方法（rl）/启发时算法（ha）
        self.model_name = model_name            # model的具体名称：比如TD3、SA
        self.model = None
        self.agent = None
        self.env = None
        self.rpm = None
        self.pca = None
        self.TD3_logger = None
        
        
    def run_episode(self, model = None, agent = None, env = None, rpm = None, pca = None, f_step_reward = None, TD3_logger = None, mode = 1):
        
        if model == None:   model = self.model
        if agent == None:   agent = self.agent
        if env == None:     env = self.env
        if rpm == None:     rpm = self.rpm
        if pca == None:     pca = self.pca
        if f_step_reward == None:   f_step_reward = open("./test_model/scores.txt", "w+", encoding="utf-8")
        if TD3_logger == None:      TD3_logger = self.TD3_logger
            
            
        globalValue.EVAL_TEST = False
        obs = np.array([0])
        reset_val = True
        env.start_time = time.time()
        while reset_val:
            obs, reset_val = env.reset()
        rear_obs = obs
        p_exp = env.action_trend_choice
        # 某轮总得分
        total_reward = 0
        reward_a = -1
        done = False
        steps = 0
        max_action = 1
        raw_action = 0
        action = []
        if env.state == 1:
            env.episode += 1
        accumulate_loss = 0

        if env.info == 'CE':
            TD3_logger.info("\n[{} Env initialized][qps: {}, hit_ratio: {}, buffer_size: {}]".format(
                env.method, env.qps_t0, env.hit_t0, env.bp_size_t0))
        elif env.info == 'NODES':
            s = '[' + env.method + 'Env initialized]'
            s += '[ ' + str(env.se_num) + ' ses:'
            for se in env.se_info:
                s += '{se' + str(se.uuid) + ', hit_ratio: ' + str(se.hit_t0) + ', buffer_pool_size: ' + str(se.bp_size_t0) + '}'
            s += '][ ' + str(env.ce_num) + ' ces:'
            for ce in env.ce_info:
                if ce.is_primary == True:
                    s += '{ce' + str(ce.uuid) + ', qps: ' + str(ce.qps_t0) + ', hit_ratio: ' + str(ce.hit_t0 )+ ', buffer_pool_size: ' + str(ce.bp_size_t0) + '}'
                else:
                    s += '{ce' + str(ce.uuid) + ', hit_ratio: ' + str(
                        ce.hit_t0) + ', buffer_pool_size: ' + str(ce.bp_size_t0) + '}'
            s += ']'
            TD3_logger.info("\n{}".format(s))

        # # 导入已有模型
        # agent.restore("./test_model/test_save/test_save_1681301158.ckpt")
        while True:
            steps += 1
            if env.state == 0:
                print('==========>WARMUP-step', steps)
            elif env.state == 1:
                print('==========>Episode{}-step{} '.format(env.episode, steps))
            else:
                print('==========>Eval-step', steps)

            batch_obs = obs
            if done:
                print('+++ reset step +++')
                batch_obs = rear_obs
                obs = rear_obs
            done = False
            rear_obs = obs
            # 当预热阶段过去，认为pca模型已训练完毕，此时神经网络的状态空间已经被压缩，
            # 但收集到的原始状态数据维度始终不变，因此需要先用pca进行降维，在结合rpm的数据进行标准化
            if env.state != 0 and TWO_PHASE == True and USE_STATUS_DR == True:
                batch_obs = pca.transform(batch_obs.reshape(1, -1))
                batch_obs = np.array(batch_obs).flatten()
                obs = batch_obs
            # 输入agent时对state归一化
            # print('batch_obs = ', batch_obs)
            input_obs = agent.normalizer(batch_obs, rpm)
            print("Normalize obs:", input_obs)

            if env.method == 'DQN':
                raw_action = agent.sample(input_obs.astype('float32'))
                action = env.explian_DQN_action(raw_action)
                print('raw action:', raw_action)
                print('action:', action)
            elif env.method == 'TD3':
                action = agent.predict(input_obs.astype('float32'))
            elif env.method == 'SAC':
                action = agent.sample(input_obs.astype('float32'))
            if len(env.best_action_record) == 0 or len(env.best_action_record) != len(action):
                env.best_action_record = action
            if len(env.all_last_action) == 0 or len(env.all_last_action) != len(action):
                env.all_last_action = action
            # print('action by agent predict: ', action)
            # DDPG
            if env.method == 'DDPG':
                if env.state == 0:
                    action = np.clip(np.random.normal(action, EXPL_NOISE_WARMUP), -1.0, 1.0)
                else:
                    action = np.clip(np.random.normal(action, EXPL_NOISE), -1.0, 1.0)
            # 20240305: @TODO 修改第一阶段的代码让第一阶段也使用专家知识控制器
            # TD3
            if env.method == 'TD3':
                # Add exploration noise, and clip to [-max_action, max_action]
                if env.state == 0:
                    action = np.clip(
                        np.random.normal(action, EXPL_NOISE_WARMUP * max_action), -max_action,
                        max_action)
                else:
                    action = np.clip(
                        np.random.normal(action, EXPL_NOISE * max_action), -max_action,
                        max_action)
            print('action by clip: ', action)
            action_trend = []
            # 在策略网络的输出、best_action_now、expert_exp中概率选择，这时需要一个随机数，看落在哪个区间内，
            if env.state == 1 or mode == 2:
                # 正式训练时考虑best_action_now
                if env.expert_exp == True:
                    action_trend = datautils.all_nodes_labels_to_action_trend(env)
                    # 实现概率衰减
                    p_exp *= 0.995
                    print('human action pick_p: ', p_exp)
                    print('human action_trend: ', action_trend)
                    # env.action_trend_choice *= 0.9995
                action, hit_cnt = utils.action_with_knowledge_and_best_now(action, env.best_action_record, action_trend,
                                                                env.best_action_choice, p_exp, env.all_last_action)
                if hit_cnt >= 0:
                    env.human_exp_hitcnt += 1
            print('action after shape:', action)


            # buffer pool don't change over 1/2 or 2
            if env.info == 'CE':
                if env.last_action != -2:
                    bp_size = action_mapping(action[0], env.min_info[0], env.max_info[0])
                    # np.random.seed(time.time())
                    if bp_size > env.last_bp_size * 2:
                        # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                        # print("random.randint big")
                        bp_size = np.random.randint(env.last_bp_size, env.last_bp_size * 2)
                    elif bp_size < env.last_bp_size * 0.5:
                        # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                        # print("random.randint small")
                        bp_size = np.random.randint(env.last_bp_size * 0.5, env.last_bp_size)
                    action[0] = utils.real_action_to_action(bp_size, env.min_info[0], env.max_info[0])
            elif env.info == 'NODES':
                if env.last_action != -2:
                    # 依次修改node
                    index = 0
                    for se in env.se_info:
                        # buf_key = str(se.uuid) + '#buffer_pool_size'
                        bp_size_se = action_mapping(action[index], se.tune_action['buffer_pool_size'][1],
                                                    se.tune_action['buffer_pool_size'][2])
                        last_se_bp_size = se.last_bp_size
                        if bp_size_se > last_se_bp_size * 2:
                            # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                            # print("random.randint big")
                            bp_size_se = np.random.randint(last_se_bp_size, last_se_bp_size * 2)
                        elif bp_size_se < last_se_bp_size * 0.5:
                            # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                            # print("random.randint small")
                            bp_size_se = np.random.randint(last_se_bp_size * 0.5, last_se_bp_size)
                        action[index] = utils.real_action_to_action(bp_size_se,
                                                                    se.tune_action['buffer_pool_size'][1],
                                                                    se.tune_action['buffer_pool_size'][2])
                        index += len(se.tune_action)

                    for ce in env.ce_info:
                        # buf_key = ce.uuid + '#buffer_pool_size'
                        bp_size_ce = action_mapping(action[index], ce.tune_action['buffer_pool_size'][1],
                                                    ce.tune_action['buffer_pool_size'][2])
                        last_ce_bp_size = ce.last_bp_size
                        # ce
                        if bp_size_ce > last_ce_bp_size * 2:
                            # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                            # print("random.randint big")
                            bp_size_ce = np.random.randint(last_ce_bp_size, last_ce_bp_size * 2)
                        elif bp_size_ce < last_ce_bp_size * 0.5:
                            # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                            # print("random.randint small")
                            bp_size_ce = np.random.randint(last_ce_bp_size * 0.5, last_ce_bp_size)
                        action[index] = utils.real_action_to_action(bp_size_ce,
                                                                    ce.tune_action['buffer_pool_size'][1],
                                                                    ce.tune_action['buffer_pool_size'][2])
                        index += len(ce.tune_action)

            # print('action after filter: ', action)
            TD3_logger.info("\n[{}] Action: {}".format(env.method, action))
            next_obs, reward, done, info = env.step(action)
            if done:
                print('+++ reset step +++')
                continue
            actual_obs = next_obs

            if env.state != 0 and TWO_PHASE == True and USE_STATUS_DR == True:
                next_obs = pca.transform(next_obs.reshape(1, -1))
                next_obs = np.array(next_obs).flatten()
            print("-----------------------------------")
            print("obs = ", obs)
            print("obs.type = ", type(obs))
            print("-----------------------------------")
            cal_td_e_obs = agent.normalizer(obs, rpm)
            cal_td_e_next_obs = agent.normalizer(next_obs, rpm)
            if env.method != 'DQN':
                if USE_PRIORITY_RPM:
                    action = [action]  # 方便存入replay memory
                    obs_new = np.array([cal_td_e_obs]).astype('float32')
                    action_new = np.array([action]).astype('float32').reshape(1, env.action_dim)
                    reward_new = np.array([REWARD_SCALE * reward]).astype('float32')
                    next_obs_new = np.array([cal_td_e_next_obs]).astype('float32')
                    done_new = np.array([done]).astype('float32')
                    #print("action_new = ", action_new)
                    #print("obs_new = ", obs_new)
                    #print("reward_new = ", reward_new)
                    #print("next_obs_new = ", next_obs_new)
                    #print("done_new = ", done_new)
                    trans_reward_new = reward_new.copy()
                    trans_reward_new[trans_reward_new>0] = trans_reward_new[trans_reward_new>0]/1000000.0
                    
                    td_error = self.agent.cal_td_error(obs_new, action_new, trans_reward_new, next_obs_new, done_new)
                    #td_error = self.agent.cal_td_error(obs_new, action_new, reward_new, next_obs_new, done_new)
                    #td_error = self.agent.cal_td_error(obs_new, action_new, reward_new, next_obs_new, done_new)

                    rpm.append(td_error, (obs, action, REWARD_SCALE * reward, next_obs, done))
                else:
                    action = [action]
                    rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))
            else:
                rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))
                rpm.append_DQN_action(raw_action)
            # 平均得分
            mean_step_reward = float(env.score) / env.steps
            f_step_reward.write("scores=" + str(mean_step_reward) + "\n")
            f_step_reward.flush()

            if len(rpm) >= BATCH_SIZE:# and (steps % 5) == 0:
                if env.method != 'DQN':
                    #(batch_obs, batch_action, batch_reward, batch_next_obs,
                    #batch_done) = rpm.sample(BATCH_SIZE)
                    idxs = None
                    batch_obs = None
                    batch_action = None
                    batch_reward = None
                    batch_next_obs = None
                    batch_done = None
                    if USE_PRIORITY_RPM:
                        (idxs, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(BATCH_SIZE)
                    else:
                        (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(BATCH_SIZE)

                else:
                    (batch_obs, batch_action, batch_reward, batch_next_obs,
                    batch_done) = rpm.DQN_sample(BATCH_SIZE)
                # 这里维度已经缩小了，所以不再需要pca降维，降维应该发生在收集到原始数据库状态后，但是需要进行数据标准化！！
                batch_obs = agent.normalizerBatch(batch_obs, rpm)
                batch_next_obs = agent.normalizerBatch(batch_next_obs, rpm)
                # DDPG
                if env.method == 'DDPG':
                    critic_cost = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                # TD3
                if env.method == 'TD3' or env.method == 'SAC':
                    #if USE_PRIORITY_RPM:
                        #new_td_error = self.agent.cal_td_error(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                        #print("*************************************")
                        #print("*new_td_error_before_learn = {0}".format(new_td_error))
                        #print("*************************************")
                    
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("agent learn, alg = {0}, time = run_epsisode".format(env.method))
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


                    actor_cost, critic_cost = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                                                        batch_done)
                    if USE_PRIORITY_RPM:
                        trans_batch_reward = batch_reward.copy()
                        trans_batch_reward[trans_batch_reward>0] = trans_batch_reward[trans_batch_reward>0] / 1000000.0
                        new_td_error = self.agent.cal_td_error(batch_obs, batch_action, trans_batch_reward, batch_next_obs, batch_done)
                        #new_td_error = self.agent.cal_td_error(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                        print("*************************************")
                        print("*new_td_error = {0}, critic_cost = {1}".format(new_td_error, critic_cost))
                        print("*************************************")
                    #sum_error = 0
                    #for i in range(BATCH_SIZE):
                        #print("+++++ obs = ", batch_obs)
                        #print("+++++ action = ", batch_action)
                        #print("+++++ reward = ", batch_reward)
                        #print("+++++ next_obs = ", batch_next_obs)
                        #print("+++++ done = ", batch_done)
                        #obs_new = np.array([batch_obs[i]]).astype('float32')
                        #action_new = np.array([batch_action[i]]).astype('float32').reshape(1, env.action_dim)
                        #reward_new = np.array([batch_reward[i]]).astype('float32')
                        #next_obs_new = np.array([batch_next_obs[i]]).astype('float32')
                        #done_new = np.array([batch_done[i]]).astype('float32')
                
                        #tmp_td_error = self.agent.cal_td_error(obs_new, action_new, reward_new, next_obs_new, done_new)
                        #sum_error += tmp_td_error
                        #print("=================== single td error = ", tmp_td_error)
                    #print("********************** avg td error = ", sum_error / BATCH_SIZE)
                        for i in range(BATCH_SIZE):
                            idx = idxs[i]
                            rpm.update(idx, new_td_error[i])                

                # DQN
                if env.method == 'DQN':
                    critic_cost = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

                if env.state != 0:
                    print("1111111111111111111112222222222222222222222", critic_cost)
                    if env.method == 'SAC':
                        accumulate_loss += critic_cost
                    else:
                        accumulate_loss += critic_cost[0]
                    critic_cost_mean = accumulate_loss / env.steps
                    with open("./test_model/critic_loss.txt", "a+", encoding="utf-8") as loss_f:
                        loss_f.write("critic_loss="+str(critic_cost_mean)+"\n")

            if env.info == 'CE':
                TD3_logger.info(
                    "\n[{}][Episode: {}][Step: {}][Metric qps:{} hit_ratio:{} buffer_size:{}]Reward: {} Score: {} Done: {}".format(
                        env.method, env.episode, steps, env.last_qps, env.last_hr, env.last_bp_size, reward, total_reward, done
                    ))
            elif env.info == 'NODES':
                s = '[' + env.method + '][Episode: ' + str(env.episode) + '][Step: ' + str(steps) + ']'
                s += '[ ' + str(env.se_num) + ' ses:'
                for se in env.se_info:
                    s += '{se' + str(se.uuid) + ', hit_ratio: ' + str(se.last_hr) + ', buffer_pool_size: ' + str(se.last_bp_size) + '}'
                s += '][ ' + str(env.ce_num) + ' ces:'
                for ce in env.ce_info:
                    if ce.is_primary == True:
                        s += '{ce' + str(ce.uuid) + ', qps: ' + str(ce.last_qps) + ', hit_ratio: ' + str(
                            ce.last_hr) + ', buffer_pool_size: ' + str(ce.last_bp_size) + '}'
                    else:
                        s += '{ce' + str(ce.uuid) + ', hit_ratio: ' + str(
                            ce.last_hr) + ', buffer_pool_size: ' + str(ce.last_bp_size) + '}'
                s += ']'
                s += 'Reward: ' + str(reward) + ', Score: ' + str(total_reward) + ', Done: ' + str(done)
                TD3_logger.info("\n{}".format(s))

            # if steps % 10 == 0:
            #     # 保存模型
            #     ckpt = './test_model/test_save/test_save_{}'.format(int(time.time()))
            #     # f = open("./test_model/test_save/ce_rpm.txt", "wb")
            #     print('-----------save_model-----------')
            #     print('ckpt = ', ckpt)
            #
            #     # DDPG
            #     if env.method == 'DDPG':
            #         agent.save(save_path=ckpt+'.cpkt')
            #     # TD3
            #     if env.method == 'TD3':
            #         agent.save(save_path=ckpt+'_predict.ckpt', mode='predict')
            #         agent.save(save_path=ckpt+'_train_actor.ckpt', mode='train_actor')
            #         agent.save(save_path=ckpt+'_train_critic.ckpt', mode='train_critic')

            obs = actual_obs
            total_reward += reward
            if env.state == 0:
                if steps >= WARMUP_MOVE_STEPS or total_reward < -50 or total_reward > 20000000:
                    print('WARMUP DONE : steps = ', steps)
                    break
            else:
                if steps >= MOVE_STEPS or done or total_reward < -50 or total_reward > 20000000:
                    print('DONE : steps = ', steps)
                    break
        env.end_time = time.time()
        return total_reward, steps

    def evaluate_ce(self, env = None, agent = None, rpm = None, pca = None, TD3logger = None):
        
        if agent == None:   agent = self.agent
        if env == None:     env = self.env
        if rpm == None:     rpm = self.rpm
        if pca == None:     pca = self.pca
        if TD3logger == None:      TD3logger = self.TD3_logger
        
        # eval_reward = []
        total_reward = 0
        env.state = 2
        max_reward = -1
        recommand_action = ''
        globalValue.EVAL_TEST = True
        done = False
        env.start_time = time.time()
        for i in range(1):
            reset_val = True
            while reset_val:
                obs, reset_val = env.reset()
            rear_obs = obs
            if env.info == 'CE':
                TD3logger.info("\n[{} Env initialized][qps: {}, hit_ratio: {}, buffer_size: {}]".format(
                    env.method, env.qps_t0, env.hit_t0, env.bp_size_0))
            elif env.info == 'NODES':
                s = '[' + env.method + 'EVAL Env initialized]'
                s += '[ ' + str(env.se_num) + ' ses:'
                for se in env.se_info:
                    s += '{se' + str(se.uuid) + ', hit_ratio: ' + str(se.hit_t0) + ', buffer_pool_size: ' + str(se.bp_size_t0) + '}'
                s += '][ ' + str(env.ce_num) + ' ces:'
                for ce in env.ce_info:
                    if ce.is_primary == True:
                        s += '{ce' + str(ce.uuid) + ', qps: ' + str(ce.qps_t0) + ', hit_ratio: ' + str(
                            ce.hit_t0) + ', buffer_pool_size: ' + str(ce.bp_size_t0) + '}'
                    else:
                        s += '{ce' + str(ce.uuid) + ', hit_ratio: ' + str(
                            ce.hit_t0) + ', buffer_pool_size: ' + str(ce.bp_size_t0) + '}'
                s += ']'
                TD3logger.info("\n{}".format(s))

            # 给负载预热
            # time.sleep(10)
            steps = 1
            env.eval += 1
            # f = open("./1ce/recommand_knobs/knobs04.txt", 'a')
            # f.write('=======================The recommand knobs==========================')
            # timestamp = get_timestamp()
            # date_str = time_to_str(timestamp)
            # f.write(date_str)
            # f.write("\n")
            # f.close()

            while True:
                print('==========>Eval{}-step{} '.format(env.eval, steps))
                batch_obs = obs
                if done:
                    batch_obs = rear_obs
                done = False
                if TWO_PHASE == True and USE_STATUS_DR == True:
                    batch_obs = pca.transform(batch_obs.reshape(1, -1))
                batch_obs = np.array(batch_obs).flatten()
                # 输入agent时对state标准化
                input_obs = agent.normalizer(batch_obs, rpm)
                # print("Normalize obs:", input_obs)

                if env.method != 'DQN':
                    action = agent.predict(input_obs.astype('float32'))
                    action = np.clip(action, -1.0, 1.0)
                else:
                    raw_action = agent.sample(input_obs.astype('float32'))
                    action = env.explian_DQN_action(raw_action)

                # bpsize一次的变化最好不要超过2或1/2，但env reset时无需遵循此规则
                if env.info == 'CE':
                    if env.last_action != -2:
                        bp_size = action_mapping(action[0], env.min_info[0], env.max_info[0])
                        # np.random.seed(time.time())
                        if bp_size > env.last_bp_size * 2:
                            # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                            # print("random.randint big")
                            bp_size = np.random.randint(env.last_bp_size, env.last_bp_size * 2)
                        elif bp_size < env.last_bp_size * 0.5:
                            # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                            # print("random.randint small")
                            bp_size = np.random.randint(env.last_bp_size * 0.5, env.last_bp_size)
                        action[0] = utils.real_action_to_action(bp_size, env.min_info[0], env.max_info[0])
                elif env.info == 'NODES':
                    if env.last_action != 0:
                        # 依次修改node
                        index = 0
                        for se in env.se_info:
                            # buf_key = se.uuid + '#buffer_pool_size'
                            bp_size_se = action_mapping(action[index], se.tune_action['buffer_pool_size'][1], se.tune_action['buffer_pool_size'][2])
                            last_se_bp_size = se.last_bp_size
                            if bp_size_se > last_se_bp_size * 2:
                                # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                                # print("random.randint big")
                                bp_size_se = np.random.randint(last_se_bp_size, last_se_bp_size * 2)
                            elif bp_size_se < last_se_bp_size * 0.5:
                                # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                                # print("random.randint small")
                                bp_size_se = np.random.randint(last_se_bp_size * 0.5, last_se_bp_size)
                            action[index] = utils.real_action_to_action(bp_size_se,
                                                                        se.tune_action['buffer_pool_size'][1],
                                                                        se.tune_action['buffer_pool_size'][2])
                            index += len(se.tune_action)

                        for ce in env.ce_info:
                            # buf_key = ce.uuid + '#buffer_pool_size'
                            bp_size_ce = action_mapping(action[index], ce.tune_action['buffer_pool_size'][1],
                                                        ce.tune_action['buffer_pool_size'][2])
                            last_ce_bp_size = ce.last_bp_size
                            # ce
                            if bp_size_ce > last_ce_bp_size * 2:
                                # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                                # print("random.randint big")
                                bp_size_ce = np.random.randint(last_ce_bp_size, last_ce_bp_size * 2)
                            elif bp_size_ce < last_ce_bp_size * 0.5:
                                # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                                # print("random.randint small")
                                bp_size_ce = np.random.randint(last_ce_bp_size * 0.5, last_ce_bp_size)
                            action[index] = utils.real_action_to_action(bp_size_ce,
                                                                        ce.tune_action['buffer_pool_size'][1],
                                                                        ce.tune_action['buffer_pool_size'][2])
                            index += len(ce.tune_action)

                # print('action after filter: ', action)

                TD3logger.info("\n[{}] Action: {}".format(env.method, action))

                # #action = [action]
                # action_record = utils.action_change(env, action)
                # # 将推荐参数封装为发送数据标准格式
                # if env.info == 'CE':
                #     var_names = list(env.all_actions.keys())
                #     send_variables = utils.get_set_variables_string(var_names, action_record, False, 3)
                # elif env.info == 'NODES':
                #     var_names = list(env.all_actions.keys())
                #     send_variables_se, send_variables_ce = utils.get_set_variables_string_nodes(var_names, action_record, 3)
                #     send_variables = send_variables_se + ' ' + send_variables_ce

                steps += 1
                next_obs, reward, done, info = env.step(action)
                if done:
                    continue
                rear_obs = obs
                obs = next_obs
                total_reward += reward

                if max_reward < reward:
                    max_reward = reward
                    # recommand_action = send_variables

                if env.info == 'CE':
                    TD3logger.info(
                        "\n[{}][Eval: {}][Step: {}][Metric qps:{} hit_ratio:{} buffer_size:{}]Reward: {} Score: {} Done: {}".format(
                            env.method, env.eval, steps, env.last_qps, env.last_hr, env.last_bp_size, reward,
                            total_reward, done
                        ))
                elif env.info == 'NODES':
                    s = '[' + env.method + '][Eval: ' + str(env.eval) + '][Step: ' + str(steps) + ']'
                    s += '[ ' + str(env.se_num) + ' ses:'
                    for se in env.se_info:
                        s += '{se' + str(se.uuid) + ', hit_ratio: ' + str(se.last_hr) + ', buffer_pool_size: ' + str(se.last_bp_size) + '}'
                    s += '][ ' + str(env.ce_num) + ' ces:'
                    for ce in env.ce_info:
                        if ce.is_primary == True:
                            s += '{ce' + str(ce.uuid) + ', qps: ' + str(ce.last_qps) + ', hit_ratio: ' + str(
                                ce.last_hr) + ', buffer_pool_size: ' + str(ce.last_bp_size) + '}'
                        else:
                            s += '{ce' + str(ce.uuid) + ', hit_ratio: ' + str(
                                ce.last_hr) + ', buffer_pool_size: ' + str(ce.last_bp_size) + '}'
                    s += ']'
                    s += 'Reward: ' + str(reward) + ', Score: ' + str(total_reward) + ', Done: ' + str(done)
                    TD3logger.info("\n{}".format(s))

                if done or steps >= MOVE_STEPS:
                    # env.unuse_step = 0
                    break
            # eval_reward.append(total_reward)
        # return np.mean(eval_reward)
        #
        # f = open("./1ce/recommand_knobs/knobs04.txt", 'a')
        # f.write(recommand_action)
        # f.write('        reward = ')
        # f.write(str(max_reward))
        # f.write("\n")
        # f.close()
        globalValue.EVAL_TEST = False
        env.state = 1
        env.end_time = time.time()
        return total_reward

    def evaluate_t(self, flag):
        if flag:
            print('------SE Train thread start...------')
        else:
            print('------CE Train thread start...------')
        if flag:
            env = SEEnv()
        else:
            env = CEEnv()

        #
        # obs_dim = env.state_dim
        # act_dim = env.action_dim
        # globalValue.EVAL_TEST = True
        #
        # model = Model(act_dim)
        # algorithm = DDPG(
        #     model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        # agent = Agent(algorithm, obs_dim, act_dim)

        obs_dim = env.state_dim
        act_dim = env.action_dim
        max_action = 1.0

        model = TestModel(act_dim, max_action)
        algorithm = TD3(
            model,
            max_action=max_action,
            gamma=GAMMA,
            tau=TAU,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            policy_noise=POLICY_NOISE,  # Noise added to target policy during critic update
            noise_clip=NOISE_CLIP,  # Range to clip target policy noise
            policy_freq=POLICY_FREQ
        )
        agent = TestAgent(algorithm, obs_dim, act_dim)

        rpm = None
        if not USE_PRIORITY_RPM:
            rpm = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim)
        else:
            rpm = PrioritizedReplayMemory(MEMORY_SIZE, act_dim, obs_dim)


        # 导入已有模型
        agent.restore("/home/fox/subject/delta_qps/tune (复件)/1ce/model_dir/ce_steps_1617630080.ckpt")


        if flag:
            f = open("./1se/rpm_dir/se_rpm.txt", "rb")
        else:
            f = open("./1ce/rpm_dir/ce_rpm_full0330.txt", "rb")
        rpm = pickle.load(f)
        f.close()

        eval_reward = self.evaluate_ce(env, agent)
        print('Test reward:{}'.format(eval_reward))



    '''
        验证高质量数据集（经验值）对前期参数重要性排序的重要性：
            1. 在前期不使用高质量的数据集（原方法）
            2. 在前期使用高质量的数据集：
                (1) 在前期训练时加入专家知识控制器
                (2) 在前期训练时使用已经训练好的模型获取经验
    '''
    
    '''
        [param] mode: 1: 原方法，筛选参数前不刻意使用高质量数据集
                      2: 筛选参数前使用高质量数据集，在前期训练时加入专家知识控制器
                      3: 筛选参数前使用高质量数据集，在前期训练时使用已经训练好的模型获取经验


    '''
    def train(self, mode = 1):

        print("------MultiNodes Train (use {0} {1}) thread start...------".format(self.model_type, self.model_name))
        if self.model_type == "RL":
            pass
        else:
            if self.model_name == "SA":
                return SA_train()
            else:
                return SA_train()
        
        self.env = env = NodesEnv()
        env.init_nodes()
        if env.method == 'DQN':
            env.init_env_for_DQN()

        env.method = self.model_name
        env.expert_exp = USE_EXPERT_EXP
        obs_dim = env.state_dim
        act_dim = env.action_dim
        max_action = 0.99

        if env.expert_exp == True:
            env.action_trend_choice = ACTION_TREND_P
        else:
            env.action_trend_choice = 0
        env.best_action_choice = 0

        # DQN
        if env.method == 'DQN':
            self.model =  DQNModel(env.DQN_act_dim)
            algorithm = DQN(self.model, act_dim=env.DQN_act_dim, gamma=GAMMA, lr=LEARNING_RATE)
            self.agent =  DQNAgent(
                algorithm,
                obs_dim=obs_dim,
                act_dim=env.DQN_act_dim,
                e_greed=0.1,  # 有一定概率随机选取动作，探索
                e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低
            print('agent obs_dim{}, act_dim{}'.format(self.agent.obs_dim, self.agent.act_dim))

        # DDPG
        if env.method == 'DDPG':
            self.model =  Model(act_dim)
            algorithm = DDPG(
                self.model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
            self.agent =  Agent(algorithm, obs_dim, act_dim)

        # TD3
        if env.method == 'TD3':
            self.model =  TestModel(act_dim, max_action)
            algorithm = TD3(
                self.model,
                max_action=max_action,
                gamma=GAMMA,
                tau=TAU,
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
                policy_noise=POLICY_NOISE,  # Noise added to target policy during critic update
                noise_clip=NOISE_CLIP,  # Range to clip target policy noise
                policy_freq=POLICY_FREQ
            )
            self.agent = TestAgent(algorithm, obs_dim, act_dim)
            #self.agent.algorithm.cal_td_error()

        # SAC
        if env.method == 'SAC':
            self.model =  SACModel(act_dim)
            algorithm = SAC(
                actor=self.model.actor_model,
                critic=self.model.critic_model,
                max_action=max_action,
                alpha=H_SAC_ALPHA,
                gamma=H_SAC_GAMMA,
                tau=H_SAC_TAU,
                actor_lr=H_SAC_ACTOR_LR,
                critic_lr=H_SAC_CRITIC_LR,
            )
            self.agent = SACAgent(algorithm, obs_dim, act_dim)

        # 打开存储信息的文件，准备写入
        f_mode = 'w+'
        f_step_reward = open("./test_model/scores.txt", f_mode, encoding="utf-8")
        f_time_store = open("./test_model/timestore.txt", f_mode, encoding="utf-8")

        f_qps_store = open("./test_model/qps_store.txt", f_mode, encoding="utf-8")
        f_qps_store.close()

        f_critc_loss = open("./test_model/critic_loss.txt", f_mode, encoding="utf-8")
        f_critc_loss.close()
        f_pr = open("./test_model/PCA&RF.txt", f_mode, encoding="utf-8")
        f_pr.close()

        if env.info == 'CE':
            f_bp_size = open("./test_model/buffer_pool_size.txt", f_mode, encoding="utf-8")
            f_bp_size.close()
            f_hit_ratio = open("./test_model/hit_ratio.txt", f_mode, encoding="utf-8")
            f_hit_ratio.close()
        if env.info == 'NODES':
            f_bp_size_se = open("./test_model/buffer_pool_size_se.txt", f_mode, encoding="utf-8")
            f_bp_size_se.close()
            f_hit_ratio_se = open("./test_model/hit_ratio_se.txt", f_mode, encoding="utf-8")
            f_hit_ratio_se.close()
            f_bp_size_ce = open("./test_model/buffer_pool_size_ce.txt", f_mode, encoding="utf-8")
            f_bp_size_ce.close()
            f_hit_ratio_ce = open("./test_model/hit_ratio_ce.txt", f_mode, encoding="utf-8")
            f_hit_ratio_ce.close()
            f_best_action = open("./test_model/best_action.log", f_mode, encoding="utf-8")
            f_best_action.close()

        if os.path.exists('./test_model/bestnow.log'):
            os.remove('./test_model/bestnow.log')

        f_human_exp_hit = open("./test_model/human_exp_hit.txt", f_mode, encoding="utf-8")

        # 经验池预热时，需要写入actions原始值及对应reward到文件中
        # 随机森林使用到的原始数据(actions,reward)
        # 先写表头
        header = list(env.all_actions.keys())
        header.append('reward')
        with open('./test_model/actions_reward.csv', 'w', encoding='utf-8', newline='') as file_obj:
            # 1:创建writer对象
            writer = csv.writer(file_obj)
            # 2:写表头
            writer.writerow(header)

        # 为状态空间降维，特征提取
        new_components = PCA_REMAIN
        self.pca = pca = PCA(n_components=new_components)

        expr_name = 'train_{}_{}_{}'.format(env.info, env.method, str(utils.time_to_str(utils.get_timestamp())))
        self.TD3_logger = TD3_logger = utils.Logger(
            name=env.method,
            log_file='./test_model/log/{}.log'.format(expr_name)
        )
        
        rpm = None
        # 创建经验池
        if not USE_PRIORITY_RPM:
            self.rpm = rpm = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim)
        else:
            self.rpm = rpm = PrioritizedReplayMemory(MEMORY_SIZE, act_dim, obs_dim)


        src = globalValue.RPM_SRC
        dest = globalValue.RPM_DEST
        # 往经验池中预存数据
        while len(rpm) < MEMORY_WARMUP_SIZE:
            print("经验池中数据数量: ", len(rpm))
            
            if not USE_FIXED_RPM:
                total_reward, steps = self.run_episode(self.model, self.agent, env, rpm, pca, f_step_reward, TD3_logger, mode)
                f_time_store.write('warmup_time=' + str(env.end_time - env.start_time) + '\n')
                f_time_store.flush()
                f_human_exp_hit.write('human_exp_hit=' + str(env.human_exp_hitcnt * 1.0 / steps) + '\n')
                f_human_exp_hit.flush()
            else:
                if USE_PRIORITY_RPM:
                    with open(FIXED_RPM_PATH, 'rb') as f:
                        # 读取并反序列化数据
                        rpm_tmp = pickle.load(f)
                    f.close()
                    for i in range(len(rpm_tmp)):
                        s, a, r, s_p, done = rpm_tmp.buffer[i]
                        new_a = [a]
                        cal_td_s = self.agent.normalizer([s], rpm_tmp)
                        cal_td_s_p = self.agent.normalizer([s_p], rpm_tmp)

                        s_new = np.array([cal_td_s]).astype('float32')
                        a_new = np.array([new_a]).astype('float32').reshape(1, env.action_dim)
                        r_new = np.array([r]).astype('float32')
                        s_p_new = np.array([cal_td_s_p]).astype('float32')
                        done_new = np.array([done]).astype('float32')
                        trans_r_new = r_new.copy()
                        trans_r_new[trans_r_new>0] = trans_r_new[trans_r_new>0] / 1000000.0
                            
                            
                        td_error = self.agent.cal_td_error(s_new, a_new, trans_r_new, s_p_new, done_new)
                    
                    
                        rpm.append(td_error, rpm_tmp.buffer[i])
                else:
                    with open(FIXED_RPM_PATH, 'rb') as f:
                        # 读取并反序列化数据
                        rpm = pickle.load(f)
                    f.close()
                globalValue.RPM_SRC = src = FIXED_AR_PATH
            
            # total_reward, steps = self.run_episode(self.model, self.agent, env, rpm, pca, f_step_reward, TD3_logger, mode)
            # f_time_store.write('warmup_time=' + str(env.end_time - env.start_time) + '\n')
            # f_time_store.flush()
            # f_human_exp_hit.write('human_exp_hit=' + str(env.human_exp_hitcnt * 1.0 / steps) + '\n')
            # f_human_exp_hit.flush()
            if TWO_PHASE == True:
                utils.handle_csv(src, dest, env.action_dim)

                # 缓冲池预热完成后实现随机森林对动作空间的降维
                train_data = pd.read_csv(dest, encoding="utf-8")
                # 输出数据预览
                print(train_data.head())
                # 自变量
                x = train_data.iloc[:, :-1].values

                # 因变量（该数据集的最后1项:reward）
                y = train_data.loc[:, "reward"].values

                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.3, random_state=42)

                forest = RandomForestClassifier(n_estimators=100,
                                                criterion='entropy',
                                                random_state=42,
                                                n_jobs=-1,
                                                max_depth=None)
                forest.fit(x_train, y_train)
                y_train_pred = forest.predict(x_train)
                y_test_pred = forest.predict(x_test)

                test_score = forest.score(x_test, y_test)
                print('test score: %.3f' % (test_score) )
                cross_score = cross_val_score(forest, x_train, y_train, cv=10).mean()
                print('交叉验证得分:%.4f' % cross_score)
                importances = list(forest.feature_importances_)
                print(importances)
                
                

                # Saving feature names for later use
                feature_list = list(train_data.columns)[0:env.action_dim]

                feature_importances = [(feature, round(importance, 3)) for feature, importance in
                                    zip(feature_list, importances)]
                # Sort the feature importances by most important first
                feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

                # Print out the feature and importances
                print("random forest importances: \n{0}".format(feature_importances))
                
                
                
                
                
                
                explainer = shap.TreeExplainer(forest)
                #将想得到SHAP值的数据输入解释器中，得到每个样本的每个特征的SAHP值
                shap_values = explainer.shap_values(x_train)
                
                # Saving feature names for later use
                shap_feature_list = list(train_data.columns)[0:env.action_dim]

                
                # # Sort the feature importances by most important first
                # shap_feature_importances = sorted(shap_feature_importances, key=lambda x: x[1], reverse=True)

                # Print out the feature and importances
                # print("shap function importances: \n{0}".format(shap_values))
                print("shap function importances: \n{0}".format(np.array(abs(shap_values[0])).sum(0)))
                sum_shap_values = np.array(abs(shap_values[0])).sum(0)
                print("sum_shap_values =\n{0}".format(sum_shap_values))
                if sum_shap_values.sum()!= 0.0:
                    sum_shap_values = sum_shap_values / sum_shap_values.sum()
                
                    shap_feature_importances = [(feature, importance) for feature, importance in zip(shap_feature_list, sum_shap_values)]
                
                    print("shap function importances: \n{0}".format(shap_feature_importances))
                
                    shap_feature_importances = sorted(shap_feature_importances, key=lambda x: x[1], reverse=True)
                    print("shap function importances: \n{0}".format(shap_feature_importances))

                
                
                
                
                
                
                

                if cross_score == 1 or test_score == 1 or cross_score < 0.8 or test_score <= 0.8:
                    # 保存PCA和RF的结果
                    with open("./test_model/PCA&RF.txt", "a+", encoding="utf-8") as f_pr:
                        f_pr.write("++++RF========")
                        f_pr.write(" warmup steps: " + str(len(rpm)) + "\n")
                        f_pr.write("cross_score：" + str(cross_score) + " test_score: " + str(test_score) + "\n")
                        # f_pr.write(str(feature_importances) + "\n")
                    continue
                else:
                    break
                
                
        if USE_SHAP and sum_shap_values.sum() != 0.0:
            feature_importances = shap_feature_importances

        if TWO_PHASE == True:
            # 根据重要性选出尤其重要的动作参数来调整
            # 数据库节点端也需要进行相应调整，不再固定参数顺序而是能够根据参数名称识别调整
            # 读取feature_importances字典，删除除相对不重要的参数，但不删除bp_size
            action_keys = list(env.all_actions.keys())
            cnt = 0
            del_index = []
            del_key = []
            val_now = 0
            # pick importance of 0.8
            for info in feature_importances:
                cnt += 1
                val_thr = 0.75
                if USE_KNOBS_DR:
                    pass
                else:
                    val_thr = 1.5
                if val_now >= val_thr:
                    key = info[0]
                    # 如果是bpsize则跳过不删除
                    actual_key = key.split("#")[1]
                    if actual_key == 'buffer_pool_size':
                        continue
                    del_index.append(action_keys.index(key))
                    del_key.append(key)
                    del env.all_actions[key]
                    env.action_dim -= 1
                val_now += float(info[1])
            # TODO: 每个node的tune_action也需要相应地删除
            for k in del_key:
                uuid = int(k.split('#')[0])
                key = k.split('#')[1]
                if uuid < env.se_num:
                    # print('del se{} {}'.format(uuid, key))
                    del env.se_info[uuid].tune_action[key]
                else:
                    # print('del ce{} {}'.format(uuid, key))
                    del env.ce_info[uuid - env.se_num].tune_action[key]

            print('降维后动作参数们：', env.all_actions)
            print('降维后动作参数维度：', env.action_dim)
            # print('del_index:', del_index)
            # 降维后需要重新构建神经网络
            act_dim = env.action_dim
            # 保存PCA和RF的结果
            with open("./test_model/PCA&RF.txt", "a+", encoding="utf-8") as f_pr:
                f_pr.write("++++RF========" + "\n")
                f_pr.write("warmup steps:" + str(len(rpm)) + "\n")
                f_pr.write("cross_score：" + str(cross_score) + " test_score: " + str(test_score) + "\n")
                f_pr.write(str(feature_importances) + "\n")
                f_pr.write('降维后动作参数维度：' + str(env.action_dim) + "\n")

        print('===>经验池预热完成!')
        print("===>经验池中数据数量: ", len(rpm))

        ckpt = './test_model/test_save/first_phase_{}'.format(str(utils.time_to_str(utils.get_timestamp())))
        f = open(ckpt + "_nodes_rpm.txt", "wb")
        # 保存回放内存,写盘很费时间，注意控制写盘频率
        pickle.dump(rpm, f)
        f.close()
        print('save rpm ok')
        # 预热完成修改env状态
        env.state = 1
        # reset the variables in WARMUP
        flag = True
        while flag:
            obs, flag = env.reset()

        if TWO_PHASE == True:
            # 为状态空间降维，特征提取
            obs_all = rpm.getStates()
            # normalize
            mean, std = rpm.countRes(rpm.getStates())
            t = 0
            for i in mean:
                if i == 0:
                    np.delete(obs_all, t, axis=1)
                t += 1
            res_obs = []
            for obs in obs_all:
                obs = np.array(obs)
                obs = (obs - mean) / std
                # 2:写表
                res_obs.append(obs)
            obs_all = np.array(res_obs)
            # pca = PCA(n_components=new_components)
            pca.fit(obs_all)
            # newdata = pca.fit_transform(obs_all)
            print('原数据的特征值or解释的方差：{}'.format(pca.explained_variance_))
            print('主成分的方差贡献率：{}'.format(pca.explained_variance_ratio_))
            print('主成分的累积方差贡献率：{}'.format(pca.explained_variance_ratio_.cumsum()))
            print('原数据的特征向量：')
            print(pca.components_.T)
            k1_spss = pca.components_.T
            weight = (np.dot(k1_spss, pca.explained_variance_ratio_)) / np.sum(pca.explained_variance_ratio_)
            weight_weight = weight / np.sum(weight)
            obs_dim = new_components
            # 新数据加入时，pca.transform(newdata)的方法实现降维

            skplt.decomposition.plot_pca_2d_projection(pca, obs_all, y=None)
            save_name = './test_model/pca_fig.png'
            plt.savefig(save_name)
            plt.show()

            # 保存PCA和RF的结果
            with open("./test_model/PCA&RF.txt", "a+", encoding="utf-8") as f_pr:
                f_pr.write("++++PCA========" + "\n")
                f_pr.write("主成分的方差贡献率：" + str(pca.explained_variance_ratio_) + "\n")
                f_pr.write("主成分的累积方差贡献率：" + str(pca.explained_variance_ratio_.cumsum()) + "\n")
                f_pr.write("原数据的特征值or解释的方差：" + str(pca.explained_variance_) + "\n")
                f_pr.write("主成分：" + str(obs_dim) + "\n")
                f_pr.write("weight：" + str(weight) + "\n")
                f_pr.write("weight_weight：" + str(weight_weight) + "\n")


            # TODO：为实现后续的归一化，经验池内数据reshape,将actions和states维度降低
            # 1 根据action_info的名字找出具体的index
            # 2 只留下这些index或者使用这些index重新构建actions数组
            # 需要从头开始pop整个buffer，对每一条经验进行action的reshape后push至尾部
            # 也可以直接创建一个新经验池代替：

            # DDPG
            if env.method == 'DDPG':
                self.model =  Model(act_dim)
                algorithm = DDPG(
                    self.model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
                self.agent =  Agent(algorithm, obs_dim, act_dim)

            # TD3
            if env.method == 'TD3':
                self.model =  TestModel(act_dim, max_action)
                algorithm = TD3(
                    self.model,
                    max_action=max_action,
                    gamma=GAMMA,
                    tau=TAU,
                    actor_lr=ACTOR_LR,
                    critic_lr=CRITIC_LR,
                    policy_noise=POLICY_NOISE,  # Noise added to target policy during critic update
                    noise_clip=NOISE_CLIP,  # Range to clip target policy noise
                    policy_freq=POLICY_FREQ
                )
                if USE_STATUS_DR:
                    self.agent =  TestAgent(algorithm, obs_dim, act_dim)
                else:
                    self.agent =  TestAgent(algorithm, self.env.state_dim, act_dim)           


            # SAC
            if env.method == 'SAC':
                self.model =  SACModel(act_dim)
                algorithm = SAC(
                    actor=self.model.actor_model,
                    critic=self.model.critic_model,
                    max_action=max_action,
                    alpha=H_SAC_ALPHA,
                    gamma=H_SAC_GAMMA,
                    tau=H_SAC_TAU,
                    actor_lr=H_SAC_ACTOR_LR,
                    critic_lr=H_SAC_CRITIC_LR,
                )
                if USE_STATUS_DR:
                    self.agent =  SACAgent(algorithm, obs_dim, act_dim)
                else:
                    self.agent =  SACAgent(algorithm, self.env.state_dim, act_dim) 
                

            
 
            rpm_new = None
            if not USE_PRIORITY_RPM:
                rpm_new = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim)
            else:
                rpm_new = PrioritizedReplayMemory(MEMORY_SIZE, act_dim, obs_dim)


            # env.method = 'TD3'
            rpm_len = len(rpm)
            for i in range(0, rpm_len):
                s, a, r, s_p, done = rpm.pop()
                # print('states:', s)
                # 将actions和state reshape
                new_a = np.delete(a, del_index)
                new_a = [new_a]
                if USE_STATUS_DR:
                    s = pca.transform(s.reshape(1, -1))
                    s = np.array(s).flatten()
                    s_p = pca.transform(s_p.reshape(1, -1))
                    s_p = np.array(s_p).flatten()
                
                if USE_PRIORITY_RPM:

                    cal_td_s = self.agent.normalizer([s], rpm_new)
                    cal_td_s_p = self.agent.normalizer([s_p], rpm_new)

                    s_new = np.array([cal_td_s]).astype('float32')
                    a_new = np.array([new_a]).astype('float32').reshape(1, env.action_dim)
                    r_new = np.array([r]).astype('float32')
                    s_p_new = np.array([cal_td_s_p]).astype('float32')
                    done_new = np.array([done]).astype('float32')
                    trans_r_new = r_new.copy()
                    trans_r_new[trans_r_new>0] = trans_r_new[trans_r_new>0] / 1000000.0
                    
                    
                    td_error = self.agent.cal_td_error(s_new, a_new, trans_r_new, s_p_new, done_new)
                    rpm_new.append(td_error, (s, new_a, r, s_p, done))
                    
                else:
                    
                # td_error = self.agent.cal_td_error(s_new, a_new, r_new, s_p_new, done_new)

                #td_error = self.agent.cal_td_error(cal_td_s, new_a, r, cal_td_s_p, done)
                
                    rpm_new.append((s, new_a, r, s_p, done))
            print('rpm_new len = ', len(rpm_new))
            
            if USE_STATUS_DR == False:
                rpm_new = rpm
            # 至此，将神经网络的状态维度和动作维度降维完成，经验池内数据reshape完成

            # TODO:用reshape后的经验池数据对新网络进行快速预训练
            for i in range(0, len(rpm_new)):
                #(batch_obs, batch_action, batch_reward, batch_next_obs,
                #batch_done) = rpm_new.sample(BATCH_SIZE)
                idxs = None
                batch_obs = None
                batch_action = None
                batch_reward = None
                batch_next_obs = None
                batch_done = None
                if USE_PRIORITY_RPM:
                    (idxs, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm_new.sample(BATCH_SIZE)
                else:
                    (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm_new.sample(BATCH_SIZE)


                # 这里维度已经缩小了，所以不再需要pca降维，降维应该发生在收集到原始数据库状态后
                # print('batch_obs:', batch_obs)
                batch_obs = self.agent.normalizerBatch(batch_obs, rpm_new)
                batch_next_obs = self.agent.normalizerBatch(batch_next_obs, rpm_new)
                # DDPG
                if env.method == 'DDPG':
                    critic_cost = self.agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                # TD3
                if env.method == 'TD3' or env.method == 'SAC':
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("agent learn, alg = {0}, time = train".format(env.method))
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                    actor_cost, critic_cost = self.agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                                                        batch_done)
                    if USE_PRIORITY_RPM:
                        trans_batch_reward = batch_reward.copy()
                        trans_batch_reward[trans_batch_reward>0] = trans_batch_reward[trans_batch_reward>0] / 1000000.0
                        new_td_error = self.agent.cal_td_error(batch_obs, batch_action, trans_batch_reward, batch_next_obs, batch_done)
                        #new_td_error = self.agent.cal_td_error(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                        print("*************************************")
                        print("*new_td_error = {0}, critic_cost = {1}".format(new_td_error, critic_cost))
                        print("*************************************")
                        for i in range(BATCH_SIZE):
                            idx = idxs[i]
                            rpm_new.update(idx, new_td_error[i])

            with open("./test_model/PCA&RF.txt", "a+", encoding="utf-8") as f_pr:
                f_pr.write("Fast learn finished!" + "\n")

        f_train_reward = open("./test_model/train_reward_cal.txt", f_mode, encoding="utf-8")
        f_eval_reward = open("./test_model/eval_reward_cal.txt", f_mode, encoding="utf-8")

        episode = 0
        if env.expert_exp == True:
            env.action_trend_choice = ACTION_TREND_P
        env.best_action_choice = BEST_NOW_P
        # if os.path.exists('./test_model/bestnow.log'):
        #     os.remove('./test_model/bestnow.log')
        while episode < TRAIN_EPISODE:
            # 每训练5个episode，做一次评估
            print('Start a new round,this round include %d episode and 1 evaluate process!' % EVAL_INTERVAL)
            for i in range(EVAL_INTERVAL):
                print('=========>>> episode = ', episode)
                if TWO_PHASE == True:
                    total_reward, steps = self.run_episode(self.model, self.agent, env, rpm_new, pca, f_step_reward, TD3_logger)
                else:
                    total_reward, steps = self.run_episode(self.model, self.agent, env, rpm, pca, f_step_reward, TD3_logger)
                env.action_trend_choice = ACTION_TREND_P - (episode+1) * 0.001
                if env.action_trend_choice < 0.1:
                    env.action_trend_choice = 0.1
                f_train_reward.write("train_reward="+str(total_reward)+"\n")
                f_train_reward.flush()
                f_time_store.write('episode_time='+str(env.end_time - env.start_time)+"\n")
                f_time_store.flush()
                f_human_exp_hit.write('human_exp_hit='+str(env.human_exp_hitcnt * 1.0 / steps)+'\n')
                f_human_exp_hit.flush()
                print('Episode:{}    Test reward:{}'.format(episode, total_reward))
                episode += 1
                if flag == 0:
                    f = open("./1se/rpm_dir/se_rpm_new.txt", "wb")
                elif flag == 1:
                    f = open("./test_model/test_save/ce_rpm_new.txt", "wb")
                else:
                    f = open("./test_model/test_save/nodes_rpm_new.txt", "wb")
                    # 保存回放内存,写盘很费时间，注意控制写盘频率
                print('-----------save_rpm-----------')
                if TWO_PHASE == True:
                    pickle.dump(rpm_new, f)
                else:
                    pickle.dump(rpm, f)
                f.close()

            print('-------------start_eval_test-------------')
            if TWO_PHASE == True:
                eval_reward = self.evaluate_ce(env, self.agent, rpm_new, pca, TD3_logger)
            else:
                eval_reward = self.evaluate_ce(env, self.agent, rpm, pca, TD3_logger)
            f_eval_reward.write("eval_reward="+str(eval_reward)+"\n")
            f_eval_reward.flush()
            f_time_store.write('eval_time=' + str(env.end_time - env.start_time) + "\n")
            f_time_store.flush()
            print('Eval:{}    Test reward:{}'.format(env.eval, eval_reward))

        TD3_logger.end_time = TD3_logger.get_timestr()
        TD3_logger.info("ALL the episode done, start time = %s, end time = %s" % (TD3_logger.start_time, TD3_logger.end_time))

        # 关闭文件
        f_step_reward.close()
        f_train_reward.close()
        f_eval_reward.close()
        f_time_store.close()
        f_human_exp_hit.close()
        # 保存rpm
        # if flag == 0:
        #     ckpt = './1se/model_dir/se_steps_{}'.format(int(time.time()))
        #     f = open("./1se/rpm_dir/se_rpm_new.txt", "wb")
        # elif flag == 1:
        #     ckpt = './test_model/test_save/test_save_final_ce_{}'.format(int(time.time()))
        #     f = open("./test_model/test_save/ce_rpm_new.txt", "wb")
        # else:
        ckpt = './test_model/test_save/test_save_final_nodes_{}'.format(str(utils.time_to_str(utils.get_timestamp())))
        f = open(ckpt + "_nodes_rpm_new.txt", "wb")
        # 保存回放内存,写盘很费时间，注意控制写盘频率
        print('-----------save_rpm_new-----------')
        if TWO_PHASE == True:
            pickle.dump(rpm_new, f)
        else:
            pickle.dump(rpm, f)
        f.close()

        # 保存模型
        print('-----------save_model-----------')
        print('ckpt = ', ckpt)
        # DDPG
        if env.method == 'DDPG':
            self.agent.save(save_path=ckpt + '.cpkt')
        # TD3
        if env.method == 'TD3':
            self.agent.save(save_path=ckpt + '_predict.ckpt', mode='predict')
            self.agent.save(save_path=ckpt + '_train_actor.ckpt', mode='train_actor')
            self.agent.save(save_path=ckpt + '_train_critic.ckpt', mode='train_critic')

        # SAC
        if env.method == 'SAC':
            self.agent.save(save_path=ckpt + '_predict_sac.ckpt', mode='predict')
            self.agent.save(save_path=ckpt + '_train_learn_sac.ckpt', mode='learn')








    '''
        SHAP 方法验证
            [func] cal_kernel_shap: 使用 kernel shap 计算 shapely value
            [func] kernel_predict_reward: 使用 kernel shap 需要使用到 kernel_predict_reward 与模型交互进行采样
            [func] reshape_action: kernel_predict_reward 中使用本函数对 action 重新 reshape
            [func] cal_tree_shap: 使用 kernel shap 计算 shapely value
    
    '''
    def cal_kernel_shap(self):
        
        self.env = env = NodesEnv()
        env.init_nodes()
        if env.method == 'DQN':
            env.init_env_for_DQN()

        env.method = self.model_name
        env.expert_exp = True
        obs_dim = env.state_dim
        act_dim = env.action_dim
        max_action = 0.99

        if env.expert_exp == True:
            env.action_trend_choice = ACTION_TREND_P
        else:
            env.action_trend_choice = 0
        env.best_action_choice = 0

        # TD3
        if env.method == 'TD3':
            self.model =  TestModel(act_dim, max_action)
            algorithm = TD3(
                self.model,
                max_action=max_action,
                gamma=GAMMA,
                tau=TAU,
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
                policy_noise=POLICY_NOISE,  # Noise added to target policy during critic update
                noise_clip=NOISE_CLIP,  # Range to clip target policy noise
                policy_freq=POLICY_FREQ
            )
            self.agent =  TestAgent(algorithm, obs_dim, act_dim)
        
        
        # src = globalValue.RPM_SRC
        dest = globalValue.QUICK_SHAP_DATA
        
        # utils.handle_csv(src, dest, self.env.action_dim)

        # 缓冲池预热完成后实现随机森林对动作空间的降维
        train_data = pd.read_csv(dest, encoding="utf-8")
        # 输出数据预览
        print(train_data.head())
        # 自变量
        x = train_data.iloc[:, :-1].values

        # 因变量（该数据集的最后1项:reward）
        y = train_data.loc[:, "reward"].values

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42)
        
        print("[CAL SHAP] x_train action = \n{0}".format(x_train))
        
        
        # SHAP importances
        #选择核函数解释，使用聚类后的数据训练模型
        explainerKNN = shap.KernelExplainer(self.kernel_predict_reward, data=x_train)
        #将想得到SHAP值的数据输入解释器中，得到每个样本的每个特征的SAHP值
        shap_values = explainerKNN.shap_values(x_train)
        
        # Saving feature names for later use
        shap_feature_list = list(train_data.columns)[0:self.env.action_dim]

        shap_feature_importances = [(feature, round(importance, 3)) for feature, importance in
                            zip(shap_feature_list, shap_values)]
        # Sort the feature importances by most important first
        shap_feature_importances = sorted(shap_feature_importances, key=lambda x: x[1], reverse=True)

        # Print out the feature and importances
        print("shap function importances: \n{0}".format(shap_feature_importances))


    def kernel_predict_reward(self, action):
        
        res = []
        
        print("****action.shape = {0}****\n".format(action.shape))
        for i in range(action.shape[0]):
            new_action = self.reshape_action(action[i])
            next_obs, reward, done, info = self.env.step(new_action, False)
            res.append(reward)
        res = np.array(res)
        print("****shap reward = {0}****\n".format(res))
        print("****shap reward shape = {0}****".format(res.shape))
        
        

        return res


    def cal_tree_shap(self):
        
        
        self.env = env = NodesEnv()
        env.init_nodes()
        if env.method == 'DQN':
            env.init_env_for_DQN()

        env.method = self.model_name
        env.expert_exp = True
        obs_dim = env.state_dim
        act_dim = env.action_dim
        max_action = 0.99

        if env.expert_exp == True:
            env.action_trend_choice = ACTION_TREND_P
        else:
            env.action_trend_choice = 0
        env.best_action_choice = 0

        # TD3
        if env.method == 'TD3':
            self.model =  TestModel(act_dim, max_action)
            algorithm = TD3(
                self.model,
                max_action=max_action,
                gamma=GAMMA,
                tau=TAU,
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
                policy_noise=POLICY_NOISE,  # Noise added to target policy during critic update
                noise_clip=NOISE_CLIP,  # Range to clip target policy noise
                policy_freq=POLICY_FREQ
            )
            self.agent =  TestAgent(algorithm, obs_dim, act_dim)
        
        
        
        
        src = globalValue.QUICK_SHAP_DATA
        dest = globalValue.QUICK_SHAP_DATA_DEST
        
        tmp_action_dim = 0
        for se in self.env.se_info:
            tmp_action_dim += len(se.tune_action.items())
        
        for ce in self.env.ce_info:
            tmp_action_dim += len(ce.tune_action.items())
        
        
        
        utils.handle_csv(src, dest, tmp_action_dim)

        # 缓冲池预热完成后实现随机森林对动作空间的降维
        train_data = pd.read_csv(dest, encoding="utf-8")
        # 输出数据预览
        print(train_data.head())
        # 自变量
        x = train_data.iloc[:, :-1].values

        # 因变量（该数据集的最后1项:reward）
        y = train_data.loc[:, "reward"].values

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42)
        
        print("[CAL SHAP] x_train action = \n{0}".format(x_train))
        
        forest = RandomForestClassifier(n_estimators=100,
                                                criterion='entropy',
                                                random_state=42,
                                                n_jobs=-1,
                                                max_depth=None)
        forest.fit(x_train, y_train)
        
        
        test_score = forest.score(x_test, y_test)
        print('test score: %.3f' % (test_score) )
        cross_score = cross_val_score(forest, x_train, y_train, cv=10).mean()
        print('交叉验证得分:%.4f' % cross_score)
        importances = list(forest.feature_importances_)
        print(importances)
        
        

        # Saving feature names for later use
        feature_list = list(train_data.columns)[0:self.env.action_dim]

        feature_importances = [(feature, round(importance, 3)) for feature, importance in
                            zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        # Print out the feature and importances
        print("random forest importances: \n{0}".format(feature_importances))
        
        
        
        explainer = shap.TreeExplainer(forest)
        #将想得到SHAP值的数据输入解释器中，得到每个样本的每个特征的SAHP值
        shap_values = explainer.shap_values(x_train)
        
        
        # Saving feature names for later use
        shap_feature_list = list(train_data.columns)[0:self.env.action_dim]

        
        # # Sort the feature importances by most important first
        # shap_feature_importances = sorted(shap_feature_importances, key=lambda x: x[1], reverse=True)

        # Print out the feature and importances
        # print("shap function importances: \n{0}".format(shap_values))
        print("shap function importances: \n{0}".format(np.array(abs(shap_values[0])).sum(0)))
        sum_shap_values = np.array(abs(shap_values[0])).sum(0)
        sum_shap_values = sum_shap_values / sum_shap_values.sum()
        
        shap_feature_importances = [(feature, importance) for feature, importance in zip(shap_feature_list, sum_shap_values)]
        
        print("shap function importances: \n{0}".format(shap_feature_importances))
        
        shap_feature_importances = sorted(shap_feature_importances, key=lambda x: x[1], reverse=True)
        print("shap function importances: \n{0}".format(shap_feature_importances))
 
    
    def tree_predict_reward(self, action):
        pass



    def reshape_action(self, action):
        new_action = []
        cnt = 0
        
        print("****resahpe action = {0}****\n".format(action))
        
        for se in self.env.se_info:
            node_action = []
            for key, val in se.tune_action.items():
                node_action.append(action[cnt])
                cnt += 1
            new_action.append(node_action)
        
        for ce in self.env.ce_info:
            node_action = []
            for key, val in ce.tune_action.items():
                node_action.append(action[cnt])
                cnt += 1
            new_action.append(node_action)
        
        return new_action





        










if __name__ == '__main__':
    pass
