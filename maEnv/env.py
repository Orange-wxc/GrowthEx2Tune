#环境
import csv

from maEnv import utils
import globalValue
import time
import numpy as np
from maEnv import datautils
import random
import os
import threading


# 注意以下两个时间加起来要<load_bash的时间,load_bash = 80s
# SLEEP_TIME = 60     # 动作参数应用后等待数据库状态稳定的时间
SLEEP_TIME = 10   # 动作参数应用后等待数据库状态稳定的时间
# BUFFER_TIME = 35    # 收集完数据库状态后等待load_bash结束的时间
BUFFER_TIME = 5    # 收集完数据库状态后等待load_bash结束的时间
BASH_TYPE = 'sysbench'

class SEEnv():
    def __init__(self):
        # 状态变量的维度
        self.state_dim = 18
        # 动作变量的维度
        self.action_dim = 16
        # 动作变量基本情况，说明最大值、最小值、步长等情况
        # 默认值，最小值，最大值
        self.action_info = {"buffer_pool_size": [33554432, 33554432, 3355443200],
                            "old_blocks_pct": [37, 5, 95],
                            "old_threshold_ms": [1000, 0, 10000],
                            "flush_neighbors": [1, 0, 1],
                            "lru_sleep_time_flush": [1000, 50, 2000],
                            "flush_n": [0, 0, 3200],
                            "SE_LRU_scan_depth": [20, 0, 100],
                            "lru_sleep_time_remove": [0, 0, 1000],
                            "lru_scan_depth": [8192, 1024, 10240],
                            "reserve_free_page_pct_for_se": [50, 5, 95],
                            "free_page_threshold": [8192, 1024, 10240],# down here are new parameters
                            "innodb_max_dirty_pages_pct": [75, 0, 99.99],
                            "innodb_adaptive_flushing_lwm": [10, 0, 70],
                            "innodb_io_capacity_max": [2000, 100, pow(2, 64) - 1],
                            "lock_wait_timeout": [31536000, 1, 31536000],
                            "innodb_flushing_avg_loops": [30, 1, 1000]
                            }
        self.max_info = [3355443200,95,pow(2,32)-1,1,2000,3200,100,1000,10240,95,10240,64,99.99,70,pow(2,64)-1]
        self.min_info = [33554432, 5, 0,    0,50,  0,   0,  0,    1024,5, 1024,0,0, 0, 0, 100]
        # step用，计算无用step
        self.unuse_step = 0

        # self.ip = '192.168.2.40'
        self.ip = globalValue.CONNECT_MA_IP
        self.port = 40002
        self.last_bp_size = 268435456

        self.state = 0


    #action为一组动作变量的值
    def step(self, action):
        #将action转为可应用的值
        real_action = utils.action_change(self,action)
        # 判断action是否合法
        if not utils.action_cirtic(self,real_action):
            return np.array(globalValue.GLOBAL_CURRENT_STATUS), -1000000, False, {}
        # 收集应用参数之前的性能,10s
        h_before = utils.get_hr()
        print('the hit_ratio before apply action is: ', h_before)
        # 将推荐参数封装为发送数据标准格式
        # True为SE
        # 3代表参数更改请求
        var_names = list(self.action_info.keys())
        send_variables = utils.get_set_variables_string(var_names, real_action, True, 3)
        # 发送参数给数据库
        received_msg = utils.send_msg_to_server(send_variables, self.ip, self.port)
        print('The recommand knobs setting is : ', send_variables, ' and  ', received_msg)
        # 默认参数应用成功
        # 休息10秒给参数应用缓冲
        sleep_time = (abs(real_action[0] - self.last_bp_size) / 1024 / 1024 / 10) * 2
        print('sleep_time = ', sleep_time)
        time.sleep(sleep_time)
        # 收集应用参数之后的性能信息,10s
        # h_after = utils.get_hr(self.ip, self.port)
        h_after = utils.get_hr()
        print('the hit_ratio after apply action is: ', h_after)
        # 获取缓冲池大小
        bps = utils.get_bps(self.ip, self.port)
        self.last_bp_size = bps
        # 用h_before和h_after计算reward
        reward = utils.cal_reward_se(h_before, h_after, bps)
        # 获取当前状态,注意对于send_msg_to_server接收回的数据要做相应处理
        current_state = utils.send_msg_to_server('1', self.ip, self.port)
        current_state = datautils.GetState(current_state)
        print("The current_state is : ", current_state)


        # if h_after - h_before < 0.1 and h_after - h_before > -0.1:
        #     self.unuse_step += 1
        next_state = current_state

        done = globalValue.SE_FLAG

        # !!!!!!!考虑好什么是最终情况!!!!!!!!!!!
        # done = False
        # 目前先认为4次性能改变不大则进入最终情况
        # if self.unuse_step > 4:
        #     done = True
        return np.array(next_state), reward, done, {}

    def reset(self):
        # 重开一轮训练
        # 唤醒压测线程，进行压测
        print("se===reset===reset")
        globalValue.LOAD_EVENT.set()
        time.sleep(20)
        # 给一定时间稳定后，获取当前状态，作为初始状态
        current_state = utils.send_msg_to_server('1',self.ip,self.port)
        current_state = datautils.GetState(current_state)
        print("After reset, The initial_state is : ", current_state)
        return current_state


class CEEnv():
    def __init__(self):
        self.info = 'CE'
        # 状态变量的维度
        self.state_dim = 4  # TODO：增加当前缓冲池大小作为状态之一
        # 动作变量的维度
        self.action_dim = 7
        # 动作变量基本情况，说明最大值、最小值、步长等情况
        # 默认值，最小值，最大值
        # action_name --- action_index
        self.action_info = {
                       'buffer_pool_size': 0,
                       'old_blocks_pct': 1,
                       'old_threshold_ms': 2,
                       'ce_coordinator_sleep_time': 3,
                       'ce_free_page_threshold': 4,
        # add new parameters
        #                'innodb_random_read_ahead': [0, 0, 1],
        #                'innodb_read_ahead_threshold':[56,0,64],
        #                'lock_wait_timeout':[31536000, 1, 31536000],
                       'innodb_flushing_avg_loops': 5,
                       'innodb_adaptive_max_sleep_delay': 6
        }
        #
        self.max_info = [3355443200,95,10000,2000,10240,1000,100000]
        # self.min_info = [33554432, 5, 0,    50,  1024,0]
        # self.max_info = [95, pow(2, 32) - 1, 2000, 10240, 1000, 100000]
        self.min_info = [5242880, 5, 0, 50, 1024, 1, 0]
        # self.max_info = [95, 10000]
        # self.min_info = [5, 0]
        # self.max_info = [3355443200, 95, 10000, 1000]
        # self.min_info = [5242880, 5, 0, 1]
        self.default = [134217728, 37, 1000, 1024, 1024, 30, 1000]
        # step用，计算无用step
        self.unuse_step = 0
        self.ip = globalValue.CONNECT_MA_IP
        self.port = 2000

        # last---上一个step
        self.last_bp_size = 134217728
        self.last_hr = -1
        self.max_hit_ratio = -1
        self.last_action = -2

        # t0---reset时的表现
        self.hit_t0 = -1
        self.qps_t0 = 0
        self.bp_size_0 = 134217728
        # before---至今最优表现
        self.hit_before = 0
        self.qps_before = 0
        self.bpsize_before = 0

        self.labelA = -1
        self.labelB = -1
        self.action_trend_choice = 0

        # 0-经验池预热中，1-经验池预热完毕, 2-eval
        self.state = 0
        self.score = 0
        self.steps = 0

        self.episode = 0
        self.eval = 0

        self.method = 'TD3'

    # action为一组动作变量的值
    def step(self, action):
        # print('-----------------------one step----------------------')
        # 将action转为可应用的值
        real_action = utils.action_change(self, action)
        # print('real_action:', real_action)
        # 1.重启mysql----每一个step重启，避免缓冲池内的热点数据页比例过多
        buffer_set = '--innodb_buffer_pool_size=' + str(real_action[0])
        # 防止启动mysqld失败
        while True:
            utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.MYSQLD_CLOSE_EXEC)
            time.sleep(1)
            utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.NEW_DATA_CMD)
            time.sleep(1)
            # start se
            utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.MYSQLD_OPEN_EXEC_SE + globalValue.MYSQLD_OUTPUT_SE)
            time.sleep(2)
            # then start ce
            utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.MYSQLD_OPEN_EXEC_CE + buffer_set + globalValue.MYSQLD_OUTPUT_CE)

            # result = utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
            #                       globalValue.MYSQLD_CHRCK)
            # if result is None:
            #     continue

            delta_bps = real_action[0] - self.last_bp_size
            if delta_bps > 0:
                sleep_time = delta_bps / 1024 / 1024 / 256
            else:
                sleep_time = abs(delta_bps) / 1024 / 1024 / 128

            print('--->1.启动数据库：', sleep_time)
            time.sleep(sleep_time)
            time.sleep(5)
            # 将推荐参数封装为发送数据标准格式
            var_names = list(self.action_info.keys())
            send_variables = utils.get_set_variables_string(var_names, real_action, False, 3)
            # 2.动作变量默认值设置，此时需要确认是否成功连接mysql
            received_msg, flag = utils.send_msg_to_server(send_variables, self.ip, self.port)
            if flag:
                break

        print('The recommand knobs setting is : ', send_variables, ' and  ', received_msg)
        print('--->2.等待5s,数据库应用动作参数')
        time.sleep(5)

        sleep_time = SLEEP_TIME
        buffer_time = BUFFER_TIME
        utils.load_bash_remote(BASH_TYPE)
        # result_f, buffer_time = utils.load_bash(sleep_time)
        print('--->3.等待状态稳定时间：', sleep_time)
        time.sleep(sleep_time)
        # 收集应用参数之后的性能信息,10s
        q_after = utils.get_qps()
        h_after = utils.get_hr()

        print('the hr_after apply action is: ', h_after)
        print('the qps_after apply action is: ', q_after)

        # 获取缓冲池大小
        bps = utils.get_bps(self.ip, self.port)
        # bps = utils.show_sql('innodb_buffer_pool_size')[0][1]
        # bps = int(bps)
        globalValue.GLOBAL_BUFFER_POOL_SIZE = bps

        # 计算reward
        # reward = utils.cal_reward_ce_1(self, self.hit_before, h_after, self.qps_before, q_after)
        reward = utils.cal_reward_ce_my(self, self.hit_before, h_after, self.qps_before, q_after, action[0],
                                        self.min_info, self.max_info)
        globalValue.REWARD_NOW = reward
        globalValue.MAX_REWARD = max(reward, globalValue.MAX_REWARD)
        print('@@@one step action @@@reward is ', reward)
        if self.state != 2:
            self.score += reward
            self.steps += 1
        # 根据当前reward大小判断是否是best
        flag = utils.record_best(q_after, h_after, bps)
        if flag == True:
            print('Better performance changed!')
            with open("./test_model/best_action.log", "w+", encoding="utf-8") as f_best_action:
                f_best_action.write("recommand actions of bestnow"+send_variables+"\n")
        else:
            print('Performance remained!')
            # get the best performance so far to calculate the reward

        best_now_performance = utils.get_best_now('bestnow.log')
        self.qps_before = best_now_performance[0]
        self.hit_before = best_now_performance[1]
        self.bpsize_before = best_now_performance[2]

        with open("./test_model/qps_store.txt", "a+", encoding="utf-8") as f_qps:
            f_qps.write("qps="+str(q_after)+"\n")

        with open("./test_model/buffer_pool_size.txt", "a+", encoding="utf-8") as f_bp:
            f_bp.write("buffer_pool_size="+str(bps)+"\n")

        with open("./test_model/hit_ratio.txt", "a+", encoding="utf-8") as f_hr:
            f_hr.write("hit_ratio=" + str(h_after) + "\n")

        if self.state == 0:
            with open('./test_model/actions_reward.csv', 'a', encoding='utf-8', newline='') as file_obj:
                b = [item for sublist in real_action for item in sublist]
                b.append(reward)
                # 1:创建writer对象
                writer = csv.writer(file_obj)
                # 2:写表
                writer.writerow(b)

        # 获取当前状态,注意对于send_msg_to_server接收回的数据要做相应处理
        current_state, flag = utils.send_msg_to_server('1',self.ip,self.port)
        current_state = datautils.GetState(current_state)
        print("The current_state is : ", current_state)
        self.labelA, self.labelB = datautils.status_to_labels(current_state)

        if globalValue.EVAL_TEST:
            f = open("./1ce/recommand_knobs/knobs.txt", 'a')
            # f.write('=======================The recommand knobs==========================')
            # f.write("\n")
            f.write(send_variables)
            f.write('        reward = ')
            f.write(str(reward))
            f.write("\n")
            f.close()

        next_state = current_state
        # 当前qps和当前bp size为下一阶段的上一个qps和上一个bp size
        self.last_qps = q_after
        self.last_hr = h_after
        self.last_bp_size = bps
        self.last_action = action[0]
        # !!!!!!!考虑好什么是最终情况!!!!!!!!!!!
        # 不需要最终情况？
        done = False
        globalValue.SE_FLAG = done
        print('--->4.等待load bash结束时间：', buffer_time)
        time.sleep(buffer_time)
        utils.send_msg_to_server("exit", self.ip, self.port)
        utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                     globalValue.MYSQLD_CLOSE_EXEC)
        # current_time = time.time()

        # 依据此轮训练的时间来判断是否done，若时间不够再进行一次取样，则done为True
        # if time_remain < 85:
        #     done = True
        #     globalValue.SE_FLAG = done
        #     # 睡20秒给压测清理环境
        #     time.sleep(time_remain + 20)
        #  print('Episode time_remain: ', time_remain)
        #  连续4次qps变化不大则判断结束
        # if abs(delta_qps) < 50:
        #     self.unuse_step += 1
        # else:
        #     self.unuse_step = 0
        #
        # if self.unuse_step == 4:
        #     self.unuse_step = 0

        if self.score < -500:
            done = True

        # TODO:持久化action信息

        return np.array(next_state), reward, done, {}

    def reset(self):
        # 一个episode的开始，重置环境
        print("==========>ce===reset")
        # 1.重启mysql
        buffer_set = '--innodb_buffer_pool_size=' + str(self.default[0])
        # 防止启动mysqld失败
        while True:
            utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.MYSQLD_CLOSE_EXEC)
            time.sleep(1)
            utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.NEW_DATA_CMD)
            time.sleep(1)
            # start se
            utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.MYSQLD_OPEN_EXEC_SE + globalValue.MYSQLD_OUTPUT_SE)
            time.sleep(2)
            # then start ce
            utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.MYSQLD_OPEN_EXEC_CE + buffer_set + globalValue.MYSQLD_OUTPUT_CE)
            # 检查mysqld进程是否启动
            # result = utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
            #                       globalValue.MYSQLD_CHRCK)
            # if result is None:
            #     continue

            delta_bps = self.default[0] - self.last_bp_size
            if delta_bps > 0:
                sleep_time = delta_bps / 1024 / 1024 / 256
            else:
                sleep_time = abs(delta_bps) / 1024 / 1024 / 128
            print('--->1.启动数据库：', sleep_time)
            time.sleep(sleep_time)
            time.sleep(5)

            var_names = list(self.action_info.keys())
            send_variables = utils.get_set_variables_string(var_names, self.default, False, 3)
            # print(send_variables)
            # 2.动作变量默认值设置，此时需要确认是否成功连接mysql
            msg, flag_no_reset = utils.send_msg_to_server(
                send_variables,
                # "3$ce$4$buffer_pool_size$134217728$old_blocks_pct$37$old_threshold_ms$1000$innodb_flushing_avg_loops$30",
                self.ip, self.port)
            if flag_no_reset:
                break

        # 给一定时间让系统应用动作参数
        print('--->2.等待5s,数据库应用动作参数')
        time.sleep(5)

        sleep_time = SLEEP_TIME
        buffer_time = BUFFER_TIME
        utils.load_bash_remote(BASH_TYPE)
        # result_f, buffer_time = utils.load_bash(sleep_time)
        print('--->3.等待状态稳定时间：', sleep_time)
        time.sleep(sleep_time)

        # 收集应用参数之后的性能信息,10s
        q_after = utils.get_qps()
        h_after = utils.get_hr()
        globalValue.GLOBAL_BUFFER_POOL_SIZE = utils.get_bps(self.ip, self.port)
        current_state, flag = utils.send_msg_to_server('1',self.ip,self.port)
        current_state = datautils.GetState(current_state)
        print("The current_state is : ", current_state)
        self.labelA, self.labelB = datautils.status_to_labels(current_state)
        # self.last_bp_size = int(utils.show_sql('innodb_buffer_pool_size')[0][1])
        # self.last_bp_size = utils.get_bps(self.ip, self.port)
        # self.last_hr = utils.get_hr(self.ip, self.port)

        # 初始化值
        self.last_hr = h_after
        self.last_qps = q_after
        self.last_bp_size = self.default[0]
        self.hit_t0 = h_after
        self.qps_t0 = q_after
        self.bp_size_0 = self.default[0]
        self.hit_before = self.hit_t0
        self.qps_before = self.qps_t0
        self.unuse_step = 0
        self.bpsize_before = self.bp_size_0
        self.last_action = -2
        # self.score = 0
        print('[Note]The initial hit_ratio = ', self.hit_t0)
        print('[Note]The initial qps = ', self.qps_t0)
        # 等待load_bash结束
        print('--->4.等待load bash结束时间 = ', buffer_time)
        time.sleep(buffer_time)
        utils.send_msg_to_server("exit", self.ip, self.port)
        utils.sshExe(globalValue.CONNECT_MA_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                     globalValue.MYSQLD_CLOSE_EXEC)

        with open("./test_model/qps_store.txt", "a+", encoding="utf-8") as f_qps:
            f_qps.write("qps="+str(q_after)+"\n")
        with open("./test_model/buffer_pool_size.txt", "a+", encoding="utf-8") as f_bp:
            f_bp.write("buffer_pool_size="+str(self.bp_size_0)+"\n")
        with open("./test_model/hit_ratio.txt", "a+", encoding="utf-8") as f_hr:
            f_hr.write("hit_ratio="+str(h_after)+"\n")

        if self.max_hit_ratio < 0:
            self.max_hit_ratio = round(self.last_hr, 2)
            # print('The initial max_hit_ratio = ', self.max_hit_ratio)

        return np.array(current_state)

class NodeState():
    def __init__(self, name, uuid):
        self.name = name
        self.ip = globalValue.CONNECT_CE_IP
        self.port = globalValue.CE_PORT
        self.is_primary = False
        self.uuid = uuid

        # 上一步状态
        self.last_bp_size = 134217728
        self.last_hr = -1
        self.last_action = []
        self.max_hit_ratio = -1
        self.state_now = []
        self.delta_q0 = 0
        self.delta_qt = 0
        self.delta_h0 = 0
        self.delta_ht = 0

        # 初始状态
        self.hit_t0 = -1
        self.qps_t0 = 0
        self.bp_size_0 = 134217728
        # 最佳表现
        self.hit_before = 0
        self.qps_before = 0
        self.bpsize_before = 0

        self.labelA = -1
        self.labelB = -1
        self.labelC = -1
        self.action_trend_choice = 0.9999995
        self.tune_action = {
            # se
            'se_buffer_pool_size': [134217728, 33554432, 3355443200],
            'se_old_blocks_pct': [37, 5, 95],
            'se_old_threshold_ms': [1000, 0, 10000],
            'se_flush_neighbors': [1, 0, 2],
            'se_lru_sleep_time_flush': [1000, 50, 2000],
            'se_flush_n': [1024, 0, 3200],
            'se_SE_LRU_idle_scan_depth': [20, 0, 100],
            'se_lru_scan_depth': [1024, 1024, 10240],
            'se_lru_sleep_time_remove': [1000, 0, 1000],
            'se_reserve_free_page_pct_for_se': [50, 5, 95],
            'se_free_page_threshold': [8192, 1024, 10240],  # down here are new parameters
            # 'se_max_dirty_pages_pct': 11, #[75, 0, 99.99],
            'se_max_dirty_pages_pct_lwm': [0, 0, 99.99],
            'se_adaptive_flushing_lwm': [10, 0, 70],
            'se_flushing_avg_loops': [30, 1, 1000],
            'se_random_read_ahead': [0, 0, 1],
            'se_read_ahead_threshold': [56,0,64],
            'se_io_capacity': [200, 100, 10000],
            # 'se_io_capacity_max': 17,  # [400, 100, 2*32-1],
            # ce 不需要考虑数据页的刷脏，主要考虑数据页的加载和淘汰
            'ce_buffer_pool_size': [134217728, 33554432, 3355443200],
            'ce_old_blocks_pct': [37, 5, 95],
            'ce_old_threshold_ms': [1000, 0, 10000],
            'ce_ce_coordinator_sleep_time': [1024, 50, 2000],
            'ce_ce_free_page_threshold': [8192, 1024, 10240],
            'ce_random_read_ahead': [0, 0, 1],
            'ce_read_ahead_threshold': [56,0,64],
            'ce_flushing_avg_loops': [30, 1, 1000],
        }

    def initActions(self):
        # 主ce从ce的actions有否不同？
        # TODO：action的key信息中要包含唯一标识uuid信息，因为在多个ce和多个se联调情况下，仅看变量名无法准确调整参数，
        # 所有节点的状态都必须加入神经网络去拟合，但是节点的可调参数action可以分情况讨论：
        # 对于SE：因为是对等的，仅把主SE的action输入神经网络，而且调整时是调所有SE（负载均衡？）
        # 对于CE：主CE的action始终需要输入，对于从CE，仅在不同的读写比例下，不同从CE产生的影响可能不同，因此将其action作为神经网络的维度进行输入，否则只输入和调整1个

        # 根据name和is_primary处理tune_action，并且为name加入uuid信息
        keys = list(self.tune_action.keys())
        if(self.name == 'ce'):
            for info in keys:
                key = info[3:]
                val = self.tune_action[info]
                del self.tune_action[info]
                if info[0] == 'c':
                    self.tune_action[key] = val
        else:
            for info in keys:
                key = info[3:]
                val = self.tune_action[info]
                del self.tune_action[info]
                if info[0] == 's':
                    self.tune_action[key] = val

                # 集群（主SE和主CE）环境
class NodesEnv():
    def __init__(self):
        self.info = 'NODES'
        self.se_num = len(globalValue.CONNECT_SE_IP)
        self.ce_num = len(globalValue.CONNECT_CE_IP)
        self.se_info = []
        self.ce_info = []
        # 状态变量的维度
        self.state_dim = 14 * 2  # TODO：增加当前缓冲池大小作为状态之一
        # 动作变量的维度
        self.action_dim = 25
        self.DQN_act_dim = 25
        # 动作变量基本情况 分为SE和CE
        self.all_actions = dict()
        self.action_info = {
                        # se
                        'se_buffer_pool_size': 0,   #[134217728, 33554432, 3355443200]
                        'se_old_blocks_pct': 1,     #[37, 5, 95]
                        'se_old_threshold_ms': 2,   #[1000, 0, 10000]
                        'se_flush_neighbors': 3,    #[default-1, 0,1,2]
                        'se_lru_sleep_time_flush': 4,   #[1000, 50, 2000]
                        'se_flush_n': 5,                #[1024, 0, 3200]
                        'se_SE_LRU_idle_scan_depth': 6,      #[20, 0, 100]
                        'se_lru_scan_depth': 7,  #[1024, 1024, 10240]
                        'se_lru_sleep_time_remove': 8,  #[1000, 0, 1000]
                        'se_reserve_free_page_pct_for_se': 9, #[50, 5, 95]
                        'se_free_page_threshold': 10,  #[8192, 1024, 10240],  # down here are new parameters
                        # 'se_max_dirty_pages_pct': 11, #[75, 0, 99.99],
                        'se_max_dirty_pages_pct_lwm': 11, #[0, 0, 99.99],
                        'se_adaptive_flushing_lwm': 12, #[10, 0, 70],
                        'se_flushing_avg_loops': 13, #[30, 1, 1000],
                        'se_random_read_ahead': 14,  # [0, 0, 1],
                        'se_read_ahead_threshold': 15,  # [56,0,64],
                        'se_io_capacity': 16,  # [200, 100, 2**32-1],
                        # 'se_io_capacity_max': 17,  # [400, 100, 2*32-1],
                        # ce 不需要考虑数据页的刷脏，主要考虑数据页的加载和淘汰
                        'ce_buffer_pool_size': 17,  #[134217728, 33554432, 3355443200]
                        'ce_old_blocks_pct': 18,    #[37, 5, 95]
                        'ce_old_threshold_ms': 19,  #[1000, 0, 10000]
                        'ce_ce_coordinator_sleep_time': 20,    #[1024, 50, 2000]
                        'ce_ce_free_page_threshold': 21,   #[1024, 1, 10240]//#[8192, 1024, 10240]
                        'ce_random_read_ahead': 22, #[0, 0, 1],
                        'ce_read_ahead_threshold': 23,  #[56,0,64],
                        'ce_flushing_avg_loops': 24, #[30, 1, 1000],
                        }

        # 0-经验池预热中，1-经验池预热完毕, 2-eval
        self.state = 0
        # 累积得分 score
        self.score = 0
        self.steps = 0
        self.last_action = -2
        self.all_last_action = []
        self.last_step_mean_score = 0

        # step用，计算无用step
        self.unuse_step = 0
        self.human_exp_hitcnt = 0

        self.episode = 0
        self.eval = 0

        self.method = 'TD3'
        self.expert_exp = False
        self.action_trend_choice = 0.9995

        # 历史最优动作选择---概率衰减且warmup时不加入
        self.best_action_record = []
        self.best_action_choice = 0
        self.last_bps_all = 0
        self.init_bps_all = 0

        # mark the time
        self.start_time = ""
        self.end_time = ""

    # 根据节点数量初始化所有节点信息，以及初始化all_actions信息
    # 格式为 'uuid#innodb_buffer_pool:[default, min, max}]'
    def init_nodes(self):
        self.se_num = len(globalValue.CONNECT_SE_IP)
        self.ce_num = len(globalValue.CONNECT_CE_IP)
        cnt = 0
        for i in range(self.se_num):
            se = NodeState('se', cnt)
            se.ip = globalValue.CONNECT_SE_IP[i]
            se.port = globalValue.SE_PORT
            se.initActions()
            cnt += 1
            self.se_info.append(se)
            keys = list(se.tune_action.keys())
            for info in keys:
                key = str(se.uuid) + '#' + info
                val = se.tune_action.get(info)
                self.all_actions[key] = val

        for i in range(self.ce_num):
            ce = NodeState('ce', cnt)
            ce.ip = globalValue.CONNECT_CE_IP[i]
            if i == 0:
                ce.is_primary = True
            ce.port = globalValue.CE_PORT
            ce.initActions()
            cnt += 1
            self.ce_info.append(ce)
            keys = list(ce.tune_action.keys())
            for info in keys:
                key = str(ce.uuid) + '#' + info
                val = ce.tune_action.get(info)
                self.all_actions[key] = val

        # 状态变量的维度
        self.state_dim = 14 * (self.ce_num + self.se_num)  # TODO：增加当前缓冲池大小作为状态之一
        # 动作变量的维度
        self.action_dim = len(self.all_actions)
        print('all_actions = ', self.all_actions)

    # TODO:缩小action的范围
    def action_scale_shrink(self):
        # 先读取DQN的推荐值 (得到各个维度action获得的最优值？）

        # 再根据每个action维度上的最优值去缩水
        return 0

    # DQN[0, act_dim - 1]
    def init_env_for_DQN(self):
        # 每个动作分三份取中间两点
        self.DQN_act_dim = 2 * self.action_dim

    def explian_DQN_action(self, raw_action):
        action = self.all_last_action
        index = 0
        act_index = 0
        # 定位具体动作的位置
        remain = raw_action % 2
        if remain == 0:
            index = int(raw_action / 2)
        else:
            index = int((raw_action - 1) / 2)
        act_index = index
        # 定位节点
        se_act_n = len(self.se_info[0].tune_action)
        ce_act_n = len(self.ce_info[0].tune_action)
        tune_act = dict()
        if index < self.se_num * se_act_n:
            index = index % se_act_n
            tune_act = self.se_info[0].tune_action
        else:
            index = index % ce_act_n
            tune_act = self.ce_info[0].tune_action
        cnt = 0
        scale = []
        for scal in tune_act.values():
            if cnt == index:
                scale = scal
                break
            cnt += 1
        low = int(scale[1])
        high = int(scale[2])
        # 还原real_action
        real_action = low + ((high - low) / 3.0) * (1 + remain)
        action[act_index] = utils.real_action_to_action(real_action, low, high)
        print('index = {}, real_action = {}, low = {}, high = {}'.format(act_index, real_action, low, high))
        return action

    # 启动某节点
    def connect_with_node(self, node, real_action, flag_restart):
        buffer_set = ' --innodb_buffer_pool_size=' + str(real_action[0])
        #if node.name == 'se':
         #   start_cmd = globalValue.MYSQLD_OPEN_EXEC_SE + buffer_set + globalValue.MYSQLD_OUTPUT_SE
          #  close_cmd = globalValue.MYSQLD_SE_CLOSE_EXEC
           # if BASH_TYPE == 'sysbench':
            #    new_data_cmd = globalValue.NEW_DATA_SE_CMD
            #else:
             #   new_data_cmd = globalValue.NEW_DATA_SE_TPCC_CMD
        #else:
         #   if node.is_primary == False:
          #      start_cmd = globalValue.MYSQLD_OPEN_EXEC_CE + "--sehost=" + globalValue.CONNECT_SE_IP[0] + buffer_set + globalValue.MYSQLD_OUTPUT_CE
           # else:
            #    start_cmd = globalValue.MYSQLD_OPEN_EXEC_CE + "--sehost=127.0.0.1" + buffer_set + globalValue.MYSQLD_OUTPUT_CE
           # close_cmd = globalValue.MYSQLD_CE_CLOSE_EXEC
           # if BASH_TYPE == 'sysbench':
            #    new_data_cmd = globalValue.NEW_DATA_CE_CMD
            #else:
             #   new_data_cm
        if node.name == 'se':
            # start_cmd = globalValue.MYSQLD_OPEN_EXEC_SE + "--sehost=" + globalValue.CONNECT_CE_IP[0] + buffer_set + globalValue.MYSQLD_OUTPUT_SE
            start_cmd = globalValue.MYSQLD_OPEN_EXEC_SE + "--sehost=%"  + buffer_set + globalValue.MYSQLD_OUTPUT_SE
            close_cmd = globalValue.MYSQLD_SE_CLOSE_EXEC
            if BASH_TYPE == 'sysbench':
                new_data_cmd = globalValue.NEW_DATA_SE_CMD
            else:
                new_data_cmd = globalValue.NEW_DATA_SE_TPCC_CMD

        else:
            if node.ip!=globalValue.CONNECT_SE_IP[0]:
                start_cmd = globalValue.MYSQLD_OPEN_EXEC_CE + "--sehost=" + globalValue.CONNECT_SE_IP[0] + buffer_set + globalValue.MYSQLD_OUTPUT_CE
            else:
                start_cmd = globalValue.MYSQLD_OPEN_EXEC_CE + "--sehost=127.0.0.1" + buffer_set + globalValue.MYSQLD_OUTPUT_CE
            close_cmd = globalValue.MYSQLD_CE_CLOSE_EXEC
            if BASH_TYPE == 'sysbench':
                new_data_cmd = globalValue.NEW_DATA_CE_CMD
            else:
                new_data_cmd = globalValue.NEW_DATA_CE_TPCC_CMD


        print("\nnode_name=" + node.name + "\t" + node.ip)
        print(start_cmd + "\n")

        # first time restart, need make new data
        if flag_restart == True:
            text = utils.start_for_a_fresh_node(start_cmd, new_data_cmd)
            # start node
            utils.sshExe(node.ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, text)
        else:
            text = utils.start_for_a_not_fresh_node(close_cmd, start_cmd)
            # start node
            utils.sshExe(node.ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, text)
        if node.name == 'ce':
            time.sleep(10)
        delta_bps = real_action[0] - node.last_bp_size
        if delta_bps > 0:
            sleep_time = delta_bps / 1024 / 1024 / 256
        else:
            sleep_time = abs(delta_bps) / 1024 / 1024 / 128
        print('--->1.启动%s%d' % (node.name, node.uuid))
        time.sleep(sleep_time+5)
        # 将推荐参数封装为发送数据标准格式
        var_names = list(node.tune_action.keys())
        send_variables = utils.get_set_variables_string(var_names, real_action, node.name, 3)
        # 动作变量默认值设置，此时需要确认是否成功连接mysql
        received_msg, flag = utils.send_msg_to_server(send_variables, node.ip, node.port)
        return received_msg, flag, send_variables

    # def start_all_nodes(self, real_action, ses_send, ces_send):
    #     flag = True
    #     flag_se = True
    #     flag_ce = True
    #     # 需要考虑启动失败的情况（异常处理）
    #     reset_times = 0
    #     while True:
    #         self.close_all_nodes()
    #         reset_times += 1
    #         if reset_times == 3:
    #             flag = False
    #             break
    #         # 依次启动所有se和ce
    #         cnt = 0
    #         for se in self.se_info:
    #             received_msg_se, flag_se, send_variables_se = self.connect_with_node(se, real_action[cnt], flag_se)
    #             if flag_se == False:
    #                 break
    #             ses_send.append(send_variables_se)
    #             cnt += 1
    #             time.sleep(2)

    #         if flag_se == False:
    #             continue

    #         for ce in self.ce_info:
    #             received_msg_ce, flag_ce, send_variables_ce = self.connect_with_node(ce, real_action[cnt], flag_ce)
    #             if flag_ce == False:
    #                 time.sleep(5)
    #                 received_msg_ce, flag_ce, send_variables_ce = self.connect_with_node(ce, real_action[cnt], flag_ce)
    #                 if flag_ce == False:
    #                     break
    #             ces_send.append(send_variables_ce)
    #             cnt += 1
    #             time.sleep(2)

    #         if flag_ce == False:
    #             continue
    #         else:
    #             break

    #     return flag




    def start_all_nodes(self, real_action, ses_send, ces_send):
        flag = True
        flag_se = True
        flag_ce = True
        # 需要考虑启动失败的情况（异常处理）
        

        if globalValue.USE_FIX_ACTION == True:
            real_action = globalValue.FIX_ACTION


        reset_times = 0

        for x in range(len(globalValue.CONNECT_SE_IP)):
            ses_send.append("")

        for x in range(len(globalValue.CONNECT_CE_IP)):
            ces_send.append("")
        


        #ses_send = ["" for x in range(len(globalValue.CONNECT_SE_IP))]
        #ces_send = ["" for x in range(len(globalValue.CONNECT_CE_IP))]
        while True:
            self.close_all_nodes()
            reset_times += 1
            if reset_times == 3:
                flag = False
                break
            # 依次启动所有se和ce
            cnt = 0
            
            
            def connect_with_se(node, count, flag_se):
                received_msg_se, flag_se, send_variables_se = self.connect_with_node(node, real_action[count], flag_se)
                if flag_se == False:
                    return
                ses_send[count] = send_variables_se
                time.sleep(2)
            
            ts_se = []
            for se in self.se_info:
                t = threading.Thread(target=connect_with_se, args=(se, cnt, flag_se))
                ts_se.append(t)
                cnt += 1
            for t in ts_se:
                t.start()
            for t in ts_se:
                t.join()
            
            

            if flag_se == False:
                continue

            
            def connect_with_ce(node, count, flag_ce):
                received_msg_ce, flag_ce, send_variables_ce = self.connect_with_node(node, real_action[count], flag_ce)
                if flag_ce == False:
                    time.sleep(5)
                    received_msg_ce, flag_ce, send_variables_ce = self.connect_with_node(node, real_action[count], flag_ce)
                    if flag_ce == False:
                        return
                ces_send[count - len(globalValue.CONNECT_SE_IP)] = send_variables_ce
                time.sleep(2)

            ts_ce = []
            for idx, ce in enumerate(self.ce_info):
                if idx == 0:
                    connect_with_ce(ce, cnt, flag_ce)
                    cnt += 1
                else:
                    t = threading.Thread(target=connect_with_ce, args=(ce, cnt, flag_ce))
                    ts_ce.append(t)
                    cnt += 1
                
            for t in ts_ce:
                t.start()
            for t in ts_ce:
                t.join()
                

            if flag_ce == False:
                continue
            else:
                break

        return flag








    def exit_all_nodes(self):
        for ce in self.ce_info:
            utils.send_msg_to_server("exit", ce.ip, ce.port)
            utils.sshExe(ce.ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.MYSQLD_CE_CLOSE_EXEC)
        for se in self.se_info:
            utils.send_msg_to_server("exit", se.ip, se.port)
            utils.sshExe(se.ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.MYSQLD_SE_CLOSE_EXEC)

    def close_all_nodes(self):
        for ce in self.ce_info:
            utils.sshExe(ce.ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.MYSQLD_CE_CLOSE_EXEC)
        for se in self.se_info:
            utils.sshExe(se.ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                         globalValue.MYSQLD_SE_CLOSE_EXEC)

    def get_all_nodes_state(self):
        current_nodes_state = []
        for se in self.se_info:
            current_state_se, flag1 = utils.send_msg_to_server('1', se.ip, se.port)
            # print("current state se->: ", current_state_se)
            current_state_se = datautils.GetNodeState(current_state_se, se.name)
            se.state_now = current_state_se
            current_nodes_state += current_state_se
            if self.expert_exp == True and self.state == 1:
                datautils.status_to_labels(se)

        for ce in self.ce_info:
            current_state_ce, flag1 = utils.send_msg_to_server('1', ce.ip, ce.port)
            # print("current state ce->: ", current_state_ce)
            current_state_ce = datautils.GetNodeState(current_state_ce, ce.name)
            ce.state_now = current_state_ce
            current_nodes_state += current_state_ce
            if self.expert_exp == True and self.state == 1:
                datautils.status_to_labels(ce)

        print("The current_state is : ", current_nodes_state)
        return current_nodes_state

    def print_all_nodes(self):
        print('[Note]The state of all nodes after apply action:')
        for se in self.se_info:
            print('Node {}{}: hit_ratio = {}, bp_size = {}'.format(se.name, se.uuid, se.last_hr, se.last_bp_size))
        for ce in self.ce_info:
            if ce.is_primary == True:
                print('Node {}{}: hit_ratio = {}, bp_size = {}, qps = {}'.format(ce.name, ce.uuid, ce.last_hr, ce.last_bp_size, ce.last_qps))
            else:
                print('Node {}{}: hit_ratio = {}, bp_size = {}'.format(ce.name, ce.uuid, ce.last_hr,
                                                                                 ce.last_bp_size))
        return

    # action为一组动作变量的值
    def step(self, action):
        self.last_action = 1
        # print('-----------------------one step----------------------')
        # 将action转为可应用的值
        # real_actions = utils.action_change(self, action)
        # 按节点划分real_action
        real_action = utils.split_action(self, action)
        print(real_action)
        
        ces_send = []
        ses_send = []
        # 依次启动所有se和ce
        cnt = 0
        start_flag = self.start_all_nodes(real_action, ses_send, ces_send)
        if start_flag == False:
            return np.array([0]), 0, True, {}

        print("000000000000000000000000000 ces_send =  00000000000000000000000000")
        print(ces_send)
        print("000000000000000000000000000 ses_send =  00000000000000000000000000")
        print(ses_send)


        print('--->2.load bash>>>')
        #time.sleep(100000)
        sleep_time = SLEEP_TIME
        buffer_time = BUFFER_TIME
        utils.load_bash_remote(BASH_TYPE)
        # result_f, buffer_time = utils.load_bash(sleep_time)
        print('--->3.等待状态稳定时间：', sleep_time)
        time.sleep(sleep_time)

        # 收集应用参数之后的性能信息,10s
        # 分别收集各个节点的信息并计算reward 先se再ce
        reward_ces = 0
        reward_ses = 0
        reward_qps = []
        reward_hit_r = []
        # reward_bp_size = []
        bps_all = []
        # 获取当前节点状态 计算奖励 更新节点状态
        try:
            cnt = 0
            for se in self.se_info:
                # q_after_se = utils.get_node_qps(se)
                h_after_se = utils.get_se_hr(se)
                bps_se = utils.get_bps(se.ip, se.port)
                globalValue.GLOBAL_BUFFER_POOL_SIZE_SE.append(bps_se)
                # reward_ses += utils.cal_reward_se_single_node(se, se.hit_before, h_after_se, real_action[cnt][0])
                reward_h = utils.cal_reward_se_single_node(se, se.hit_before, h_after_se, real_action[cnt][0])
                print("[REWARD LOG] se reward_h = {0}".format(reward_h))
                reward_hit_r.append(reward_h)
                # reward_bp_size.append(reward_b)
                bps_all.append(bps_se)
                # 当前qps和当前bp size为下一阶段的上一个qps和上一个bp size
                se.last_hr = h_after_se
                se.last_bp_size = bps_se
                # se.last_qps = q_after_se
                se.last_action = real_action[cnt]
                cnt += 1

            
            def paralell_get_nodes_qps_base(parallel_qps_list, node):
                q_after_node = utils.get_node_qps(node)
                parallel_qps_list.append(q_after_node)
            
            
            parallel_qps_list = []
            
            for ce in self.ce_info:
                # if ce.is_primary == True:
                #     q_after_ce = utils.get_node_qps(ce)
                # if globalValue.CE_LOAD_NUM > 1 and ce.is_primary == True:
                #     for i in range(globalValue.CE_LOAD_NUM - 1):
                #         q_after_ce += utils.get_node_qps(ce)
                q_after_ce = 0
                if ce.is_primary == True:
                    qps_ts_list = []
                    for i in range(globalValue.CE_LOAD_NUM):
                        t = threading.Thread(target = paralell_get_nodes_qps_base, args=(parallel_qps_list, self.ce_info[i]))
                        qps_ts_list.append(t)
                    for t in qps_ts_list:
                        t.start()
                    for t in qps_ts_list:
                        t.join()
                    
                    for qps in parallel_qps_list:
                        q_after_ce += qps
                        
                
                h_after_ce = utils.get_node_hr(ce)
                bps_ce = utils.get_bps(ce.ip, ce.port)
                bps_all.append(bps_ce)
                globalValue.GLOBAL_BUFFER_POOL_SIZE_CE.append(bps_ce)
                if ce.is_primary:
                    # reward_ces += utils.cal_reward_ce_single_node(ce, ce.hit_before, h_after_ce, ce.qps_before, q_after_ce, real_action[cnt][0])
                    reward_q, reward_h, delta_q0 = utils.cal_reward_ce_single_node(ce, ce.hit_before, h_after_ce, ce.qps_before, q_after_ce, real_action[cnt][0])
                    print("[REWARD LOG] p_ce reward_q = {0}, reward_h = {1}, delta_q0 = {2}".format(reward_q, reward_h, delta_q0))
                    reward_qps.append(reward_q)
                    reward_hit_r.append(reward_h)
                    # reward_bp_size.append(reward_b)
                else:
                    # reward_ces += utils.cal_reward_se_single_node(ce, ce.hit_before, h_after_ce, real_action[cnt][0])
                    reward_h = utils.cal_reward_se_single_node(ce, ce.hit_before, h_after_ce, real_action[cnt][0])
                    print("[REWARD LOG] s_ce reward_h = {0}".format(reward_h))
                    reward_hit_r.append(reward_h)
                    # reward_bp_size.append(reward_b)
                # 当前qps和当前bp size为下一阶段的上一个qps和上一个bp size
                ce.last_hr = h_after_ce
                ce.last_bp_size = bps_ce
                if ce.is_primary == True:
                    ce.last_qps = q_after_ce
                ce.last_action = real_action[cnt]
                cnt += 1
            # 获取当前状态,注意对于send_msg_to_server接收回的数据要做相应处理
            current_state = self.get_all_nodes_state()
        except Exception as e:
            print(e)
            return np.array([0]), 0, True, {}

        present_bps_all = np.sum(bps_all)
        if self.last_bps_all >= present_bps_all and self.init_bps_all >= present_bps_all:
            reward_bps = 1
        else:
            reward_bps = -1

        self.last_bps_all = present_bps_all
        # 计算均值，之后再ce和se一起计算
        # reward_ces = reward_ces / (self.ce_num * 1.0)
        # reward_ses = reward_ses / (self.se_num * 1.0)
        # SE和CE总奖励
        # reward = reward_ces * 0.5 + reward_ses * 0.5
        wq = 0.3
        wb = 0.6
        wh = 0.1
        # reward = np.mean(reward_qps) * wq + np.mean(reward_bp_size) * wb + np.mean(reward_hit_r) * wh
        if self.init_bps_all >= present_bps_all and delta_q0 > 0:
            print("[REWARD LOG] FIND self.init_bps_all >= present_bps_all and delta_q0 > 0 !!!!!")
            print("[REWARD LOG] init_bps_all = {0}, present_bps_all = {1}, delta_q0 = {2}".format(self.init_bps_all, present_bps_all, delta_q0))
            delta_b0 = (self.init_bps_all - present_bps_all) * 1.0 / self.init_bps_all
            print("[REWARD LOG] delta_b0 = {0}".format(delta_b0))
            reward = pow(1 + delta_q0, 2) * wq + delta_b0 * 0.5 * wb + np.mean(reward_hit_r) * wh
        else:
            print("[REWARD LOG] FIND self.init_bps_all < present_bps_all or delta_q0 <= 0 !!!!! ")
            print("[REWARD LOG] init_bps_all = {0}, present_bps_all = {1}, delta_q0 = {2}".format(self.init_bps_all, present_bps_all, delta_q0))
            reward = np.mean(reward_qps) * wq + reward_bps * wb + np.mean(reward_hit_r) * wh
            if reward > 0:
                reward = 0
        print("[REWARD LOG] final reward = {0}".format(reward))
        reward_raw = reward
        if reward > 0:
            reward = reward * 1000000
        elif reward < -1:
            reward = -1

        globalValue.REWARD_NOW = reward
        globalValue.MAX_REWARD = max(reward, globalValue.MAX_REWARD)
        print('@@@one step action @@@reward is ', reward)
        if self.state != 2:
            self.score += reward
            self.steps += 1
        # 根据当前reward大小判断是否是best
        # flag = utils.record_best_nodes(q_after_ce, h_after_ce, bps_ce, h_after_se, bps_se)
        flag = utils.record_all_best(self.se_info, self.ce_info)
        if flag == True:
            print('Better performance changed!')
            self.best_action_record = action
            with open("./test_model/best_action.log", "a+", encoding="utf-8") as f_best_action:
                if self.state == 0:
                    f_best_action.write("===Warmup step %d," % (self.steps))
                elif self.state == 1:
                    f_best_action.write("===Episode %d step %d," % (self.episode, self.steps))
                else:
                    f_best_action.write("===Eval %d step %d," % (self.eval, self.steps))
                f_best_action.write(" reward = %.3f , recommand actions of bestnow : \n" % (reward))
                print("BNbBBBBB000000000000000000000000000 ces_send =  00000000000000000000000000")
                print(ces_send)
                print("BBBBBBB000000000000000000000000000 ses_send =  00000000000000000000000000")
                print(ses_send)
 
                for ce_s in ces_send:
                    f_best_action.write(ce_s + "\n")
                for se_s in ses_send:
                    f_best_action.write(se_s + "\n")
            # record best performance for every node
            best_now_performance = utils.get_best_now_all_nodes('bestnow.log')
            # 先ce 再se
            cnt = 0
            for ce in self.ce_info:
                if ce.is_primary == True:
                    ce.qps_before = float(best_now_performance[cnt])
                    ce.hit_before = float(best_now_performance[cnt + 1])
                    ce.bpsize_before = int(best_now_performance[cnt + 2])
                    cnt += 3
                else:
                    ce.hit_before = float(best_now_performance[cnt])
                    ce.bpsize_before = int(best_now_performance[cnt + 1])
                    cnt += 2

            for se in self.se_info:
                se.hit_before = float(best_now_performance[cnt])
                se.bpsize_before = int(best_now_performance[cnt + 1])
                cnt += 2

        else:
            print('Performance remained!')
            # get the best performance so far to calculate the reward

        # 一个文件内记录多个节点的值
        with open("./test_model/qps_store.txt", "a+", encoding="utf-8") as f_qps:
            for ce in self.ce_info:
                if ce.is_primary == True:
                    # $node1@qps=xxx$node2@qps=xxx
                    f_qps.write("node" + str(ce.uuid) + "@qps=" + str(ce.last_qps) + "$")
            f_qps.write("\n")

        with open("./test_model/buffer_pool_size_se.txt", "a+", encoding="utf-8") as f_bp_se:
            for se in self.se_info:
                f_bp_se.write("node" + str(se.uuid) + "@buffer_pool_size_se=" + str(se.last_bp_size) + "$")
            f_bp_se.write("\n")

        with open("./test_model/buffer_pool_size_ce.txt", "a+", encoding="utf-8") as f_bp_ce:
            for ce in self.ce_info:
                f_bp_ce.write("node" + str(ce.uuid) + "@buffer_pool_size_ce=" + str(ce.last_bp_size) + "$")
            f_bp_ce.write("\n")

        with open("./test_model/hit_ratio_se.txt", "a+", encoding="utf-8") as f_hr_se:
            for se in self.se_info:
                f_hr_se.write("node" + str(se.uuid) + "@hit_ratio_se=" + str(se.last_hr) + "$")
            f_hr_se.write("\n")

        with open("./test_model/hit_ratio_ce.txt", "a+", encoding="utf-8") as f_hr_ce:
            for ce in self.ce_info:
                f_hr_ce.write("node" + str(ce.uuid) + "@hit_ratio_ce=" + str(ce.last_hr) + "$")
            f_hr_ce.write("\n")

        if self.state == 0:
            with open('./test_model/actions_reward.csv', 'a', encoding='utf-8', newline='') as file_obj:
                b = [item for sublist in real_action for item in sublist]
                b.append(reward_raw)
                # 1:创建writer对象
                writer = csv.writer(file_obj)
                # 2:写表
                writer.writerow(b)

        self.print_all_nodes()

        # if globalValue.EVAL_TEST:
        #     f = open("./1ce/recommand_knobs/knobs.txt", 'a')
            # f.write('=======================The recommand knobs==========================')
            # f.write("\n")
            # f.write(ses_send+" "+ces_send)
            # f.write('        reward = ')
            # f.write(str(reward))
            # f.write("\n")
            # f.close()

        next_state = current_state
        # !!!!!!!考虑好什么是最终情况!!!!!!!!!!!
        # 不需要最终情况？
        done = False
        # print('[Note]The hit_ratio after apply action = se {}, ce {}'.format(self.node1.last_hr, self.node2.last_hr))
        # print('[Note]The bp_size after apply action = se {}, ce {}'.format(self.node1.last_bp_size, self.node2.last_bp_size))
        # print('[Note]The qps after apply action = ', self.node2.last_qps)
        # globalValue.SE_FLAG = done
        print('--->4.等待load bash结束时间：', buffer_time)
        time.sleep(buffer_time)
        self.exit_all_nodes()
        self.all_last_action = action
        # utils.send_msg_to_server("exit", self.node2.ip, self.node2.port)
        # utils.send_msg_to_server("exit", self.node1.ip, self.node1.port)
        # utils.sshExe(globalValue.CONNECT_CE_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
        #              globalValue.MYSQLD_CLOSE_EXEC)
        # utils.sshExe(globalValue.CONNECT_SE_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
        #              globalValue.MYSQLD_CLOSE_EXEC)
        # current_time = time.time()

        # 依据此轮训练的时间来判断是否done，若时间不够再进行一次取样，则done为True
        # if time_remain < 85:
        #     done = True
        #     globalValue.SE_FLAG = done
        #     # 睡20秒给压测清理环境
        #     time.sleep(time_remain + 20)
        #  print('Episode time_remain: ', time_remain)
        #  连续4次qps变化不大则判断结束
        # if abs(delta_qps) < 50:
        #     self.unuse_step += 1
        # else:
        #     self.unuse_step = 0
        #
        # if self.unuse_step == 4:
        #     self.unuse_step = 0

        # if self.score < -50:
        #     done = True

        # TODO:持久化action信息

        return np.array(next_state), reward, done, {}

    def reset(self):
        # 一个episode的开始，重置环境
        if globalValue.EVAL_TEST == True:
            print("==========>{}===eval_reset====>".format(self.info))
        else:
            print("==========>{}===reset====>".format(self.info))

        self.last_action = 1
        # print('-----------------------one step----------------------')
        # 按节点划分real_action
        real_action = utils.split_default_actions(self)
        ces_send = []
        ses_send = []
        bps_all = []
        start_flag = self.start_all_nodes(real_action, ses_send, ces_send)
        if start_flag == False:
            return np.array([0]), True

        print('--->2.load bash>>>')
        sleep_time = SLEEP_TIME
        buffer_time = BUFFER_TIME
        #time.sleep(100000)
        utils.load_bash_remote(BASH_TYPE)
        # result_f, buffer_time = utils.load_bash(sleep_time)
        print('--->3.等待状态稳定时间：', sleep_time)
        time.sleep(sleep_time)

        try:
            # 收集应用参数之后的性能信息,10s
            # 获取当前节点状态 计算奖励 更新节点状态
            cnt = 0
            for se in self.se_info:
                # q_after_se = utils.get_node_qps(se)
                h_after_se = utils.get_se_hr(se)
                bps_se = utils.get_bps(se.ip, se.port)
                globalValue.GLOBAL_BUFFER_POOL_SIZE_SE.append(bps_se)
                se.hit_t0 = se.last_hr = se.hit_before = h_after_se
                se.bp_size_t0 = se.last_bp_size = se.bpsize_before = bps_se
                bps_all.append(bps_se)
                # se.qps_t0 = se.last_qps = seq_after_se
                se.last_action = -2
                cnt += 1

            # for ce in self.ce_info:
            #     if ce.is_primary == True:
            #         q_after_ce = utils.get_node_qps(ce)
            #     if globalValue.CE_LOAD_NUM > 1 and ce.is_primary == True:
            #         for i in range(globalValue.CE_LOAD_NUM - 1):
            #             q_after_ce += utils.get_node_qps(ce)
            
            def paralell_get_nodes_qps_base(parallel_qps_list, node):
                q_after_node = utils.get_node_qps(node)
                parallel_qps_list.append(q_after_node)
            
            
            parallel_qps_list = []
            
            for ce in self.ce_info:
                # if ce.is_primary == True:
                #     q_after_ce = utils.get_node_qps(ce)
                # if globalValue.CE_LOAD_NUM > 1 and ce.is_primary == True:
                #     for i in range(globalValue.CE_LOAD_NUM - 1):
                #         q_after_ce += utils.get_node_qps(ce)
                q_after_ce = 0
                if ce.is_primary == True:
                    qps_ts_list = []
                    for i in range(globalValue.CE_LOAD_NUM):
                        t = threading.Thread(target = paralell_get_nodes_qps_base, args=(parallel_qps_list, self.ce_info[i]))
                        qps_ts_list.append(t)
                    for t in qps_ts_list:
                        t.start()
                    for t in qps_ts_list:
                        t.join()
                    
                    for qps in parallel_qps_list:
                        q_after_ce += qps
            
            
            
            
                h_after_ce = utils.get_node_hr(ce)
                bps_ce = utils.get_bps(ce.ip, ce.port)
                globalValue.GLOBAL_BUFFER_POOL_SIZE_CE.append(bps_ce)
                # 当前qps和当前bp size为下一阶段的上一个qps和上一个bp size
                ce.hit_t0 = ce.last_hr = ce.hit_before = h_after_ce
                ce.bp_size_t0 = ce.last_bp_size = ce.bpsize_before = bps_ce
                bps_all.append(bps_ce)
                # TODO: 从机不加入qps的计算
                if ce.is_primary == True:
                    ce.qps_t0 = ce.last_qps = ce.qps_before = q_after_ce
                ce.last_action = -2
                cnt += 1
            current_state = self.get_all_nodes_state()
        except Exception as e:
            print(e)
            return np.array([0]), True

        self.unuse_step = 0
        self.human_exp_hitcnt = 0
        self.last_step_mean_score = 0
        self.print_all_nodes()
        self.init_bps_all = self.last_bps_all = np.sum(bps_all)
        # 等待load_bash结束
        print('--->4.等待load bash结束时间 = ', buffer_time)
        time.sleep(buffer_time)
        self.exit_all_nodes()

        # 一个文件内记录多个节点的值
        with open("./test_model/qps_store.txt", "a+", encoding="utf-8") as f_qps:
            for ce in self.ce_info:
                # $node1@qps=xxx$node2@qps=xxx
                if ce.is_primary == True:
                    f_qps.write("node" + str(ce.uuid) + "@qps=" + str(ce.last_qps) + "$")
            f_qps.write("\n")

        with open("./test_model/buffer_pool_size_se.txt", "a+", encoding="utf-8") as f_bp_se:
            for se in self.se_info:
                f_bp_se.write("node" + str(se.uuid) + "@buffer_pool_size_se=" + str(se.last_bp_size) + "$")
            f_bp_se.write("\n")

        with open("./test_model/buffer_pool_size_ce.txt", "a+", encoding="utf-8") as f_bp_ce:
            for ce in self.ce_info:
                f_bp_ce.write("node" + str(ce.uuid) + "@buffer_pool_size_ce=" + str(ce.last_bp_size) + "$")
            f_bp_ce.write("\n")

        with open("./test_model/hit_ratio_se.txt", "a+", encoding="utf-8") as f_hr_se:
            for se in self.se_info:
                f_hr_se.write("node" + str(se.uuid) + "@hit_ratio_se=" + str(se.last_hr) + "$")
            f_hr_se.write("\n")

        with open("./test_model/hit_ratio_ce.txt", "a+", encoding="utf-8") as f_hr_ce:
            for ce in self.ce_info:
                f_hr_ce.write("node" + str(ce.uuid) + "@hit_ratio_ce=" + str(ce.last_hr) + "$")
            f_hr_ce.write("\n")

        # with open("./test_model/qps_store.txt", "a+", encoding="utf-8") as f_qps:
        #     f_qps.write("qps="+str(q_after_ce)+"\n")
        # with open("./test_model/buffer_pool_size_se.txt", "a+", encoding="utf-8") as f_bp_se:
        #     f_bp_se.write("buffer_pool_size_se="+str(self.node1.bp_size_0)+"\n")
        # with open("./test_model/buffer_pool_size_ce.txt", "a+", encoding="utf-8") as f_bp_ce:
        #     f_bp_ce.write("buffer_pool_size_ce="+str(self.node2.bp_size_0)+"\n")
        # with open("./test_model/hit_ratio_se.txt", "a+", encoding="utf-8") as f_hr_se:
        #     f_hr_se.write("hit_ratio_se="+str(h_after_se)+"\n")
        # with open("./test_model/hit_ratio_ce.txt", "a+", encoding="utf-8") as f_hr_ce:
        #     f_hr_ce.write("hit_ratio_ce="+str(h_after_ce)+"\n")

        return np.array(current_state), False


if __name__ == '__main__':
    env = NodesEnv()
    env.init_nodes()
    real_action = utils.split_default_actions(env)
    ces_send = []
    ses_send = []
    start_flag = env.start_all_nodes(real_action, ses_send, ces_send)
    print(start_flag)
    print(ces_send)
    print(ses_send)
    # utils.load_bash_remote(BASH_TYPE)



    # index_se = env.action_info['se_buffer_pool_size']
    # index_ce = env.action_info['ce_buffer_pool_size']
    # # buffer_set_se = '--innodb_buffer_pool_size=' + str(env.max_info[index_se])
    # # buffer_set_ce = '--innodb_buffer_pool_size=' + str(env.max_info[index_ce])
    # buffer_set_se = ' --innodb_buffer_pool_size=335544320'
    # buffer_set_ce = ' --innodb_buffer_pool_size=335544320'
    # utils.sshExe(globalValue.CONNECT_CE_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
    #              globalValue.MYSQLD_CLOSE_EXEC)
    # utils.sshExe(globalValue.CONNECT_SE_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
    #              globalValue.MYSQLD_CLOSE_EXEC)
    # time.sleep(1)
    # # utils.sshExe(globalValue.CONNECT_CE_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
    # #              globalValue.NEW_DATA_CMD)
    # utils.sshExe(globalValue.CONNECT_SE_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
    #              globalValue.NEW_DATA_CMD)
    # time.sleep(1)
    # # start se
    # utils.sshExe(globalValue.CONNECT_SE_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
    #              globalValue.MYSQLD_OPEN_EXEC_SE + buffer_set_se + globalValue.MYSQLD_OUTPUT_SE)
    # time.sleep(1)
    # # then start ce
    # utils.sshExe(globalValue.CONNECT_CE_IP, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
    #              globalValue.MYSQLD_OPEN_EXEC_CE + buffer_set_ce + globalValue.MYSQLD_OUTPUT_CE)
    # print('send -->')
    # # 将推荐参数封装为发送数据标准格式
    # var_names = list(env.action_info.keys())
    # send_variables_se, send_variables_ce = utils.get_set_variables_string_nodes(var_names, env.default, 3)
    # # 2.动作变量默认值设置，此时需要确认是否成功连接mysql
    # received_msg_se, flag_se = utils.send_msg_to_server(send_variables_se, env.node1.ip, env.node1.port)
    # received_msg_ce, flag_ce = utils.send_msg_to_server(send_variables_ce, env.node2.ip, env.node2.port)
    # utils.send_msg_to_server("3$se$9$buffer_pool_size$134217728$old_blocks_pct$37$old_threshold_ms$1000$flush_neighbors$1$lru_sleep_time_flush$1000$flush_n$0$SE_LRU_scan_depth$20$lru_sleep_time_remove$0$lru_scan_depth$8192",
    #                          env.node1.ip, env.node1.port)
    # utils.send_msg_to_server(
    #     "3$se$9$buffer_pool_size$134217728$old_blocks_pct$37$old_threshold_ms$1000$flush_neighbors$1$lru_sleep_time_flush$1000$flush_n$0$SE_LRU_scan_depth$20$lru_sleep_time_remove$0$lru_scan_depth$8192",
    #     env.node1.ip, env.node1.port)
    #utils.send_msg_to_server("3$ce$1375731712$50$3586$1485$8473", env.ip, env.port)

    #while True:
        #time.sleep(10)
        #h_after = utils.get_hr(env.ip, env.port)
        #qps = utils.get_qps()
        #print('h_after: ', h_after)
