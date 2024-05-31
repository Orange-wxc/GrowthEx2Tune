# utils.py
# 操作数据库的文件
import csv
import logging
import os
import random
import re
import datetime

import numpy as np
import paramiko as paramiko
import pymysql
import time
import globalValue
import socket
from parl.utils import action_mapping
import sys

################################################
############   不可使用mysql客户端   ##############
############ 与buf0tune.cc结合使用  ##############
################################################

# 远程连接linux, 并实现远程启动数据库
def sshExe(sys_ip,username,password,cmd):
    client = None
    result = None
    # print(cmd)
    try:
        #创建ssh客户端
        client = paramiko.SSHClient()
        #第一次ssh远程时会提示输入yes或者no
        # if globalValue.SSH_CNT <= 5:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        #密码方式远程连接
        client.connect(sys_ip, 22, username=username, password=password, timeout=20)
        #互信方式远程连接
        #key_file = paramiko.RSAKey.from_private_key_file("/root/.ssh/id_rsa")
        #ssh.connect(sys_ip, 22, username=username, pkey=key_file, timeout=20)
        #执行命令
        # stdin, stdout, stderr = client.exec_command(cmds)
        stdin, stdout, stderr = client.exec_command(cmd)
        #获取命令执行结果,返回的数据是一个list
        result = stdout.readlines()
    except Exception as e:
        print(e)
    finally:
        client.close()
        return result


# 将给定字符串发送给数据库
def send_msg_to_server(msg, ip, port):
    p = None
    received_msg = None
    # print('send message-> : ', msg)
    try:
        # 建立连接
        # print('msg = {} ip = {} port = {}'.format(msg, ip, port))
        p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        p.connect((ip, port))
        # 发送消息
        p.send(msg.encode('utf-8'))
        # 接收反馈
        received_msg = p.recv(1024).decode('utf-8')
        # print('received_msg = ',received_msg)
        p.send("exit".encode('utf-8'))
    except Exception as e:
        print(e)
        # print('Meet set variables failure!!')
        p.close()
        return received_msg, False
    # 关闭连接
    p.close()
    return received_msg, True

# 设置给定参数值,即将给定值封装
# 参数按照如下顺序排列
# [buffer_pool_size, old_blocks_pct, old_threshold_ms, flush_neighbors
#  sleep_time_flush, flush_n,        se_lru_scan_depth, sleep_time_remove,
#  lru_scan_depth,   RESERVE_FREE_PAGE_PCT_FOR_SE, FREE_PAGE_THRESHOLD]
# 封装后的数据格式为：set$1.0$2.0$4$3$10000$
def get_set_variables_string(var_names, new_values, node_name, type):
    length = len(var_names)
    s = str(type) + '$'
    if node_name == 'se':
        s = s + 'se'
    else:
        s = s + 'ce'
    # 封装参数个数
    s = s + '$' + str(length)
    # 增加参数姓名的封装
    for i in range(length):
        s = s + '$' + var_names[i] + '$' + str(new_values[i])
    print(s)
    return s

# 按照前缀划分SE和CE节点发送的参数
def get_set_variables_string_nodes(var_names, new_values, type):
    length = len(var_names)
    s_se = str(type) + '$se'
    s_ce = str(type) + '$ce'
    len_se = 0
    len_ce = 0

    # 先统计'se_'个数
    for i in range(length):
        if var_names[i][0] == 's':
            len_se += 1

    len_ce = length - len_se
    # 封装参数个数
    s_se = s_se + '$' + str(len_se)
    s_ce = s_ce + '$' + str(len_ce)

    # 封装参数名和对应设置值
    for j in range(length):
        if j < len_se:
            s_se = s_se + '$' + var_names[j][3:] + '$' + str(new_values[j])
        else:
            s_ce = s_ce + '$' + var_names[j][3:] + '$' + str(new_values[j])
    print(s_se)
    print(s_ce)
    return s_se, s_ce

# 将action放大为可应用的值
def action_change(env, action):
    action_len = len(action)
    real_action = [0] * action_len
    for k in range(action_len):
        # real_action[k] = int(env.max_info[k] * action[k])
        real_action[k] = action_mapping(action[k], env.min_info[k], env.max_info[k])

    # 保证buf_pool_size按照chunk_size的整数倍进行更新
    # real_action[0] = real_action[0] // globalValue.CHUNK_SIZE * globalValue.CHUNK_SIZE

    real_action = list(map(int, real_action))
    #real_action[0] = int(real_action[0])
    return real_action

def get_default_actions(env):
    default_actions = []
    actions = []
    for info in env.all_actions.keys():
        scales = env.all_actions[info]
        default_actions.append(scales[0])
        actions.append(real_action_to_action(scales[0], scales[1], scales[2]))
    # print('default_actions = ', default_actions)
    env.all_last_action = actions
    return default_actions

# 依节点信息划分并计算每个节点的实际动作real_action[][]
def split_action(env, action):
    real_actions = []
    action_ = []
    last_uuid = 0
    cnt = 0
    # all_actions格式: 'uuid#innodb_buffer_pool:[default, min, max]'
    for info in env.all_actions.keys():
        # print(info)
        uuid = info.split("#")[0]
        scales = env.all_actions[info]
        # print(scales)
        real_action = action_mapping(action[cnt], scales[1], scales[2])
        if last_uuid != uuid and action_ != []:
            action_ = list(map(int, action_))
            real_actions.append(action_)
            action_ = []
        action_.append(real_action)
        cnt += 1
        last_uuid = uuid
    action_ = list(map(int, action_))
    real_actions.append(action_)
    # real_actions = list(map(int, real_actions))
    return real_actions


def split_default_actions(env):
    real_actions = []
    action = get_default_actions(env)
    action_ = []
    last_uuid = 0
    cnt = 0
    # all_actions格式: 'uuid#innodb_buffer_pool:[default, min, max]'
    for info in env.all_actions.keys():
        uuid = info.split("#")[0]
        # print('uuid = ', uuid)
        real_action = action[cnt]
        if last_uuid != uuid and action_ != []:
            real_actions.append(action_)
            action_ = []
        action_.append(real_action)
        cnt += 1
        last_uuid = uuid
    real_actions.append(action_)
    # print('real_actions = ', real_actions)
    return real_actions

# 将real action转为action值
def real_action_to_action(real_action,low,high):
    action = -1 + (real_action - low) * (2.0 / (high - low))
    action = np.clip(action, -1, 1)
    return action

# 将专家推荐趋势解析并与agent_predict合成
def action_with_knowledge(action, action_trend, p, last_action):
    action_len = len(action)
    for k in range(action_len):
        trend = action_trend[k]
        # 有概率p选择遵守trend
        rand = np.random.uniform(0, 1)
        # print('rand = ', rand)
        # 随机生成的数落在[0,p]区间内即为遵守trend
        if rand <= p:
            if trend == 1:
                action[k] = np.random.uniform(last_action[k], 1)
            elif trend == -1:
                action[k] = np.random.uniform(-1, last_action[k])

    return action

# 将best_now、专家推荐趋势解析并与agent_predict合成
def action_with_knowledge_and_best_now(action, best_action_now, action_trend, p_best, p_exp, last_action):
    action_len = len(action)
    hit_cnt = 0
    # 概率p_best选择best_action_now, 概率p_exp选择遵守trend,
    rand = np.random.uniform(0, 1)
    for k in range(action_len):
        if action_trend == []:
            trend = 0
        else:
            trend = action_trend[k]
        # 随机生成的数落在[0,p]区间内即为遵守trend
        if trend == 1 and action[k] > last_action[k]:
            hit_cnt += 1
        elif trend == -1 and action[k] < last_action[k]:
            hit_cnt += 1
        elif trend != 0:
            hit_cnt -= 1

        if 0 < rand <= p_exp:
            if trend == 1 and action[k] < last_action[k]:
                action[k] = np.random.uniform(last_action[k], 1)
            if trend == -1 and action[k] > last_action[k]:
                action[k] = np.random.uniform(-1, last_action[k])
        elif rand <= p_best + p_exp:
            action[k] = best_action_now[k]
    return action, hit_cnt

# if __name__ == '__main__':
#     action = -0.98
#     real_action = action_mapping(action, 33554432, 3355443200)
#     print("action=", action)
#     print("real_action=", real_action)
#     action = real_action_to_action(real_action, 33554432, 3355443200)
#     print("real_actionto_action=", action)

# 判断action是否符合应用条件
# 不符合，返回False
# 符合，返回True
def action_cirtic(env,action):
    for k in range(len(action)):
        if env.min_info[k] > action[k]:
            return False
    return True


################################################
#############  可使用mysql客户端   ###############
################################################
def create_conn(ip) -> object:

    #连接配置信息
    # conn_host='192.168.1.102'
    # conn_host='127.0.0.1'
    conn_host=ip
    conn_port=3306
    conn_user='root'
    conn_password='mysql'

    #建立连接
    conn = pymysql.connect(host=conn_host, port=conn_port, user=conn_user, password=conn_password)
    return conn


# 创立连接数据库的connection
# 与get_curr一起使用
# def create_conn() -> object:
#
#     #连接配置信息
#     # conn_host='192.168.1.102'
#     # conn_host='127.0.0.1'
#     conn_host=globalValue.CONNECT_CE_IP
#     conn_port=3306
#     conn_user='dawn'
#     conn_password='mysql'
#
#     #建立连接
#     conn = pymysql.connect(host=conn_host, port=conn_port, user=conn_user, password=conn_password)
#     return conn

# GRANT ALL PRIVILEGES ON *.* TO 'dawn'@'%' identified by 'mysql';
# GRANT SELECT ON performance_schema.* TO 'dawn'@'%';
# GRANT ALL PRIVILEGES ON *.* TO 'prom'@'localhost' identified by 'mysql';
# GRANT SELECT ON performance_schema.* TO 'prom'@'localhost';

# 获取数据库最新state
def get_current_status():
    if(len(globalValue.GLOBAL_CURRENT_STATUS) == 0):
        return None
    return globalValue.GLOBAL_CURRENT_STATUS

# 设置数据库最新状态
def set_curent_status(current_status):
    globalValue.GLOBAL_CURRENT_STATUS = current_status

# 建立数据库连接
def get_cur(conn):
    cur = conn.cursor()
    #print('Conntion created successfully!!!')
    return cur

#关闭数据库连接
def close_conn_mysql(cur, conn):
    cur.close()
    conn.close()
    #print('Conntion closed successfully!!!')


# 执行set语句，设置参数值
# 需建立数据库连接
# 建议输入为数组
# variables_status = [variable_name, variable_current_value, viriable_max_value, variable_min_value, variable_change_step]
# 目前是字典输入
def execute_sql(cur,knobs_dict):
    #print(knobs_dict)
    for variable_name, variable_value in knobs_dict.items():
        print('variable_name:',variable_name)
        print('variable_value:',variable_value)
        sql = (f'''
                set global {variable_name}={variable_value}
                ''')
        try:
            cur.execute(sql)
        except:
            print('error')
            time.sleep(1)


# 执行show语句，show variables like...
# 输入为单个变量
def show_sql(variable_name):
    conn = create_conn()
    cur = get_cur(conn)
    # 待执行语句
    sql = (f'''
            show variables like '{variable_name}'
            ''')
    cur.execute(sql)
    result = cur.fetchall()
    print('The current bp is:', result)
    close_conn_mysql(cur, conn)
    return result


# 获取变量当前值，show variables like ...
# 输入为数组，可一次性获取多个变量
# return result为数组
# [('innodb_buffer_pool_size', '134217728'),
# ('innodb_old_blocks_pct', '37'),
# ('innodb_old_blocks_time', '1000'),
# ('innodb_max_dirty_pages_pct_lwm', '0.000000'),
# ('innodb_flush_neighbors', '1'),
# ('innodb_lru_scan_depth', '1024')]
def show_variables(cur,variable_names):
    result = []
    for variable_name in variable_names:
        temp_result = show_sql(cur,variable_name)
        result.append(temp_result[0])
    return result


# 获取状态变量当前值,show global status like...
# 输入为单个状态变量
# [questions]用来计算reward
def show_status(cur,status_name):
    # 待执行语句
    sql = (f'''
            show global status like '{status_name}'
            ''')
    cur.execute(sql)
    print("check executed?", cur._check_executed())
    result = cur.fetchall()
    print("execute '" + sql + "' result = ")
    idx = 0
    for r in result:
        idx += 1
        print(str(idx) + ": " + str(r))
    sys.stdout.flush()
    return result


# [questions]用来计算reward
def prepare_for_tpcc(ip):
    conn = create_conn(ip)
    cur = get_cur(conn)
    # # 待执行语句
    sql1 = (f'''
            CREATE DATABASE tpcc;
            ''')
    # sql2 = (f'''
    #         source /home/wjx/tpcc-mysql/create_table.sql;
    #             ''')
    # # sql3 = 'create_table.sql;'
    # # sql4 = 'sorce /home/wjx/tpcc-mysql/add_fkey_idx.sql;
    cur.execute(sql1)
    # cur.execute(sql2)
    # cur.execute(sql3)
    # cur.execute(sql4)

    close_conn_mysql(cur, conn)
    return

# 获取缓冲池大小
def get_bps(ip, port):
    bps, flag = send_msg_to_server('4', ip, port)
    print("[{0}:{1}] bps = {2}".format(ip, port, bps))
    bps = int(bps.split("$")[0])
    return bps


# 计算hit_ratio
def get_se_hr(node):
    ip = node.ip
    port = node.port
    # 获取当前页面总数
    pages_info_before, flag = send_msg_to_server('2', ip, port)
    # 计算5秒的hit_ratio
    time.sleep(5)
    # 获取当前页面总数
    pages_info_after, flag = send_msg_to_server('2',ip,port)

    hr = cal_hr(pages_info_before, pages_info_after)

    return hr

# 使用sql语句计算hit_ratio
def get_hr():
    conn = create_conn()
    cur = get_cur(conn)
    p1 = show_status(cur, 'innodb_buffer_pool_reads')
    # time.sleep(10)
    p2 = show_status(cur, 'innodb_buffer_pool_read_requests')
    p = 1 - (int(p1[0][1]))/(int(p2[0][1]))
    # print('The current hit_ratio is:',p)
    close_conn_mysql(cur, conn)
    return p

# 使用sql语句计算hit_ratio
def get_node_hr(node):
    conn = create_conn(node.ip)
    cur = get_cur(conn)
    p1 = show_status(cur, 'innodb_buffer_pool_reads')
    # time.sleep(10)
    p2 = show_status(cur, 'innodb_buffer_pool_read_requests')
    p = 1 - (int(p1[0][1]))/(int(p2[0][1]))
    # print('The current hit_ratio is:',p)
    close_conn_mysql(cur, conn)
    return p

def cal_hr(pages_info_before, pages_info_after):
    pages1 = pages_info_before.split("$")
    instances = int(pages1[0])
    len = 2*instances+1

    pages1 = list(map(int, pages1[0:len]))
    pages2 = list(map(int, pages_info_after.split("$")[0:len]))

    total_pages = 0.0
    read_pages = 0.0

    for i in range(instances):
        # print("i=",i)
        total_pages = pages2[i+1] - pages1[i+1] + total_pages
        # print("i+1+instances = ", i+1+instances)
        read_pages = pages2[i+1+instances] - pages1[i+1+instances] + read_pages

    if total_pages == 0:
        hit_ratio = -1
    else:
        hit_ratio = (total_pages - read_pages) / total_pages

    return hit_ratio




# 计算qps --- calculate average qps
# 10s
def get_qps():
    conn = create_conn()
    cur = get_cur(conn)
    p1 = show_status(cur, 'questions')
    time.sleep(10)
    p2 = show_status(cur, 'questions')
    time.sleep(10)
    p3 = show_status(cur, 'questions')
    p_1 = (int(p2[0][1]) - int(p1[0][1])) / 10.0
    p_2 = (int(p3[0][1]) - int(p2[0][1])) / 10.0
    p = (p_1 + p_2) / 2.0
    # print('The current qps is:',p)
    close_conn_mysql(cur, conn)
    return p

# 计算qps--- calculate average qps
# 10s
def get_node_qps(node):
    conn = create_conn(node.ip)
    cur = get_cur(conn)
    p1 = show_status(cur, 'questions')
    time.sleep(10)
    p2 = show_status(cur, 'questions')
    time.sleep(10)
    p3 = show_status(cur, 'questions')
    p_1 = (int(p2[0][1]) - int(p1[0][1])) / 10.0
    p_2 = (int(p3[0][1]) - int(p2[0][1])) / 10.0
    p = (p_1 + p_2) / 2.0
    # print('The current qps is:',p)
    close_conn_mysql(cur, conn)
    return p


def cal_r(delta_t0, delta_t1):
    if delta_t0 > 0:
        r = abs(1 + delta_t1) * (pow(1 + delta_t0, 2) - 1)
    else:
        r = (-1) * abs(1 - delta_t1) * (pow(1 - delta_t0, 2) - 1)
    if r > 0 and delta_t1 < 0:
        r = 0
    return r

def cal_reward_ce_single_node(node, h_before , h_after , q_before , q_after , action_after):
    # 先计算delta(t-t0)和delta(t-t-1)
    delta_h0 = (h_after - node.hit_t0) / node.hit_t0
    delta_q0 = (q_after - node.qps_t0) / node.qps_t0
    delta_ht = (h_after - h_before) / h_before
    delta_qt = (q_after - q_before) / q_before
    node.delta_h0 = delta_h0
    node.delta_ht = delta_ht
    node.delta_q0 = delta_q0
    node.delta_qt = delta_qt

    # print('h_before ', h_before)
    # print('h_after ', h_after)
    # print('qps_before ', q_before)
    # print('qps_after ', q_after)

    # 计算qps和hit_ratio对应的r(以过程为主导）
    # if delta_h0 > 0:
    #     rewards_h = abs(1 + delta_ht) * (pow(1 + delta_h0, 2) - 1)
    # else:
    #     rewards_h = (-1) * abs(1 - delta_h0) * (pow(1 - delta_ht, 2) - 1)
    rewards_h = cal_r(delta_h0, delta_ht)
    # if rewards_h > 0 and delta_h0 < 0:
    #     rewards_h = 0
    # if delta_q0 > 0:
    #     rewards_q = abs(1 + delta_q0) * (pow(1 + delta_qt, 2) - 1)
    # else:
    #     rewards_q = (-1) * abs(1 - delta_q0) * (pow(1 - delta_qt, 2) - 1)
    rewards_q = cal_r(delta_q0, delta_qt)
    # if rewards_q > 0 and delta_q0 < 0:
    #     rewards_q = 0

    ## 增加缓冲池资源的考虑，action的正负并不能够直接反映缓冲池调大还是调小
    bp_size_after = action_after
    bp_size_before = node.bpsize_before
    delta_b0 = (bp_size_after - node.bp_size_0) / node.bp_size_0
    delta_bt = (bp_size_after - bp_size_before) / bp_size_before
    # print('bps_before ', bp_size_before)
    # print('bps_after ', bp_size_after)
    # if delta_bt > 0:
    #     rewards_b = abs(1+delta_b0) * (pow(1+delta_bt, 2) - 1)
    # else:
    #     rewards_b = (-1) * abs(1-delta_b0) * (pow(1-delta_bt, 2) - 1)
    # if delta_b0 <= 0 and delta_q0 >= 0 and delta_h0 >= 0:
    #     rewards_b = 3
    # else:
    #     rewards_b = -2

    return rewards_q, rewards_h, delta_q0

    # 设置权重
    # wh = 0.2
    # wq = 0.5
    # wb = 0.3
    # wh = 0.1
    # wq = 0.6
    # wb = 0.3
    wh = 0.15
    wq = (1-wh) / 2.0
    wb = wq

    # 计算实际的奖励
    reward = rewards_h * wh + rewards_q * wq + rewards_b * wb
    if reward < -1:
        return -1
    elif reward > 1:
        return 1
    else:
        return reward

def cal_reward_ce_my(env, h_before , h_after , q_before , q_after , action_after, min_info, max_info):
    # 先计算delta(t-t0)和delta(t-t-1)
    delta_h0 = (h_after - env.hit_t0)/env.hit_t0
    delta_q0 = (q_after - env.qps_t0)/env.qps_t0
    delta_ht = (h_after - h_before)/h_before
    delta_qt = (q_after - q_before)/q_before

    # print('h_before ', h_before)
    # print('h_after ', h_after)
    # print('qps_before ', q_before)
    # print('qps_after ', q_after)

    #计算qps和hit_ratio对应的r(以过程为主导）
    # if delta_ht > 0:
    #     rewards_h = abs(1+delta_h0) * (pow(1+delta_ht, 2) - 1)
    # else:
    #     rewards_h = (-1) * abs(1-delta_h0) * (pow(1-delta_ht, 2) - 1)
    #
    # if delta_qt > 0:
    #     rewards_q = abs(1+delta_q0) * (pow(1+delta_qt, 2) - 1)
    # else:
    #     rewards_q = (-1) * abs(1-delta_q0) * (pow(1-delta_qt, 2) - 1)

    if delta_h0 > 0:
        rewards_h = abs(1+delta_ht) * (pow(1+delta_h0, 2) - 1)
    else:
        rewards_h = (-1) * abs(1-delta_ht) * (pow(1-delta_h0, 2) - 1)

    if delta_q0 > 0:
        rewards_q = abs(1+delta_qt) * (pow(1+delta_q0, 2) - 1)
    else:
        rewards_q = (-1) * abs(1-delta_qt) * (pow(1-delta_q0, 2) - 1)


    # if rewards_q > 0 and delta_q0 < 0:
    #     rewards_q = 0

    ## 增加缓冲池资源的考虑，action的正负并不能够直接反映缓冲池调大还是调小
    bp_size_after = action_mapping(action_after, min_info[0], max_info[0])
    bp_size_before = env.bpsize_before
    delta_b0 = (bp_size_after - env.bp_size_0) / env.bp_size_0
    delta_bt = (bp_size_after - bp_size_before) / bp_size_before
    # print('bps_before ', bp_size_before)
    # print('bps_after ', bp_size_after)
    # if delta_bt > 0:
    #     rewards_b = abs(1+delta_b0) * (pow(1+delta_bt, 2) - 1)
    # else:
    #     rewards_b = (-1) * abs(1-delta_b0) * (pow(1-delta_bt, 2) - 1)
    if delta_bt <= 0 and delta_b0 <= 0:
        rewards_b = 1
    else:
        rewards_b = -1

    #设置权重
    # wh = 0.2
    # wq = 0.5
    # wb = 0.3
    # wh = 0.1
    # wq = 0.6
    # wb = 0.3
    wh = 0.1
    wq = 0.6
    wb = 0.3

    #计算实际的奖励
    reward = rewards_h * wh + rewards_q * wq + rewards_b * wb
    if reward < -1:
        return -1
    elif reward > 1:
        return 1
    else:
        return reward

def cal_reward_se_single_node(node, h_before, h_after, action_after):
    # 先计算delta(t-t0)和delta(t-t-1)
    # print('node{} calr {} {} {}'.format(node.uuid, h_befor e, h_after, action_after))
    delta_h0 = (h_after - node.hit_t0) / node.hit_t0
    delta_ht = (h_after - h_before) / h_before
    node.delta_h0 = delta_h0
    node.delta_ht = delta_ht

    # print('h_before ', h_before)
    # print('h_after ', h_after)
    # print('qps_before ', q_before)
    # print('qps_after ', q_after)

    # 计算qps和hit_ratio对应的r(以过程为主导????）
    # if delta_ht > 0:
    #     rewards_h = abs(1 + delta_h0) * (pow(1 + delta_ht, 2) - 1)
    # else:
    #     rewards_h = (-1) * abs(1 - delta_h0) * (pow(1 - delta_ht, 2) - 1)

    # if delta_h0 > 0:
    #     rewards_h = abs(1 + delta_ht) * (pow(1 + delta_h0, 2) - 1)
    # else:
    #     rewards_h = (-1) * abs(1 - delta_ht) * (pow(1 - delta_h0, 2) - 1)
    rewards_h = cal_r(delta_h0, delta_ht)

    ## 增加缓冲池资源的考虑，action的正负并不能够直接反映缓冲池调大还是调小
    bp_size_after = action_after
    bp_size_before = node.bpsize_before
    delta_b0 = (bp_size_after - node.bp_size_0) / node.bp_size_0
    delta_bt = (bp_size_after - bp_size_before) / bp_size_before
    # print('bps_before ', bp_size_before)
    # print('bps_after ', bp_size_after)
    # if delta_bt > 0:
    #     rewards_b = abs(1+delta_b0) * (pow(1+delta_bt, 2) - 1)
    # else:
    #     rewards_b = (-1) * abs(1-delta_b0) * (pow(1-delta_bt, 2) - 1)
    # if delta_b0 <= 0 and delta_h0 >= 0:
    #     rewards_b = 1
    # else:
    #     rewards_b = -1

    return rewards_h

    # 设置权重
    wh = 0.4
    wb = 0.6

    # 计算实际的奖励
    reward = rewards_h * wh + rewards_b * wb
    if reward < -1:
        return -1
    elif reward > 1:
        return 1
    else:
        return reward

def cal_reward_se_my(env, h_before, h_after, action_after, min_info, max_info):
    # 先计算delta(t-t0)和delta(t-t-1)
    delta_h0 = (h_after - env.hit_t0)/env.hit_t0
    delta_ht = (h_after - h_before)/h_before

    # print('h_before ', h_before)
    # print('h_after ', h_after)
    # print('qps_before ', q_before)
    # print('qps_after ', q_after)

    #计算qps和hit_ratio对应的r(以过程为主导）
    if delta_ht > 0:
        rewards_h = abs(1+delta_h0) * (pow(1+delta_ht, 2) - 1)
    else:
        rewards_h = (-1) * abs(1-delta_h0) * (pow(1-delta_ht, 2) - 1)

    ## 增加缓冲池资源的考虑，action的正负并不能够直接反映缓冲池调大还是调小
    bp_size_after = action_mapping(action_after, min_info[0], max_info[0])
    bp_size_before = env.bpsize_before
    delta_b0 = (bp_size_after - env.bp_size_0) / env.bp_size_0
    delta_bt = (bp_size_after - bp_size_before) / bp_size_before
    # print('bps_before ', bp_size_before)
    # print('bps_after ', bp_size_after)
    # if delta_bt > 0:
    #     rewards_b = abs(1+delta_b0) * (pow(1+delta_bt, 2) - 1)
    # else:
    #     rewards_b = (-1) * abs(1-delta_b0) * (pow(1-delta_bt, 2) - 1)
    if delta_bt <= 0 and delta_b0 <= 0:
        rewards_b = 1
    else:
        rewards_b = -1

    #设置权重
    wh = 0.5
    wb = 0.5

    #计算实际的奖励
    reward = rewards_h * wh + rewards_b * wb
    if reward < -1:
        return -1
    elif reward > 1:
        return 1
    else:
        return reward

def cal_reward_ce_1(env, h_before , h_after , q_before , q_after):
    # 先计算delta(t-t0)和delta(t-t-1)
    delta_h0 = (h_after - env.hit_t0)/env.hit_t0
    delta_q0 = (q_after - env.qps_t0)/env.qps_t0
    delta_ht = (h_after - h_before)/h_before
    delta_qt = (q_after - q_before)/q_before

    # print('delta_h0', delta_h0)
    # print('delta_ht', delta_ht)
    # print('delta_q0', delta_q0)
    # print('delta_qt', delta_qt)

    # rate0 = 0.02
    # rate1 = 0.08
    # #计算qps和hit_ratio对应的r(以过程为主导）
    # rewards_h = (rate0 * delta_h0) + (rate1 * delta_ht)
    # rewards_q = (rate0 * delta_q0) + (rate1 * delta_qt)

    # 计算qps和hit_ratio对应的r(以过程为主导）
    # TODO:计算一个指标的reward可以单独写一个函数cal_reward_from()
    if delta_ht > 0:
        rewards_h = abs(1 + delta_h0) * (pow(1 + delta_ht, 2) - 1)
    else:
        rewards_h = (-1) * abs(1 - delta_h0) * (pow(1 - delta_ht, 2) - 1)
    if rewards_h > 0 and delta_ht < 0:
        rewards_h = 0

    if delta_qt > 0:
        rewards_q = abs(1 + delta_q0) * (pow(1 + delta_qt, 2) - 1)
    else:
        rewards_q = (-1) * abs(1 - delta_q0) * (pow(1 - delta_qt, 2) - 1)

    if rewards_q > 0 and delta_qt < 0:
        rewars_q = 0

    # print('rewards_h', rewards_h)
    # print('rewards_q', rewards_q)
    # if rewards_q > 0 and delta_q0 < 0:
    #     rewards_q = 0

    ## 增加缓冲池资源的考虑


    #设置权重
    wh = 0
    wq = 1

    #计算实际的奖励
    reward = rewards_h * wh + rewards_q * wq
    return reward


def cal_reward_ce(env, h_before, h_after, action_before, action_after):
    # reward = 0
    # 先处理缓存命中率
    # h_before = round(h_before, 3)
    # h_after = round(h_after, 3)
    # delta = round((h_after - h_before), 2)
    # delta_bps = 0
    # if delta < 0:
    #     reward = 0
    # elif delta > 0:
    #     hps_r = delta * 1000
    #     reward = hps_r
    # else:
    #     delta_bps = b_after - b_before
    #
    #     if delta_bps >= 0:
    #         reward = 0
    #     else:
    #         reward = delta_bps / globalValue.MAX_POOL_SIZE
    #         reward = -reward * 100
    # print('delta_hit_ratio: ', delta, '    delta_bps: ', delta_bps, '    reward: ', reward)
    h_after = round(h_after, 2)

    delta_hit_ratio = h_after - env.max_hit_ratio
    #delta_bps = round((action_after - action_before), 2)
    #
    if delta_hit_ratio >= 0:
        env.max_hit_ratio = h_after
        #reward = abs(delta_bps)
    #    reward = 1
    #elif delta_hit_ratio == 0:
        #reward = -delta_bps
        reward = -abs(action_after + 1) / 2
    elif delta_hit_ratio >= -0.011:
        reward = -abs(action_after + 1) / 2
    else:
        #reward = -abs(delta_bps)
        reward = -1

    # h_after = round(h_after, 2)
    # if h_after > env.max_hit_ratio:
    #    env.max_hit_ratio = h_after
    # r = b_after / globalValue.MAX_POOL_SIZE
    # if(h_after == env.max_hit_ratio):
    #    reward = -r
    # else:
    #    reward = r
    #
    # reward = 10 * reward





    # hps_r = (round(h_after, 2) - 0.80) * 10
    # bps_r = (b_after / globalValue.MAX_POOL_SIZE) * 10

    # if h_after > env.max_hit_ratio:
    #     env.max_hit_ratio = h_after
    #     reward =
    # else:




    # reward = 0.8 * hps_r - 0.2 * bps_r
    #
    # if b_after == 31457280:
    #     reward = -1000
    #
    #
    # print('h_after = ', h_after, 'reward = ', reward)




    # if b_after > 838860800:
    #     reward = (1 - b_after /( globalValue.MAX_POOL_SIZE+100)) * 10
    # elif b_after < 209715200:
    #     reward = (b_after / globalValue.MAX_POOL_SIZE - 1) * 10
    # else:
    #     reward = (b_after / globalValue.MAX_POOL_SIZE) * 10



    # hps_r = (round(h_after, 2) - 0.80) * 10
    # # bps_r = (b_after / globalValue.MAX_POOL_SIZE) * 10
    # # reward = 0.8 * hps_r - 0.2 * bps_r
    # reward = hps_r

    #######################################################################################
    # 以下为只调整缓冲池大小可以获得推荐值的奖励函数  0414
    # if h_after == -1:
    #     return 0
    #
    # h_before = round(env.max_hit_ratio, 3)
    # h_after = round(h_after, 3)
    # delta = round((h_after - h_before), 2)
    # delta_bps=action_after-action_before
    #
    # if delta < 0:
    #    reward = -1
    # elif delta > 0:
    #    env.max_hit_ratio = h_after
    #    reward = 1
    # else:
    #    reward = 0
    #######################################################################################

    # if delta<0:
    #     reward=-1.0
    # elif delta>0:
    #     reward=1.0
    # else:
    #     if delta_bps>0:
    #         reward=-1.0
    #     elif delta_bps<0:
    #         reward=0.0
    #     else:
    #         reward=0.0

    # delta_bps = delta_bps = b_after - b_before
    # delta_bps = (b_after - b_before) / globalValue.MAX_POOL_SIZE
    # delta_action = action_after - action_before
    #if delta < 0:
    #    reward = -abs(delta) * 500
    #elif delta > 0:
    #    env.max_hit_ratio = h_after
        #     hps_r = delta * 1000
    #    reward = abs(delta) * 500
    #else:
    #    reward = -abs(action_after) * 50

    print('h_after = ', h_after, 'action_after = ', action_after, 'current_max_hit_ratio = ', env.max_hit_ratio, 'reward = ', reward)
    return reward




    # hps_r = delta * 10
    # bps_r = (b_after - b_before) / globalValue.MAX_POOL_SIZE
    #
    # reward = 1000 * (0.8 * hps_r - 0.2 * bps_r)
    #
    # print('delta = ', delta)

    # delta = q_after - q_before
    # if abs(delta) <= 50:
    #     qps_r = 0
    # else:
    #     qps_r = delta / 1000.0
    #
    # bps_r = (b_after - b_before) / globalValue.MAX_POOL_SIZE
    #
    # reward = 0.8 * qps_r - 0.2 * bps_r

    # if value2 > value1:
    #    qps_r = 1
    # elif value2 == value1:
    #    qps_r = 0
    # else:
    #    qps_r = -1


    # qps_r = value2 - value1
    #
    # bps_r = value3 / globalValue.MAX_POOL_SIZE * 50
    #
    # if abs(qps_r) < 50:
    #     qps_r = 0
    #
    # reward = qps_r - bps_r

    # print('hit_ratio reward = ', hps_r, 'bps reward = ', bps_r, 'reward = ', reward)
    #
    # return reward


def cal_reward_se(value1, value2, value3):
    if value2 > value1:
        qps_r = 1
    elif value2 == value1:
        qps_r = 0
    else:
        qps_r = -1

    bps_r = value3 / globalValue.MAX_POOL_SIZE

    reward = qps_r - bps_r

    return reward

def load_bash_remote(type):
    # load 80s
    # print('LOAD BASH++++')
    timestp = time_to_str(get_timestamp())
    file_name = globalValue.TEST_RES_FILE + 'res' + timestp + '.txt'
    # make_file_cmd = 'touch ' + file_name
    to_file = '> %s 2>%s &' % (file_name, file_name)
    if type == 'sysbench':
        cmd = globalValue.LOAD_BASH
        # print(cmd)
        flag = list()
        for i in range(globalValue.CE_LOAD_NUM):
            tmp_flag = sshExe(globalValue.CONNECT_CE_IP[i], globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, cmd)
            flag = flag.append(tmp_flag)
    elif type == 'tpcc':
        cmd = globalValue.LOAD_TPCC
        # prepare_for_tpcc(globalValue.CONNECT_CE_IP[0])
        # print(cmd)
        flag = sshExe(globalValue.CONNECT_CE_IP[0], globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, cmd)
    return flag

def start_for_a_fresh_node(node_start_cmd, new_data_cmd):
    # sleep
    sleep_cmd = 'sleep 1;'
    # newdata
    newdata = new_data_cmd + ';'
    # start with exact string
    start_node = node_start_cmd
    text = sleep_cmd + newdata + start_node
    return text

def start_for_a_not_fresh_node(node_close_cmd, node_start_cmd):
    # sleep
    sleep_cmd = 'sleep 1;'
    # close
    close_node = node_close_cmd + ';'
    # start with exact string
    start_node = node_start_cmd
    text = close_node + sleep_cmd + start_node
    return text

def load_bash(bash_time):
    f = globalValue.TMP_FILE
    if not os.path.exists(f):
        os.mkdir(f)
    f += 'sysbench_result_{}.log'.format(int(time.time()))
    cmd = 'touch ' + f
    os.system(cmd)

    test_f = globalValue.TMP_FILE + 'test.sh'
    cmd = 'touch ' + test_f
    os.system(cmd)
    buffer_time = 30
    bash_time = bash_time + buffer_time

    s = '# !/bin/bash\n'\
        'nohup '\
        '/usr/local/bin/sysbench /usr/local/share/sysbench/oltp_read_only.lua ' \
        '--mysql-user=dawn --mysql-password=mysql ' \
        '--mysql-host=%s ' \
        '--mysql-port=3306 ' \
        '--mysql-db=test --mysql-storage-engine=innodb ' \
        '--table-size=10000 ' \
        '--tables=100 ' \
        '--threads=32 ' \
        '--events=0 ' \
        '--report-interval=10 ' \
        '--range_selects=off ' \
        '--time=%d run ' \
        '> %s ' \
        '2>%s &' \
        % (globalValue.CONNECT_MA_IP, bash_time, f, f)
    with open(test_f, 'w') as fs:
        fs.write(s)

    # print('bash_time ', bash_time)

    # 记录训练开始时间
    # globalValue.EPISODE_START_TIME = time.time()
    os.system('sh ' + test_f)
    print('bash_end ', f)
    return f, buffer_time + 10



def parse_tpcc(file_path):
    with open(file_path) as f:
        lines = f.read()
    temporal_pattern = re.compile(".*?trx: (\d+.\d+), 95%: (\d+.\d+), 99%: (\d+.\d+), max_rt:.*?")
    temporal = temporal_pattern.findall(lines)
    tps = 0
    latency = 0
    qps = 0

    for i in temporal[-10:]:
        tps += float(i[0])
        latency += float(i[2])
    num_samples = len(temporal[-10:])
    tps /= num_samples
    latency /= num_samples
    # interval
    tps /= 1
    return [tps, latency, qps]

def parse_sysbench(file_path):
    with open(file_path) as f:
        lines = f.read()
    temporal_pattern = re.compile(
        "tps: (\d+.\d+) qps: (\d+.\d+) \(r/w/o: (\d+.\d+)/(\d+.\d+)/(\d+.\d+)\)" 
        " lat \(ms,95%\): (\d+.\d+) err/s: (\d+.\d+) reconn/s: (\d+.\d+)")
    temporal = temporal_pattern.findall(lines)
    tps = 0
    latency = 0
    qps = 0

    for i in temporal[-10:]:
        tps += float(i[0])
        latency += float(i[5])
        qps += float(i[1])
    num_samples = len(temporal[-10:])
    tps /= num_samples
    qps /= num_samples
    latency /= num_samples
    return [tps, latency, qps]

def record_best(qps, hit_ratio, bps):
    filename = 'bestnow.log'
    best_flag = False
    if os.path.exists(globalValue.BEST_FILE + filename):
        qps_best = qps
        hit_ratio_best = hit_ratio
        bps_best = bps
        if hit_ratio_best != 0 and qps_best != 0:
            # with open(globalValue.BEST_FILE + filename) as f:
            #     lines = f.readlines()
            # best_now = lines[0].split(',')
            # if qps_best >= float(best_now[0]) and hit_ratio_best >= float(best_now[1]) and bps_best <= float(best_now[2]):
            if globalValue.REWARD_NOW >= globalValue.MAX_REWARD:
                best_flag = True
                with open(globalValue.BEST_FILE + filename, 'w') as f:
                    f.write(str(qps_best) + ',' + str(hit_ratio_best) + ',' + str(bps_best))
    else:
        with open(globalValue.BEST_FILE + filename, 'w') as file:
            qps_best = qps
            hit_ratio_best = hit_ratio
            bps_best = bps
            file.write(str(qps_best) + ',' + str(hit_ratio_best) + ',' + str(bps_best))
            best_flag = True
    return best_flag

def record_best_nodes(q_after_ce, h_after_ce, bps_ce, h_after_se, bps_se):
    filename = 'bestnow.log'
    best_flag = False
    if os.path.exists(globalValue.BEST_FILE + filename):
        if q_after_ce != 0:
            if globalValue.REWARD_NOW >= globalValue.MAX_REWARD:
                best_flag = True
                with open(globalValue.BEST_FILE + filename, 'w') as f:
                    f.write(str(q_after_ce) + ',' + str(h_after_ce) + ',' + str(bps_ce) + ',' + str(h_after_se) + ',' + str(bps_se))
    else:
        with open(globalValue.BEST_FILE + filename, 'w') as file:
            file.write(str(q_after_ce) + ',' + str(h_after_ce) + ',' + str(bps_ce) + ',' + str(h_after_se) + ',' + str(bps_se))
            best_flag = True
    return best_flag

def record_all_best(ses, ces):
    filename = 'bestnow.log'
    best_flag = False
    if os.path.exists(globalValue.BEST_FILE + filename):
        if ces[0].last_qps != 0 and globalValue.REWARD_NOW >= globalValue.MAX_REWARD:
            best_flag = True
            with open(globalValue.BEST_FILE + filename, 'w') as f:
                for ce in ces:
                    if ce.is_primary == True:
                        f.write(
                            str(ce.last_qps) + ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                    else:
                        f.write(
                            ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                for se in ses:
                    f.write(
                        ',' + str(se.last_hr) + ',' + str(se.last_bp_size))
    else:
        with open(globalValue.BEST_FILE + filename, 'w') as f:
            best_flag = True
            for ce in ces:
                if ce.is_primary == True:
                    f.write(
                        str(ce.last_qps) + ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                else:
                    f.write(
                        ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
            for se in ses:
                f.write(
                    ',' + str(se.last_hr) + ',' + str(se.last_bp_size))
    return best_flag

def get_best_now(filename):
    with open(globalValue.BEST_FILE + filename) as f:
        lines = f.readlines()
    best_now = lines[0].split(',')
    return [float(best_now[0]), float(best_now[1]), float(best_now[2])]

def get_best_now_nodes(filename):
    with open(globalValue.BEST_FILE + filename) as f:
        lines = f.readlines()
    best_now = lines[0].split(',')
    return [float(best_now[0]), float(best_now[1]), float(best_now[2]), float(best_now[3]), float(best_now[4])]

def get_best_now_all_nodes(filename):
    with open(globalValue.BEST_FILE + filename) as f:
        lines = f.readlines()
    best_now = lines[0].split(',')
    return best_now

def handle_csv(src, dest, num):
    cnt = 0
    rewards = list()
    with open(src, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        reward_all = 0
        for row in rows:
            cnt += 1
            if cnt == 1:
                continue
            # reward_all += float(row[num])
            rewards.append(float(row[num]))
        # reward_all = reward_all / cnt
        rewards = sorted(rewards, key = float)
        print(rewards)
        print(len(rewards))
        reward_all = rewards[len(rewards) // 2]
        max_val = rewards[len(rewards) - 1]
        min_val = rewards[0]
        print(min_val)
        print(max_val)
        mid_1 = (max_val-min_val) / 3.0 + min_val
        mid_2 = (max_val-min_val) / 3.0 + mid_1
        print(mid_1)
        print(mid_2)
        cnt = 0
        for row in rows:
            cnt += 1
            if cnt == 1:
                continue
            reward = float(row[num])
            if reward > mid_2:
                reward = 2
            elif reward > mid_1:
                reward = 1
            else:
                reward = 0
            rows[cnt-1][num] = reward

    with open(dest, 'w', encoding='utf-8') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerows(rows)

def get_timestamp():
    """
    获取UNIX时间戳
    """
    return int(time.time())

def time_to_str(timestamp):
    """
    将时间戳转换成[YYYY-MM-DD HH:mm:ss]格式
    """
    return datetime.datetime.\
        fromtimestamp(timestamp).strftime("%Y-%m-%d_%H:%M:%S")

# Logger utils
class Logger:

    def __init__(self, name, log_file=''):
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        self.logger.addHandler(sh)
        self.start_time = get_timestamp()
        self.end_time = get_timestamp()
        print('LOG_FILE :', log_file)
        with open(self.log_file, 'w+') as f:
            f.write('=====log============' + '\n')

        if len(log_file) > 0:
            self.log2file = True
        else:
            self.log2file = False

    def _write_file(self, msg):
        if self.log2file:
            with open(self.log_file, 'a+') as f:
                f.write(msg + '\n')

    def get_timestr(self):
        timestamp = get_timestamp()
        date_str = time_to_str(timestamp)
        return date_str

    def warn(self, msg):
        msg = "%s[WARN] %s" % (self.get_timestr(), msg)
        self.logger.warning(msg)
        self._write_file(msg)

    def info(self, msg):
        msg = "%s[INFO] %s" % (self.get_timestr(), msg)
        #self.logger.info(msg)
        self._write_file(msg)

    def error(self, msg):
        msg = "%s[ERROR] %s" % (self.get_timestr(), msg)
        self.logger.error(msg)
        self._write_file(msg)

################################################
#############       测试部分      ###############
###############################################


# from maEnv.env import SEEnv
#
if __name__ == '__main__':
    for i in range(10):
        rand = np.random.uniform(0, 1)
        print(rand)

    ########################数据库连接测试###############################
    #conn = create_conn()
    #cur = get_cur(conn)

    # 执行语句
    #variables = ['innodb_buffer_pool_size', 'innodb_old_blocks_pct', 'innodb_old_blocks_time',
    #             'innodb_max_dirty_pages_pct_lwm', 'innodb_flush_neighbors', 'innodb_lru_scan_depth']
    #result = show_variables(cur,variables)
    #print(result)

    #close_conn_mysql(cur,conn)
    # variables = ['innodb_buffer_pool_size','innodb_old_blocks_pct','innodb_old_blocks_time','innodb_max_dirty_pages_pct_lwm','innodb_flush_neighbors','innodb_lru_scan_depth']
    #name = "innodb_buffer_pool_size"
    #value=134217728
    #conn_mysql(name,value)

    ########################数据封装测试###############################
    # new_variables = [1.0,2.0,4,3,10000]
    # set_variables_by_tune(new_variables)
    # print(s)
