import globalValue
from maEnv import utils
# 数据处理

# 人类先验知识控制器
# 数据库状态转化为标签
def status_to_labels(node):
    data_list = node.state_now
    
    # you can customise your own label identify rules here
    if data_list[10] < data_list[12]:
        node.labelA = 0
    else:
        node.labelA = 1

    
def node_labels_to_action_trend(node, info, action_trend, index):
    
    if node.labelA == 0:
        if info == 'old_blocks_time':
            action_trend[index] = -1
        elif info == 'random_read_ahead':
            action_trend[index] = -1
        elif info == 'read_ahead_threshold':
            action_trend[index] = 1
    
    # You can customise your label's rules here

def all_nodes_labels_to_action_trend(env):
    action_len = env.action_dim
    action_trend = [0] * action_len
    # print(action_trend)
    cnt = 0
    for se in env.se_info:
        for key in se.tune_action.keys():
            node_labels_to_action_trend(se, key, action_trend, cnt)
            # if globalValue.MAX_REWARD <= 0 and key == 'buffer_pool_size':
            #     action_trend[cnt] = -1
            cnt += 1

    for ce in env.ce_info:
        for key in ce.tune_action.keys():
            node_labels_to_action_trend(ce, key, action_trend, cnt)
            # if globalValue.MAX_REWARD <= 0 and key == 'buffer_pool_size':
            #     action_trend[cnt] = -1
            cnt += 1

    return action_trend

def divide_node_info(action_info):
    # 按照前缀划分SE和CE节点发送的参数name
    length = len(action_info)
    list_se = []
    list_ce = []
    len_se = 0
    len_ce = 0

    # 先统计'se_'个数
    for i in range(length):
        if action_info[i][0] == 's':
            len_se += 1

    len_ce = length - len_se

    # 封装参数名和对应设置值
    for j in range(length):
        if j < len_se:
            list_se.append(action_info[j][3:])
        else:
            list_ce.append(action_info[j][3:])
    print(list_se)
    print(list_ce)
    return list_se, list_ce

# 标签生成动作趋势
# TODO：需要增加考虑SE端
def labels_to_action_trend(action_info, env):
    action_len = len(action_info)
    action_trend = [0] * action_len
    cnt = 0
    for se in env.se_info:
        node_labels_to_action_trend(action_info, se, action_trend, cnt)
        cnt += 1
    for ce in env.ce_info:
        node_labels_to_action_trend(action_info, ce, action_trend, cnt)
        cnt += 1
    return action_trend


# 对接收到的数据进行处理

# 监控参数的格式
#5[bp_size,                free_size,             lru_size,            old_lru_size,                    flush_size,
#4 wait_read_pages,         wait_write_lru,        wait_write_flush_,   wait_write_single
#4 made_young_pages,        made_young_not_pages,  youngs_per_second,   non_young_per_second
#5 read_pages_num,          create_pages_num,      write_pages_num,     read_pages_rate(d),    read_pages_per_second,
#4 create_pages_per_second, young_make_rate,       not_young_make_rate, pages_evictied_without_access]
#1 bp_page_hit_rate
# eg: [319, 0, 312, 0, 124, 55, 0, 2, 1, 0, 0, 0, 0, 137884, 471, 44536, 758.0, 2.0, 279.81, 0.0, 0.0, 0.0952381, 810.0]

def GetState(data):
    # 对接收到的数据进行处理
    data_list = data.split("$")
    # globalValue.GLOBAL_BUFFER_POOL_SIZE = (data_list[0])
    # data_list1 = data_list[0:10]
    # data_list2 = data_list[10:18]
    # data_list1 = list(map(int,data_list1))
    # data_list2 = list(map(float,data_list2))
    # data_list1.extend(data_list2)

    data_list1 = data_list[0:4]
    data_list1 = list(map(int, data_list1))
    # print('data_list : ', data_list1)

    max_pool_len = globalValue.GLOBAL_BUFFER_POOL_SIZE // 16 // 1024
    data_list1[0] = round(data_list1[0] / max_pool_len, 8)
    data_list1[1] = round(data_list1[1] / max_pool_len, 8)
    data_list1[2] = round(data_list1[2] / max_pool_len, 8)
    data_list1[3] = round(data_list1[3] / max_pool_len, 8)

    # data_list1[4] = round(data_list1[4] / globalValue.MAX_POOL_LEN, 4)
    # data_list1[5] = round(data_list1[5] / globalValue.MAX_POOL_LEN, 4)
    # data_list1[6] = round(data_list1[6] / globalValue.MAX_POOL_LEN, 4)
    #
    # data_list1[7] = data_list1[7] / 10000
    # data_list1[8] = data_list1[8] / 1000000
    # data_list1[9] = round(data_list1[9] / globalValue.MAX_POOL_LEN, 4)
    #
    # data_list1[10] = round(data_list1[10] / 10000, 4)
    # data_list1[11] = round(data_list1[11] / 10000, 4)
    #
    # data_list1[12] = round(data_list1[12] / globalValue.MAX_POOL_LEN, 4)
    # data_list1[13] = round(data_list1[13] / globalValue.MAX_POOL_LEN, 4)
    # data_list1[14] = round(data_list1[14] / globalValue.MAX_POOL_LEN, 4)
    # data_list1[15] = data_list1[15] / 1000
    # data_list1[16] = data_list1[16] / 1000
    # data_list1[17] = round(data_list1[17] / globalValue.MAX_POOL_LEN, 4)

    # 设置数据库最新状态
    utils.set_curent_status(data_list1)
    return data_list1

def GetNodeState(data, nodes_info):
    # print(data)
    # 对接收到的数据进行处理
    if data == None:
        return []
    data_list = data.split("$")
    # globalValue.GLOBAL_BUFFER_POOL_SIZE = (data_list[0])
    data_list1 = data_list[0:10]
    data_list2 = data_list[10:14]
    data_list1 = list(map(int,data_list1))
    data_list2 = list(map(float,data_list2))
    data_list1.extend(data_list2)

    # data_list1 = data_list[0:17]
    # data_list1 = list(map(int, data_list1))
    # print('data_list : ', data_list1)
    # if nodes_info == 'se':
    #     max_pool_len = globalValue.GLOBAL_BUFFER_POOL_SIZE_SE // 16 // 1024
    # else:
    #     max_pool_len = globalValue.GLOBAL_BUFFER_POOL_SIZE_CE // 16 // 1024
    # print('max_pool_len_{} = {}, datalist = {} '.format(nodes_info, max_pool_len, data_list1))
    # data_list1[0] = round(data_list1[0] / max_pool_len, 8)
    # data_list1[1] = round(data_list1[1] / max_pool_len, 8)
    # data_list1[2] = round(data_list1[2] / max_pool_len, 8)
    # data_list1[3] = round(data_list1[3] / max_pool_len, 8)
    #
    # data_list1[4] = round(data_list1[4] / globalValue.MAX_POOL_LEN, 4)
    # data_list1[5] = round(data_list1[5] / globalValue.MAX_POOL_LEN, 4)
    # data_list1[6] = round(data_list1[6] / globalValue.MAX_POOL_LEN, 4)
    #
    # data_list1[7] = data_list1[7] / 10000
    # data_list1[8] = data_list1[8] / 1000000
    # data_list1[9] = round(data_list1[9] / globalValue.MAX_POOL_LEN, 4)
    #
    # data_list1[10] = round(data_list1[10] / 10000, 4)
    # data_list1[11] = round(data_list1[11] / 10000, 4)
    #
    # data_list1[12] = round(data_list1[12] / globalValue.MAX_POOL_LEN, 4)
    # data_list1[13] = round(data_list1[13] / globalValue.MAX_POOL_LEN, 4)
    # data_list1[14] = round(data_list1[14] / globalValue.MAX_POOL_LEN, 4)
    # data_list1[15] = data_list1[15] / 1000
    # data_list1[16] = data_list1[16] / 1000
    # data_list1[17] = round(data_list1[17] / globalValue.MAX_POOL_LEN, 4)

    # 设置数据库最新状态
    utils.set_curent_status(data_list1)
    return data_list1
