#globalValue.py
import threading

# 线程交互事件
# EVENT = threading.Event()

# 参数配置信息
GLOBAL_ACTION = None

# status状态数组
GLOBAL_STATUS = []

# status状态数组的维度
# GLOBAL_STATUS_LIMIT = 5

# 当前指向status的位置
# GLOBAL_STATUS_POSITION = 0

# 最新的status
GLOBAL_CURRENT_STATUS = []

#innodb_buffer_pool_size大小
GLOBAL_BUFFER_POOL_SIZE = -1
GLOBAL_BUFFER_POOL_SIZE_CE = []
GLOBAL_BUFFER_POOL_SIZE_SE = []

#free链表的长度
# GLOBAL_FREE_LEN = 0

# chunk_size大小
# CHUNK_SIZE = 33554432
CHUNK_SIZE = 134217728

MAX_POOL_SIZE = 3355443200
# MAX_POOL_SIZE = pow(2, 64) - 1

MAX_POOL_LEN = GLOBAL_BUFFER_POOL_SIZE // 16 // 1024


# 每个episode开始时间
EPISODE_START_TIME = 0.0


#唤醒压测线程的事件
LOAD_EVENT = None


# 给SE判断episode是否结束的标志
SE_FLAG = False

EVAL_TEST = False

#GLOBAL_FULL_TAG = False

MAX_QPS = 0

MAX_HIT_RATIO = 0

SSH_CNT = 0

#CONNECT_MA_IP = '172.16.56.233'
# CONNECT_MA_IP = '222.20.76.69'
CONNECT_SE_IP = ['222.20.74.211']
CONNECT_CE_IP = ['222.20.74.211']   # primary ip 必须在index=0处
SE_PORT = 4000
CE_PORT = 2000
# SSH_USERNAME = 'dawn'
# SSH_PASSWD = 'vayvay'
SSH_USERNAME = 'orange'
SSH_PASSWD = '123456'
ROOT_FILE = '/home/orange/MATune/csdb_tune'

TMP_FILE = '/Users/dawn/Desktop/tmp/'

BEST_FILE = './test_model/'

ACTIONS_REWARD_FILE = "./test_model/actions_reward.csv"

BUFFER_POOL_SIZE_FILE = "./test_model/buffer_pool_size.txt"

CRITIC_LOSS_FILE = "./test_model/critic_loss.txt"

EVAL_REWARD_CAL_FILE = "./test_model/eval_reward_cal.txt"

HIT_RATIO_FILE = "./test_model/hit_ratio.txt"

QPS_STORE_FILE = "./test_model/qps_store.txt"

SCORES_FILE = "./test_model/scores.txt"

TRAIN_REWARD_CAL_FILE = "./test_model/train_reward_cal.txt"

RPM_SRC = '/home/orange/MATune/ma_-tune_0130/test_model/actions_reward.csv'
RPM_DEST = '/home/orange/MATune/ma_-tune_0130/test_model/actions_reward_new.csv'

MYSQLD_OPEN_EXEC = 'nohup /home/wjx/csdb_buffer_tune/cmake-build-debug/sql/mysqld ' \
              '--datadir=/home/wjx/MA/MA/ScriptMA/ce_data/data ' \
              '--skip-grant-tables ' \
              '--max_prepared_stmt_count=100000 '

# MYSQLD_OPEN_EXEC_CE = 'nohup /home/dawn/csdb_buffer_tune/cmake-build-debug/sql/mysqld ' \
#                       '--datadir=/home/dawn/ma/ScriptMA/ce_data/data ' \
#                       '--seuser=root --sepassword=root ' \
#                       '--innodb_use_native_aio=0 --ce=on --skip-grant-tables '
MYSQLD_OPEN_EXEC_CE = 'nohup /home/orange/MATune/csdb_tune/csdb_buffer_tune/bld/sql/mysqld ' \
'--ce=on --datadir=/home/orange/MATune/csdb/ma/ScriptMA/ce_data/data '  \
'--seuser=root --sepassword=root '  \
'--innodb_use_native_aio=0 '  \
'--configpath=/home/orange/MATune/csdb_tune/csdb_buffer_tune/ma_se/ma_ce_config.json ' \
'--skip-grant-tables --max_prepared_stmt_count=1000000 --max_connections=5000 ' \
'--user=root '


MYSQLD_OPEN_EXEC_SE ='/home/orange/MATune/csdb_tune/csdb_buffer_tune/bld/sql/mysqld ' \
'--se=on --datadir=/home/orange/MATune/csdb/ma/ScriptMA/se_data/data ' \
'--seuser=root --sepassword=root ' \
'--configpath=/home/orange/MATune/csdb_tune/csdb_buffer_tune/ma_se/ma_se_config.json ' \
'--innodb_use_native_aio=0 --skip-grant-tables ' \
'--user=root '

# MYSQLD_OUTPUT = ' > /home/dawn/mysqld_output.txt 2>/home/dawn/mysqld_output.txt &'
MYSQLD_OUTPUT_CE = ' > /home/orange/MATune/csdb_tune/ce_output.txt 2>/home/orange/MATune/csdb_tune/ce_output.txt &'
MYSQLD_OUTPUT_SE = ' > /home/orange/MATune/csdb_tune/se_output.txt 2>/home/orange/MATune/csdb_tune/se_output.txt &'

# kill all mysqld
MYSQLD_CLOSE_EXEC = "ps aux|grep -v grep|grep mysqld|awk '{print $2}'|xargs kill -9"
MYSQLD_CE_CLOSE_EXEC = "ps aux|grep -v grep|grep mysqld|awk '$12 ~/ce/ {print $2}'|xargs kill -9"
MYSQLD_SE_CLOSE_EXEC = "ps aux|grep -v grep|grep mysqld|awk '$12 ~/se/ {print $2}'|xargs kill -9"

MYSQLD_CHECK = 'ps aux|grep -v grep|grep mysqld'

NEW_DATA_CMD = '/home/dawn/new_data.sh'
NEW_DATA_CE_CMD = '/home/orange/MATune/csdb_tune/new_data_ce.sh'
NEW_DATA_SE_CMD = '/home/orange/MATune/csdb_tune/new_data_se.sh'
NEW_DATA_CE_TPCC_CMD = '/home/orange/MATune/csdb_tune/new_data_ce_tpcc.sh'
NEW_DATA_SE_TPCC_CMD = '/home/orange/MATune/csdb_tune/new_data_se_tpcc.sh'

# LOAD_BASH = 'nohup /home/dawn/test.sh ' \
#             '> /home/dawn/test_output.txt 2>/home/dawn/test_output.txt &'

LOAD_BASH = 'nohup /home/orange/MATune/csdb_tune/test.sh ' \
            '> /home/orange/MATune/csdb_tune/test_output.txt 2>/home/orange/MATune/csdb_tune/test_output.txt &'

LOAD_TPCC = 'nohup /home/orange/MATune/csdb_tune/test_tpcc.sh ' \
            '> /home/orange/MATune/csdb_tune/test_output.txt 2>/home/orange/MATune/csdb_tune/test_output.txt &'

TEST_RES_FILE = '/home/orange/MATune/csdb_tune/test_output/'
TEST_TPCC_FILE = '/home/orange/MATune/csdb_tune/tpcc_output/'
MAX_REWARD = -1

REWARD_NOW = -1

CE_LOAD_NUM = 1

USE_FIX_ACTION = True
FIX_ACTION = [
    [35600140, 37, 3910, 1, 93, 3173, 42, 1024, 1000, 38, 8192, 0, 10, 30, 0, 56, 200], 
    [141436799, 37, 5584, 842, 8192, 0, 56, 30],
    [38912349, 37, 1000, 1076, 7242, 0, 56, 997]
]
