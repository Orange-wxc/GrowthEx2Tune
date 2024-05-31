import argparse
import threading
from maEnv.server import Server
from tune import train_wxc as train
#from tune import train as train
import time
import globalValue
import random
import os
import parser

from paddle import fluid
# fluid.enable_dygraph()



def train_test_se():
    train.train(True)

def train_test_ce():
    # train.train(2)
    #train.train(2)
    matuner = train.MATuner("RL", "SAC_2")
    #matuner.cal_tree_shap()
    matuner.train(mode=1)



#使用td3算法的简单测试
def td3_test(**args):
    print("----td3 test----")
    train.td3_train(args)


def eval_ce():
    train.evaluate_t(False)

def server_test():
    server = Server(globalValue.EVENT)
    server.server()

def load_bash():
    bash_time = 30
    # tables = 100
    # thread_count = 32
    #
    # if globalValue.EVAL_TEST:
    #     tables = 100
    #     thread_count = 32

    # s = 'sysbench /home/fox/lxj/sysbench-1.0.19/src/lua/oltp_read_only.lua ' \
    #     '--mysql-user=root --mysql-password=mysql ' \
    #     '--mysql-host=127.0.0.1 ' \
    #     '--mysql-port=3306 ' \
    #     '--mysql-db=test --mysql-storage-engine=innodb ' \
    #     '--table-size=10000 ' \
    #     '--tables={} ' \
    #     '--threads={} ' \
    #     '--events=0 ' \
    #     '--report-interval=10 ' \
    #     '--range_selects=off ' \
    #     '--time=330 run'.format(tables, thread_count)

    # s = '/home/fox/TestTools/tpcc-mysql/tpcc_start ' \
    #     '-h 192.168.1.102 ' \
    #     '-P 3306 ' \
    #     '-d tpcc ' \
    #     '-u root ' \
    #     '-p mysql ' \
    #     '-w 100 ' \
    #     '-c 5 ' \
    #     '-r 0 ' \
    #     '-l 60000 ' \
    #     '-i 10 ' \
    #     '>> /home/fox/subject/matune/test_data/0330/tpcclog_0330/mysql_tpcc_20210415.log'
    f = '/Users/dawn/Desktop/sysbench_result_{}.log'.format(int(time.time()))
    cmd = 'touch '+f
    os.system(cmd)
    s = '/usr/local/bin/sysbench /usr/local/share/sysbench/oltp_read_only.lua ' \
        '--mysql-user=dawn --mysql-password=mysql ' \
        '--mysql-host=%s ' \
        '--mysql-port=3306 ' \
        '--mysql-db=test --mysql-storage-engine=innodb ' \
        '--table-size=100000 ' \
        '--tables=10 ' \
        '--threads=32 ' \
        '--events=0 ' \
        '--report-interval=10 ' \
        '--range_selects=off ' \
        '--time=%d run' \
        ' >> %s' \
        % (globalValue.CONNECT_MA_IP, bash_time, f)

    print(s)

    # 记录训练开始时间
    globalValue.EPISODE_START_TIME = time.time()

    os.system(s)

    globalValue.EVAL_TEST = False

# def load():
#     while(True):
#         if(globalValue.LOAD_EVENT.isSet()):
#             globalValue.LOAD_EVENT.clear()
#             print("load_thread start!!!")
#             load_bash()
#         else:
#             print("load_thread is waiting for LOAD_EVENT!")
#             globalValue.LOAD_EVENT.wait()


if __name__ == '__main__':

    # TODO:希望可以使用命令行指定TD3算法或DDPG算法
    # 允许指定一些特殊参数
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    # parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    # parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    # parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    # parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    # parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    # parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    # parser.add_argument("--discount", default=0.99)  # Discount factor
    # parser.add_argument("--tau", default=0.005)  # Target network update rate
    # parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    # parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    # parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    # parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    # parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    # args = parser.parse_args()
    #
    # # 为存储结果/模型创建文件夹
    # if not os.path.exists("./results"):
    #     os.makedirs("./results")
    #
    # if args.save_model and not os.path.exists("./models"):
    #     os.makedirs("./models")

    # 1.先进行一些全局变量的初始化工作
    globalValue.LOAD_EVENT = threading.Event()

    # 2.之后开启训练线程

    # 负载训练线程开启
    # load_thread = threading.Thread(target=load_bash)
    # load_thread.start()
    # print("load bash--------->")

    # 参数推荐线程开启
    # eval_thread = threading.Thread(target=eval_ce)
    # eval_thread.start()

    # CE训练线程开启
    ce_train_thread = threading.Thread(target=train_test_ce)
    ce_train_thread.start()

    # SE训练线程开启
    # se_train_thread = threading.Thread(target=train_test_se)
    # se_train_thread.start()

    # server_thread = threading.Thread(target=server_test)

    # train_thread = threading.Thread(target=train_test)

    # server_thread.start()

    # time.sleep(10)

    # train_thread.start()


