import numpy as np
import random
from tune import train
from algorithm.TD3 import Actor
from algorithm.TD3 import Critic
from algorithm.TD3 import TD3
from algorithm.al_utils import ReplayBuffer

import pickle
import time
import globalValue
from maEnv.env import CEEnv
from maEnv import utils

MEMORY_SIZE = 1000  # 经验池大小
MEMORY_WARMUP_SIZE = 30  # 预存一部分经验之后再开始训练
BATCH_SIZE = 16
REWARD_SCALE = 1  # reward 缩放系数
NOISE = 0.05  # 动作噪声方差
TRAIN_EPISODE = 100  # 训练的总episode数


# td3 test
def td3_train(args):
    env = CEEnv()
    # Set seeds(但是MA的环境里没有定义seed,TODO:后续需要定义seed?)
    # env.seed(seed)
    # env.action_space.seed(seed)
    # torch.manual_seed(seed)
    np.random.seed(args.seed)

    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # max_action = float(env.action_space.high[0])
    obs_dim = env.state_dim
    act_dim = env.action_dim
    max_action = 1
    # 动作值归一化处理之后最大动作值为1？

    # TODO: 如何确定max_action?一定需要max_action吗？
    kwargs = {
        "state_dim": obs_dim,
        "action_dim": act_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # 格式化文件名并输出
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    # model = Model(act_dim)
    # algorithm = DDPG(
    #     model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    # agent = Agent(algorithm, obs_dim, act_dim)

    # 初始化TD3算法
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    # 加载模型
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(obs_dim, act_dim)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode = 0
    # while episode < TRAIN_EPISODE:
    #     # 每训练5个episode，做一次评估
    #     print('Start a new round,this round include 5 episode and 1 evaluate process!')
    #     for i in range(5):
    #         print('episode = ', episode)
    #         episode_reward = train.run_episode(agent, env, replay_buffer)
    #         episode += 1
    #
    #     print('-------------start_eval_test-------------')
    #     eval_reward = train.evaluate_ce(env, agent)
    #     # logger.info('episode:{}    Test reward:{}'.format(
    #     # episode, eval_reward))
    #     print('episode:{}    Test reward:{}'.format(episode, eval_reward))
    #
    #     # 保存模型
    #     ckpt = './1ce/model_dir/ce_steps_{}.ckpt'.format(int(time.time()))
    #     f = open("./1ce/rpm_dir/ce_rpm_full_0416_1.txt", "wb")
    #
    #     # 保存模型
    #     print('-----------save_model-----------')
    #     print('ckpt = ', ckpt)
    #     agent.save(ckpt)
    #     # 保存回放内存,写盘很费时间，注意控制写盘频率
    #     pickle.dump(replay_buffer, f)
    #     f.close()