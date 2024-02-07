import json
import math
import os.path
import random
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
import torch.optim as optim

from Function.function import get_time_info, experiment_log_file_path, game_map_dir, experiment_save_init, log_dir, \
    model_dir, create_dir, format_time, memory_dir
from Problem.TSP import TSProblem, NODE_NUM


class DQNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNet, self).__init__()
        # 改成两层卷积层，两层池化层，一层全连接层处理图像
        # 两层全连接层处理info数据
        # 将上述两个结果拼接后，再进行一层全连接层
        # 最后输出
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_size)

    def forward(self, x):
        # x is a tuple, x[0] is image, x[1] is info

        x = torchF.relu(self.fc1(x))
        x = torchF.relu(self.fc2(x))
        x = torchF.relu(self.fc3(x))
        actions_value = self.out(x)
        return actions_value


# 定义经验回放类，并且使用tensor进行存储
class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)  # list(tuple)
        state, action, reward, next_state = zip(*batch)
        return state, action, reward, next_state

    def save(self, path):
        data = [[buf[0].tolist(), int(buf[1]), int(buf[2]), buf[3].tolist()] for buf in self.buffer]
        if os.path.exists(path):
            data_last = json.load(open(path, 'r'))
            data += data_last
        json.dump(data, open(path, 'w'))

    def load(self, path):
        if not path or not os.path.exists(path):
            return
        data = json.load(open(path, 'r'))
        self.buffer = [(np.array(buf[0]), buf[1], buf[2], np.array(buf[3])) for buf in data]

    def __len__(self):
        return len(self.buffer)


# 定义DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size,
                 learning_rate=0.001,  # 学习率
                 gamma=0.9,  # 折扣因子
                 epsilon_max=0.9,  # 最大探索概率
                 epsilon_min=0.01,  # 最小探索概率
                 epsilon_decay=1000,  # 探索概率衰减率
                 batch_size=64,  # 批次大小
                 memory_size=1000,  # 经验回放池大小
                 TAU=0.01,  # 目标网络更新率
                 memory_mode=None,  # 经验回放池模式
                 model_path=None,  # 模型路径
                 model_dict_path=None,  # 模型字典路径
                 device=None,  # 设备
                 memory_path=None,  # 经验回放池路径
                 ):
        # 保存超参数
        self.q_eval = None
        self.loss = None
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.tau = TAU

        self.memory_mode = memory_mode
        self.device = device
        # 保存状态和动作空间大小
        self.state_size = state_size
        self.action_size = action_size
        # 初始化经验回放池
        self.memory = ReplayMemory(self.memory_size)
        if memory_path:
            self.memory.load(memory_path)

        # 两个网络，一个eval_net，一个target_net，eval实时训练且更新，target每隔一段更新
        # 初始化模型
        # self.eval_net = DQNet(self.state_size, self.action_size)
        # self.target_net = DQNet(self.state_size, self.action_size)
        if model_path:
            self.eval_net = torch.load(model_path).to(self.device)
            self.target_net = torch.load(model_path).to(self.device)
        else:
            self.eval_net = DQNet(self.state_size, self.action_size).to(self.device)
            self.target_net = DQNet(self.state_size, self.action_size).to(self.device)
        # 初始化模型字典
        if model_dict_path:
            self.eval_net.load_state_dict(torch.load(model_dict_path))
        self.target_net.load_state_dict(self.eval_net.state_dict())

        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        # 初始化学习步数
        self.train_step = 0
        print(
            'state size is {}, action size is {}, device is {}'.format(self.state_size, self.action_size, self.device))

    def select_action(self, state):
        eps_threshold = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
                        math.exp(-1. * self.train_step / self.epsilon_decay)
        self.train_step += 1
        p_rand = np.random.uniform(0, 1)
        if p_rand > eps_threshold:
            with torch.no_grad():
                ret = self.eval_net(state)
                ret = ret.max(0)
                ret = ret.indices
                return ret
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.long)

    def train(self, batch_size=None, train_interval=None):
        if batch_size is None:
            batch_size = self.batch_size

        # 没有足够多的经验，不训练
        if len(self.memory) < batch_size:
            return

        # 如果没到训练间隔，那么就不训练
        if train_interval is not None:
            if self.train_step % train_interval != 0:
                return

        # 从经验回放池中抽取批次数据
        state, action, reward, next_state = self.memory.sample(batch_size)
        state = tuple(map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0), state))
        action = tuple(
            map(lambda x: torch.tensor(x, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(0), action))
        reward = tuple(
            map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0), reward))
        next_state = tuple(
            map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0), next_state))
        # 计算Q值
        state_batch = torch.cat(state).to(self.device)
        action_batch = torch.cat(action).to(self.device)
        reward_batch = torch.cat(reward).to(self.device)
        next_state_batch = torch.cat(next_state).to(self.device)

        # 计算Q值
        self.q_eval = self.eval_net(state_batch).gather(1, action_batch)

        # 计算目标Q值
        q_next = self.target_net(next_state_batch).detach()
        q_target = reward_batch + self.gamma * q_next.max(1)[0].view(batch_size, 1)

        # 计算损失
        self.loss = self.loss_func(self.q_eval, q_target)

        # 优化模型
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.update_target_net_from_eval()

    def save_dict(self, model_path):
        torch.save(self.eval_net.state_dict(), model_path)

    def save_model(self, model_path):
        torch.save(self.eval_net, model_path)

    def update_target_net_from_eval(self):
        target_net_dict = self.target_net.state_dict()
        eval_net_dict = self.eval_net.state_dict()
        for k in target_net_dict.keys():
            target_net_dict[k] = eval_net_dict[k] * self.tau + target_net_dict[k] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_dict)


TSPtransition = namedtuple(
    'TSPtransition',
    ('episode', 'rewards', 'cur_node', 'game_step')
)


def dqn_train_model(model_path=None, model_dict_path=None, memory_file=None, remark='', need_save=True,
                    game_map_file_name=None):
    # 实验保存初始化，并获取实验次数
    experiment_num = experiment_save_init()
    experiment_log_dir = os.path.join(log_dir, 'experiment_{}'.format(experiment_num))
    experiment_model_dir = os.path.join(model_dir, 'experiment_{}'.format(experiment_num))
    loss_record_path = os.path.join(experiment_log_dir, 'loss_record_{}_{}.log'.format(experiment_num, get_time_info()))
    create_dir(experiment_log_dir)
    create_dir(experiment_model_dir)
    create_dir(game_map_dir)
    create_dir(memory_dir)
    memory_path = os.path.join(memory_dir, memory_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    episodes_num = 1000 if torch.cuda.is_available() else 500

    # 初始化游戏环境
    env = TSProblem(node_num=NODE_NUM)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, device=device, model_path=model_path,
                     model_dict_path=model_dict_path, memory_path=memory_path)
    # 分数最大的一次记录
    best = TSPtransition(0, -math.inf, 0, 0)

    best_log_file = os.path.join(experiment_log_dir, 'log_{}_best.log'.format(get_time_info()))
    best_model_dict_file = os.path.join(experiment_model_dir, 'model_dict_{}_best.pth'.format(get_time_info()))

    console_output_interval = episodes_num // 20  # 打印间隔
    save_interval = episodes_num // 10  # 保存间隔

    loss_info = 'episode; step; loss\n'
    start_time = time.time()
    map_file = os.path.join(game_map_dir, game_map_file_name)

    if not os.path.exists(map_file):
        map_file = None
    for episode in range(episodes_num):
        if map_file:
            state = env.reset(reset_node=True, node_data=map_file)
        else:
            map_file = os.path.join(game_map_dir, game_map_file_name)
            state = env.reset(save_node_path=map_file)
        while not env.done:
            action = agent.select_action(torch.tensor(state, dtype=torch.float32, device=agent.device)).item()
            next_state, reward, done, info = env.step(action)

            agent.memory.push(state, action, reward, next_state)
            agent.train(train_interval=10)
            if agent.loss:
                loss_info += '{}; {}; {}\n'.format(episode, agent.train_step, agent.loss.item())
            state = next_state
        if env.rewards > best.rewards:
            best = TSPtransition(episode=episode, rewards=env.rewards, cur_node=env.current_node, game_step=env.steps)
            env.save(best_log_file)
            agent.save_dict(best_model_dict_file)

        if episode % console_output_interval == 0:
            end_time = time.time()
            # 计算每次的平均耗时
            print('Train {} rounds needs {}. Average time is {}.'.format(console_output_interval,
                                                                         format_time(end_time - start_time),
                                                                         format_time((end_time - start_time) / console_output_interval)))
            print('Current: {}, rewards={:.4f}, cur_node={}, game_steps={}\n'
                  'Best: {}, rewards={:.4f}, cur_node={}, game_steps={}\n'.format(
                episode, env.rewards, env.current_node, env.steps,
                best.episode, best.rewards, best.cur_node, best.game_step
            ))
        if episode % save_interval == 0:
            if need_save:
                env.save(os.path.join(experiment_log_dir, 'log_{}_{}.log'.format(get_time_info(), episode)))
                agent.save_dict(
                    os.path.join(experiment_model_dir, 'model_dict_{}_{}.pth'.format(get_time_info(), episode)))
            start_time = time.time()

    # experiment_log = 'experiment_times; state_size; action_size; best_episode; best_rewards; model_dict_path;
    # model_path; ' 'log_dir;model_dir; remark; game_map
    experiment_log = '{}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}\n'.format(
        experiment_num, best.episode, best.rewards, best.cur_node, best.game_step , model_dict_path, model_path, experiment_log_dir,
        experiment_model_dir, remark, map_file + '_' + str(env.node_num))

    with open(experiment_log_file_path, 'a') as f:
        f.write(experiment_log)
    with open(loss_record_path, 'w') as f:
        f.write(loss_info)
    agent.memory.save(path=memory_path)
    return best_model_dict_file
