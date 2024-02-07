import math
import os
import time

import numpy as np

from Function.function import qlearning_init, log_dir, model_dir, create_dir, game_map_dir, get_time_info, format_time, \
    experiment_qlearning_log_file_path
from Problem.TSP import NODE_NUM, TSProblem
from Problem.TSPqlearning import TSPqlearning


# Q-learning算法
class QLearning:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, epsilon_min=0.01,
                 epsilon_decay=0.995):
        self.q_table = np.zeros((num_states, num_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.routes = np.zeros(shape=(0, num_actions+1), dtype=np.int64)
        self.num_states = num_states
        self.num_actions = num_actions

    def run_episode(self, env: TSPqlearning):
        state = env.reset(reset_node=False)
        route = np.full(shape=self.num_actions+1, fill_value=-1, dtype=np.int64)
        route[0] = state
        for step in range(self.num_states - 1):
            action = self.choose_action(state)
            next_state, reward, done, info = env.step(action)
            route[step + 1] = next_state
            self.train(state, action, reward, next_state)
            state = next_state
            if env.done:
                break
        route[-1] = env.costs
        env.render()
        time.sleep(0.5)
        self.routes = np.vstack((self.routes, route))

    def train(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                reward + self.discount_factor * max_future_q)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        self.q_table[state, action] = new_q

    def choose_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() < epsilon:
            return np.random.randint(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state, :])

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            np.save(f, self.q_table)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.q_table = np.load(f)

    def save_routes(self, file_path):
        with open(file_path, 'wb') as f:
            np.save(f, self.routes)

    def load_routes(self, file_path):
        with open(file_path, 'rb') as f:
            self.routes = np.load(f)


def QLearning_TSP_path(table_path, game_map_file_name):
    env = TSPqlearning(node_num=NODE_NUM)
    agent = QLearning(num_states=env.node_num, num_actions=env.node_num, learning_rate=0.1, discount_factor=0.9,
                      epsilon=0.1)
    agent.load(table_path)
    game_map_file_name = os.path.join(game_map_dir, game_map_file_name)
    if not os.path.exists(game_map_file_name):
        raise FileNotFoundError('The game map file is not found.')
    state = env.reset(node_data=game_map_file_name)
    while not env.done:
        action = agent.choose_action(state, 1.1)
        next_state, reward, done, info = env.step(action)
        state = next_state
    return env.costs


def QLearning_train_table(table_path=None, remark='', need_save=True, game_map_file_name=None):
    # 实验保存初始化，并获取实验次数
    experiment_num = qlearning_init()
    experiment_log_dir = os.path.join(log_dir, 'experiment_qlearning_{}'.format(experiment_num))
    experiment_model_dir = os.path.join(model_dir, 'experiment_qlearning_{}'.format(experiment_num))
    create_dir(experiment_log_dir)
    create_dir(experiment_model_dir)
    episodes_num = 1000
    # 初始化游戏环境
    env = TSPqlearning(node_num=NODE_NUM)

    best_log_file = os.path.join(experiment_log_dir, 'log_{}_best.log'.format(get_time_info()))
    best_table_file = os.path.join(experiment_model_dir, 'table_{}_best.pth'.format(get_time_info()))
    console_output_interval = episodes_num // 20  # 打印间隔
    save_interval = episodes_num // 10  # 保存间隔
    start_time = time.time()
    map_file = os.path.join(game_map_dir, game_map_file_name)
    if not os.path.exists(map_file):
        env.reset(reset_node=True, save_node_path=map_file, node_data=map_file)
    else:
        env.reset(reset_node=True, node_data=map_file)
    agent = QLearning(num_states=env.node_num, num_actions=env.node_num)
    if table_path is not None:
        agent.load(table_path)
    best_costs = math.inf
    best_episode = 0
    print('Start training {} Q-learning model.'.format(experiment_num))
    for episode in range(episodes_num):
        agent.run_episode(env)
        if env.rewards < best_costs:
            best_costs = env.rewards
            best_episode = episode
            agent.save(best_table_file)
            # env.save(best_log_file)

        if episode % save_interval == 0:
            if need_save:
                # env.save(os.path.join(experiment_log_dir, 'log_{}_{}.log'.format(get_time_info(), episode)))
                agent.save(os.path.join(experiment_model_dir, 'table_{}_{}.pth'.format(get_time_info(), episode)))
                agent.save_routes(os.path.join(experiment_log_dir, 'routes_{}_{}.route'.format(get_time_info(), episode)))

        if episode % console_output_interval == 0:
            end_time = time.time()
            # 计算每次的平均耗时
            # print('Train {} rounds needs {}. Average time is {}.'.format(console_output_interval,
            #                                                              format_time(end_time - start_time),
            #                                                              format_time((end_time - start_time) /
            #                                                                          console_output_interval)))
            print('Episode: {}/{}, Costs: {}'.format(episode, episodes_num, env.costs))
            print('Best: {}/{}, Costs: {}\n'.format(best_episode, episodes_num, best_costs))
            start_time = time.time()

    # 'experiment_times; state_size; action_size; best_costs; model_path; '
    # 'log_dir;model_dir; remark; game_map\n'
    with open(experiment_qlearning_log_file_path, 'a') as f:
        f.write('{};{};{};{};{};{};{};{};{}\n'.format(experiment_num, env.node_num, env.node_num, env.costs,
                                                      best_table_file, experiment_log_dir, experiment_model_dir,
                                                      remark, game_map_file_name))
    return best_table_file

