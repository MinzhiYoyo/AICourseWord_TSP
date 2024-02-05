import math
import os.path
from collections import namedtuple

import numpy as np

DISTANCE_MIN = 1
DISTANCE_MAX = 100
NODE_NUM = 20
CITY_REWARDS = NODE_NUM ** 2


class TSProblem:
    def __init__(self, node_num):
        self.steps = None
        self.rewards = None
        self.done = None
        self.node_num = node_num
        self.map_nodes = None

        state = self.reset(reset_node=True)
        self.state_size = len(state)
        self.actions = np.arange(self.node_num, dtype=np.int64)
        self.action_size = len(self.actions)
        self.current_node = None
        self.visited = None
        self.game_log = None

    def reset(self, reset_node=True, node_data=None, save_node_path=None):
        self.rewards = 0
        self.done = False
        self.current_node = 0
        self.steps = 0
        self.visited = np.zeros(shape=self.node_num, dtype=np.bool_)
        if reset_node:
            self._generate_node(node_data)
        self.save(save_node_path)
        map_datastr = self._save_data_to_string()
        map_datastr.replace('\n', ';')
        self.game_log = map_datastr
        self.game_log += '\n'
        return self._get_state()

    def step(self, action):
        if not self.done:
            if action in self.actions:
                # if self.visited[action] == False:
                #     last_node = self.current_node
                #     self.current_node = action
                #     self.visited[action] = True
                #     self.rewards += CITY_REWARDS
                #     self.steps += 1

                last_node = self.current_node
                self.current_node = action
                self.rewards += CITY_REWARDS if self.visited[self.current_node] == False else 0
                self.visited[self.current_node] = True
                self.rewards -= (
                    TSProblem._get_two_distance_euclidean(self.map_nodes[last_node], self.map_nodes[self.current_node]))
                self.steps += 1
        if (self.steps > 0 and self.current_node == 0) or self.rewards < 0:
            self.done = True
        info = 'node={}, rewards={}, steps={}'.format(self.current_node, self.rewards, self.steps)
        self.game_log += info
        self.game_log += '\n'
        return self._get_state(), self.rewards, self.done, info

    def _generate_node(self, node_data=None):
        # 生成 node_num个坐标点，每个点都在 (node_num, node_num) 之内
        if node_data is None:
            self.map_nodes = np.random.randint([0, 0], [self.node_num, self.node_num], size=(self.node_num, 2))
        else:
            self.load(node_data)

    def _load_data_from_string(self, data):
        data_list = data.split('\n')
        ret_data = []
        for a_line in data_list:
            d = a_line.split(',')
            if len(d) < 2:
                continue
            ret_data.append([
                int(d[0][1:]),
                int(d[1][:-1])
            ])
        self.map_nodes = np.array(ret_data[1:])

    def load(self, content):
        if not content:
            return
        if os.path.exists(content):
            with open(content, 'r') as f:
                content = f.read()
        self._load_data_from_string(content)

    def save(self, file_path):
        if not file_path:
            return
        with open(file_path, 'w') as f:
            f.write(self._save_data_to_string())

    def _save_data_to_string(self):
        save_str = '({},{})\n'.format(self.map_nodes.shape[0], self.map_nodes.shape[1])
        for node in self.map_nodes:
            save_str += '({},{})\n'.format(node[0], node[1])
        return save_str

    @staticmethod
    def _get_two_distance_euclidean(p1, p2):
        return 0 if p1[0] == p2[0] and p1[1] == p2[1] else math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _get_state(self):
        # 状态包括，当前节点的node_num个权值，以及-1 0 1代表已探索、当前位置、未探索
        ans = np.zeros(shape=self.node_num, dtype=np.float32)
        for i, node in enumerate(self.map_nodes):
            ans[i] = TSProblem._get_two_distance_euclidean(node, self.map_nodes[self.current_node])

        ans2 = np.zeros(shape=self.node_num, dtype=np.float32)
        ans2[np.where(self.visited == True)] = 1
        ans2[np.where(self.visited == False)] = -1
        ans2[self.current_node] = 0
        return np.hstack((ans, ans2))
