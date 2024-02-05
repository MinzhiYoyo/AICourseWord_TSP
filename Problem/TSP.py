import math
import os.path

import numpy as np

DISTANCE_MIN = 1
DISTANCE_MAX = 100
CITY_REWARDS = 200

NODE_NUM = 20


class TSProblem:
    def __init__(self, node_num, reset_node=True):
        self.rewards = None
        self.step = None
        self.done = None
        self.node_num = node_num
        self.map_nodes = None

        state = self.reset(False)
        self.state_size = len(state)
        self.actions = np.arange(self.node_num, dtype=np.int64)
        self.action_size = len(self.actions)
        self.current_node = None
        self.visited = None

    def reset(self, reset_node=True, node_data=None):
        self.rewards = 0
        self.done = False
        self.current_node = 0
        self.steps = 0
        if reset_node:
            self._generate_node(node_data)
        return self._get_state()

    def step(self, action):
        if not self.done:
            if action in self.actions:
                last_node = self.current_node
                self.current_node = action
                self.rewards += CITY_REWARDS if self.visited[self.current_node] else 0
                self.rewards -= (
                    TSProblem._get_two_distance_euclidean(self.map_nodes[last_node], self.map_nodes[self.current_node]))
                self.steps += 1
        if (self.steps > 0 and self.current_node == 0) or self.rewards < 0:
            self.done = True
        info = 'node={}, rewards={}'.format(self.current_node, self.rewards)
        return self._get_state(), self.rewards, self.done, info

    def _generate_node(self, node_data=None):
        # 生成 node_num个坐标点，每个点都在 (node_num, node_num) 之内
        if node_data is None:
            self.map_nodes = np.random.randint([0, 0], [self.node_num, self.node_num], size=(self.node_num, 2))
        else:
            if os.path.exists(node_data):
                pass
            else:
                pass

    def _load_data_from_string(self, data):
        pass

    def _save_data_to_string(self):
        pass

    @staticmethod
    def _get_two_distance_euclidean(p1, p2):
        return 0 if p1[0] == p2[0] and p1[1] == p2[1] else math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _get_state(self):
        pass
