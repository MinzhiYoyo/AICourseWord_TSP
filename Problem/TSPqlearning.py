import math
import os
import matplotlib.pyplot as plt
import numpy as np

NODE_NUM = 20
CITY_REWARDS = NODE_NUM * 2

class TSPqlearning:
    def __init__(self, node_num):
        self.distance_table = None
        self.steps = None
        self.rewards = None
        self.costs = None
        self.done = None
        self.node_num = node_num
        self.map_nodes = None

        self.current_node = self.reset(reset_node=True)
        self.actions = np.arange(self.node_num, dtype=np.int64)
        self.visited = None
        self.game_log = None
        self.route = None

    def reset(self, reset_node=True, node_data=None, save_node_path=None):
        self.rewards = 0
        self.done = False
        self.current_node = 0
        self.steps = 0
        self.costs = 0
        self.visited = np.zeros(shape=self.node_num, dtype=np.bool_)
        self.route = np.full(shape=self.node_num, fill_value=-1, dtype=np.int64)
        self.route[0] = 0
        if reset_node:
            self._generate_node(node_data)
            self._init_distance_table()
        self.save(save_node_path)
        map_datastr = self._save_data_to_string()
        map_datastr.replace('\n', ';')
        self.game_log = map_datastr
        self.game_log += '\n'
        return self._get_state()

    def step(self, action):
        reward = -CITY_REWARDS
        if not self.done:
            if action in self.actions[1:]:
                if self.visited[action]:
                    reward = -CITY_REWARDS
                else:
                    costs = self.distance_table[self.current_node, action]
                    reward = -costs
                    self.visited[action] = True
                    self.costs += costs
                    self.current_node = action
                self.route[self.steps] = action
            elif action == 0:
                reward = -CITY_REWARDS
                if self.visited.all():
                    reward += CITY_REWARDS
                    self.done = True
                else:
                    reward -= CITY_REWARDS
            else:
                raise ValueError('action out of range. Error in TSPqlearning.step()')
            self.steps += 1
            self.route[self.steps] = action
            self.rewards += reward
        info = 'action={}, reward={}, costs={}, steps={}'.format(action, reward, self.costs, self.steps)
        self.game_log += info
        self.game_log += '\n'
        return self._get_state(), reward, self.done, info

    def render(self):
        # 画出每个点，然后依次按route画路由，显示costs和rewards，并且不阻塞
        plt.cla()
        plt.title('TSP')
        plt.scatter(self.map_nodes[:, 0], self.map_nodes[:, 1], c='b')
        plt.plot(self.map_nodes[self.route[:self.steps + 1], 0], self.map_nodes[self.route[:self.steps + 1], 1], c='r')
        plt.text(0, 0, 'costs={}, rewards={}'.format(self.costs, self.rewards), fontsize=12)
        plt.pause(0.1)


    def _generate_node(self, node_data=None):
        # 生成 node_num个坐标点，每个点都在 (node_num, node_num) 之内
        if node_data is None or not self.load(node_data):
            self.map_nodes = np.vstack((np.array([0, 0]),
                                        np.random.randint([0, 0], [self.node_num - 1, self.node_num - 1],
                                                          size=(self.node_num, 2))))

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
            return False
        if os.path.exists(content):
            with open(content, 'r') as f:
                content = f.read()
        else:
            return False
        self._load_data_from_string(content)
        return True

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

    def _init_distance_table(self):
        self.distance_table = np.zeros((self.node_num, self.node_num), dtype=np.float64)
        for i in range(self.node_num):
            for j in range(i, self.node_num):
                self.distance_table[i, j] = TSPqlearning._get_two_distance_euclidean(self.map_nodes[i], self.map_nodes[j])
                self.distance_table[j, i] = self.distance_table[i, j]

    @staticmethod
    def _get_two_distance_euclidean(p1, p2):
        return 0 if p1[0] == p2[0] and p1[1] == p2[1] else math.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _get_state(self):
        return self.current_node
