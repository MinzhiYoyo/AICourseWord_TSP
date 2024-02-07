# 使用遗传算法解决TSP问题
import math
import os.path
import time

import numpy as np
from matplotlib import pyplot as plt


class TSPGA:
    def __init__(self, node_num, file_path=None, save_path=None):
        self.nodes = None
        self.node_num = node_num
        if not file_path or not os.path.exists(file_path):
            self.generate_node_random()
        else:
            self.load(file_path)
        self._generate_distance_table()
        if save_path:
            self.save(save_path)

    def render(self, route: np.ndarray):
        # 画出每个点，然后依次按route画路由，显示costs和rewards，并且不阻塞
        plt.cla()
        plt.title('TSP')
        plt.scatter(self.nodes[:, 0], self.nodes[:, 1], c='b')
        # self.nodes 总共包含 self.node_num 个节点，依次连接，并且最后一个节点连接第一个节点
        x = np.hstack((self.nodes[route[:self.node_num], 0], self.nodes[route[0], 0]))
        y = np.hstack((self.nodes[route[:self.node_num], 1], self.nodes[route[0], 1]))
        plt.plot(x, y, c='r')
        plt.pause(0.1)

    # 评估路径
    def evaluate(self, route: np.ndarray):
        costs = np.sum([self.distance_table[route[i], route[i + 1]] for i in range(self.node_num - 1)])
        costs += self.distance_table[route[self.node_num - 1], route[0]]
        return costs

    def generate_node_random(self):
        self.nodes = np.vstack((np.array([0, 0]),
                                np.random.randint([0, 0], [self.node_num, self.node_num],
                                                  size=(self.node_num - 1, 2))))

    def _generate_distance_table(self):
        self.distance_table = np.zeros((self.node_num, self.node_num), dtype=np.float64)
        for i in range(self.node_num):
            for j in range(i, self.node_num):
                self.distance_table[i, j] = TSPGA._get_two_distance_euclidean(self.nodes[i], self.nodes[j])
                self.distance_table[j, i] = self.distance_table[i, j]

    def load(self, file_path):
        if os.path.exists(file_path):
            self.nodes = np.load(file_path)

    def save(self, file_path):
        np.save(file_path, self.nodes)

    @staticmethod
    def _get_two_distance_euclidean(param, param1):
        return 0 if param[0] == param1[0] and param[1] == param1[1] else math.sqrt(
            (param[0] - param1[0]) ** 2 + (param[1] - param1[1]) ** 2)


# 遗传算法包括以下几个步骤：
# 1. 初始化种群
# 2. 选择
# 3. 交叉
# 4. 变异
# 5. 评估
# 6. 重复2-5步骤
# 7. 终止


class GA:
    def __init__(self, generation_num, population_num, mutation_rate, crossover_rate, node_num, file_path=None,
                 save_path=None):
        # self.next_population = None
        self.idx = None
        self.fitness = None
        self.costs = None
        self.population = None
        self.crossover_rate = crossover_rate
        self.generation_num = generation_num
        self.population_num = population_num
        self.mutation_rate = mutation_rate

        self.node_num = node_num
        self.tsp = TSPGA(node_num, file_path=file_path, save_path=save_path)

    # 定义模拟
    def simulate(self, crossover_num=1, convergence_exit=False, separate=False):
        self.init_population()
        last_costs = self.costs[self.idx[0]]
        count = 10
        for i in range(self.generation_num):
            print(f'第{i}代{self.population.shape[0]}个个体最优成本为{self.costs[self.idx[0]]}')
            self.next_generation(parents_num=crossover_num, separate=separate)
            self.tsp.render(self.population[self.idx[0]])
            if convergence_exit:
                if abs(last_costs - self.costs[self.idx[0]]) < 1e-12:
                    count -= 1
                    if count == 0:
                        break
                else:
                    count = 10
                last_costs = self.costs[self.idx[0]]
        print(f'最终{self.population.shape[0]}个个体最优成本为{self.costs[self.idx[0]]}')

    def init_population(self):
        self.population = np.array([np.random.permutation(self.node_num - 1) for _ in range(self.population_num)])
        # 把所有的population都+1
        self.population += 1
        self.population = np.hstack((np.zeros((self.population_num, 1), dtype=np.int64), self.population))
        self.costs = np.zeros(self.population_num, dtype=np.float64)
        self.evaluate()  # 评估第一代

    def next_generation(self, parents_num=2, separate=False):
        """
        生成下一代
        :param separate:
        :param parents_num: 表示每次选择的父母数量，不重复，可以为1，可以为2
        :return:
        """
        next_population = np.zeros(shape=(self.population.shape[0], self.population.shape[1]), dtype=np.int64)
        i = 0
        while i < self.population_num//4:
            # 选择
            parents = self.select(num=parents_num)
            # 交叉
            if np.random.rand() < self.crossover_rate:  # 需要交叉
                next_population[i:i + parents_num] = self.crossover(parents)
            else:  # 不需要交叉
                next_population[i:i + parents_num] = self.population[parents].copy()

            # 变异
            for j in range(parents_num):
                if np.random.rand() < self.mutation_rate:
                    next_population[i+j] = self.mutation(next_population[i+j])
            i += parents_num

        if separate:
            # 只交叉，不变异
            while i < self.population_num*2//4:
                # 选择
                parents = self.select(num=parents_num)
                # 交叉
                if np.random.rand() < self.crossover_rate:  # 需要交叉
                    next_population[i:i + parents_num] = self.crossover(parents)
                else:  # 不需要交叉
                    next_population[i:i + parents_num] = self.population[parents].copy()

                i += parents_num

            # 只变异，不交叉
            while i < self.population_num*3//4:
                # 选择
                parents = self.select(num=1)
                # 变异
                next_population[i] = self.mutation(self.population[parents[0]].copy())
                i += 1

        next_population[i:] = self.population[self.idx[:len(next_population)-i]].copy()
        self.population = next_population
        self.evaluate()  # 评估这一代

    def select(self, num=2):
        # 按照costs从小到大排序，获取下标，然后选择前population_num/2个
        return np.random.choice(np.arange(self.population_num), num, replace=False,
                                p=self.fitness / np.sum(self.fitness))

    def crossover(self, parents):
        # 判断是一个父母还是两个父母
        if len(parents) == 1:
            # 父母只有一个，就是把它自己前后交叉
            # 选择交叉点
            # 交叉点不能是0和node_num-1
            # 交换 1:交叉点-1 与 交叉点:node_num-1 -> 0, idx:end, 1:idx-1
            idx = np.random.randint(2, self.node_num - 1)
            child = np.hstack((self.population[parents[0]][0:1], self.population[parents[0]][idx:],
                               self.population[parents[0]][1:idx]))
            return child
        elif len(parents) == 2:
            # parents[0]为母亲，parents[1]为父亲
            # 在父亲中选出交叉点idx，选择父亲中idx之后的点
            # 然后在母亲中寻找到父亲idx之后的所有点，按照父亲中idx之后的点的顺序，重新排列

            idx = np.random.randint(2, self.node_num - 1)  # 选择交叉点

            mother = self.population[parents[0]].copy()
            father = self.population[parents[1]].copy()

            mother_X = mother[idx:].copy()  # 母亲交叉点之后的染色体
            father_X = father[idx:].copy()  # 父亲交叉点之后的染色体
            # 找出父亲染色体在母亲中的位置
            mother_idxs = np.array([np.where(mother == i)[0][0] for i in father_X])
            # 找出母亲染色体在父亲中的位置
            father_idxs = np.array([np.where(father == i)[0][0] for i in mother_X])

            # 对位置进行排序
            mother_idxs.sort()
            father_idxs.sort()

            # 按照母亲的位置，重新排列父亲的染色体
            mother[mother_idxs] = father_X
            father[father_idxs] = mother_X

            return np.vstack((mother, father))

    def mutation(self, child):
        # 随机选择两个不同的点，进行交换
        # 不同点不能是0
        idx1, idx2 = np.random.choice(np.arange(1, self.node_num), 2, replace=False)
        child[idx1], child[idx2] = child[idx2], child[idx1]
        return child

    def evaluate(self):
        # 对当前这代进行评估
        for i in range(self.population_num):
            self.costs[i] = self.tsp.evaluate(self.population[i])
        # 对costs归一化
        min_val = np.min(self.costs)
        max_val = np.max(self.costs)
        self.fitness = (max_val - self.costs) * 100 / (max_val - min_val)
        self.idx = np.argsort(self.fitness)[::-1]


if __name__ == '__main__':
    ga = GA(500, 1000, 0.3, 0.8, 50, file_path='nodes2.npy', save_path='nodes2.npy')
    ga.simulate(crossover_num=2, convergence_exit=False, separate=True)
    time.sleep(3)
