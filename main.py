from Solutions.DQN import dqn_train_model
from Solutions.qlearning import QLearning_train_table


def train():
    remark = 'train map 1'
    game_map_file_name = 'train_map_1.txt'
    model_dict_path = None
    memory_path = 'memory_1.json'
    for i in range(1):
        model_dict_path = dqn_train_model(model_dict_path=model_dict_path, memory_file=memory_path, remark=remark,
                                          game_map_file_name=game_map_file_name)


def train_qlearning():
    remark = 'train qlearning 1'
    game_map_file_name = 'train_map_1.txt'
    table_path = None
    for i in range(20):
        table_path = QLearning_train_table(table_path=table_path, remark=remark, game_map_file_name=game_map_file_name)


if __name__ == '__main__':
    # train()
    train_qlearning()
