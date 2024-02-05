from Solution.DQN import dqn_train_model


def train():
    remark = 'train map 1'
    game_map_file_name = 'train_map_1.txt'
    model_dict_path = None
    memory_path = 'memory_1.json'
    for i in range(10):
        model_dict_path = dqn_train_model(model_dict_path=model_dict_path, memory_file=memory_path, remark=remark,
                                          game_map_file_name=game_map_file_name)


if __name__ == '__main__':
    train()
