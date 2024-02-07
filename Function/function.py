import os
import time
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

log_dir = './log/'
model_dir = './model/'
game_map_dir = './map/'
memory_dir = './model/memory/'
experiment_log_file_path = './experiment_log.log'
experiment_qlearning_log_file_path = './experiment_qlearning_log.log'


# qlearning的初始化
def qlearning_init():
    create_dir(log_dir)
    create_dir(model_dir)
    create_dir(game_map_dir)
    if not os.path.exists(experiment_qlearning_log_file_path):
        with open(experiment_qlearning_log_file_path, 'w') as f:
            f.write(
                'experiment_times; state_size; action_size; best_rewards; model_path; '
                'log_dir;model_dir; remark; game_map\n')
    else:
        with open(experiment_qlearning_log_file_path, 'r') as f:
            content = f.read()
            content = content.split('\n')
            lines = list(filter(''.__ne__, content))
            if len(lines) > 1:
                last_line = lines[-1]
                last_line = last_line.split(';')
                return int(last_line[0]) + 1
    return 1

# 创建文件夹函数
def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# 创建文件函数
def experiment_save_init():
    create_dir(game_map_dir)
    create_dir(log_dir)
    create_dir(model_dir)
    if not os.path.exists(experiment_log_file_path):
        with open(experiment_log_file_path, 'w') as f:
            # 实验次数，最大分数的一次step值，最大分数的一次分数值，最大分数的一次金币，最大分数的一次剩余行动力，使用的模型参数路径，使用的模型路径，输出的日志文件夹，输出的模型文件夹，输出最好一次的模型参数文件路径，输出最好一次的日志文件路径，训练次数，保存间隔，备注，游戏设置
            f.write(
                'experiment_times; state_size; action_size; best_episode; end_node; game_steps; best_rewards; '
                'model_dict_path; model_path;'
                'log_dir;model_dir; remark; game_map\n')
    else:
        with open(experiment_log_file_path, 'r') as f:
            content = f.read()
            content = content.split('\n')
            lines = list(filter(''.__ne__, content))
            if len(lines) > 1:
                last_line = lines[-1]
                last_line = last_line.split(';')
                return int(last_line[0]) + 1
    return 1


# 输入秒，然后格式化输出时分秒
def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '{:0>2d}h {:0>2d}m {:0>2d}s'.format(int(h), int(m), int(s))


# 获取时间信息，以下划线连接
def get_time_info():
    time_info = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    return time_info
