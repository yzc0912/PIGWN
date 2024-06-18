import numpy as np
import resource
# 获取当前进程的内存使用情况
def get_memory_usage():
    # 获取内存使用情况，单位为 KB
    memory_usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # 将单位转换为 MB
    memory_usage_mb = memory_usage_kb / 1024.0
    return memory_usage_mb
def Add_Window_Horizon_mat(data, window, horizon):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    index = 0

    while index < end_index:
        temp = data[index:index+window]
        X.append(temp[::2]+temp[1::2])
        print("Memory usage before loading data:", get_memory_usage())
        print(index)
        index = index + 1

    X = np.array(X)
    return X

if __name__ == "__main__":
    data = np.load('../data/usa/airline_matrices.npz')
    departures = data['departures']
    print("Memory usage before loading data:", get_memory_usage())
    X_train = Add_Window_Horizon_mat(departures, 8, 8)