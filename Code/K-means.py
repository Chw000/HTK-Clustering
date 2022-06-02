import psutil
import os
import numpy as np
from time import time
from sklearn.cluster import KMeans


def txt2array(txt_path, delimiter):
    data_list = []
    with open(txt_path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")
        data_split = line.split(delimiter)
        temp = list(map(float, data_split))
        data_list.append(temp)

    data_array = np.array(data_list)
    return data_array


if __name__ == '__main__':
    begin_time = time()
    X = txt2array("D:/python program/数据集/UCI/UCI数据集txt格式/txt/Ionosphere.txt", ",")
    k_means = KMeans(n_clusters=6, max_iter=1000, init='random', algorithm='auto')
    k_means.fit(X)
    y_predict = k_means.predict(X)
    end_time = time()
    run_time = end_time-begin_time
    print(u'Memory usage of the current process：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    print('The elapsed time of the loop program：', run_time)
