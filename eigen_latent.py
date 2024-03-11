import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils import load_z_data_paths
import scipy.io as sio

# 加载数据
# z_data_paths = load_z_data_paths(data_paths='./split_dataset_paths.csv', mode='train', subject_num=None)
#
# z_data = []
# for z_path in z_data_paths:
#     z_latent_variables = sio.loadmat(z_path)['z_latent_variables'].T
#     z_data.append(z_latent_variables)
# z_data = np.vstack(z_data)
# print(z_data.shape) #

# test
z_data = np.random.random((256, 12000))
# data = loadmat('./pop.mat')['data']
# z_data = data  # 每一行代表一个个体随时间变化的值
N, M = z_data.shape  # N vertex, M time points
# A = z_data - np.mean(z_data, axis=1)[:, np.newaxis]

z_data_next_time = np.concatenate((z_data[:, 1:], z_data[:, 0][:, np.newaxis]), axis=1)
z_latent_gradients = z_data - z_data_next_time

# sig = A * A
# A = A / np.sqrt(np.mean(sig, axis=1))[:, np.newaxis]
u, P, v = np.linalg.svd(z_latent_gradients)

mink = 10
P = P ** 2
acc1 = P / np.sum(P)
acc = np.zeros(N)
for i in range(N):
    acc[i] = np.sum(acc1[:i + 1])

plt.plot(acc)
minrank = np.arange(1, N + 1)
minrank = minrank[acc > 0.8]
# plt.plot(np.arange(1, 11), acc[:10])
plt.plot(minrank, acc[minrank - 1], 'ro')
plt.title("accumulating eigen value")

plt.show()
