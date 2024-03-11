import scipy.io as sio
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os
import argparse

parser = argparse.ArgumentParser(description='tSNE visualize')
parser.add_argument('--mode', type=str, default='test', metavar='N')
parser.add_argument('--saved_z_file', type=str, default='save_z.mat', metavar='N')
parser.add_argument('--n_subjects', type=int, default=20, metavar='N')
parser.add_argument('--n_time_points', type=int, default=1200, metavar='N')
parser.add_argument('--n_components', type=int, default=2, metavar='N')
parser.add_argument('--z_mode', type=str, default='z_value', choices=['z_value', 'z_distribution'])

args = parser.parse_args()
mode = args.mode
n_subjects = args.n_subjects
n_time_points = args.n_time_points
n_components = args.n_components

data_paths = './split_dataset_paths.csv'
# 读取 CSV 文件
train_paths = []
val_paths = []
test_paths = []

with open(data_paths, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        train_paths.append(row['train'])
        val_paths.append(row['valid'])
        test_paths.append(row['test'])

train_z_paths = [os.path.dirname(train_path) + f'/{args.saved_z_file}' for train_path in train_paths if train_path != '']
val_z_paths = [os.path.dirname(val_path) + f'/{args.saved_z_file}' for val_path in val_paths if val_path != '']
test_z_paths = [os.path.dirname(test_path) + f'/{args.saved_z_file}' for test_path in test_paths if test_path != '']

# choose subjects
train_z_paths = random.sample(train_z_paths, n_subjects)
val_z_paths = random.sample(val_z_paths, n_subjects)
test_z_paths = random.sample(test_z_paths, n_subjects)
mix_z_paths = random.sample(train_z_paths, int(n_subjects / 2)) + random.sample(test_z_paths, int(n_subjects / 2))

z_paths = []
if mode == 'mix':
    z_paths = mix_z_paths
elif mode == 'train':
    z_paths = train_z_paths
elif mode == 'test':
    z_paths = test_z_paths
else:
    print('[ERROR]: should choose a right mode')

z_value_samples = []
subject_ids = []
z_loaded_values = 'z_latent_values' if args.z_mode == 'z_value' else 'z_distributions'
for i, z_path in enumerate(z_paths):
    z_values = sio.loadmat(z_path)[z_loaded_values]
    time_idx = [i for i in range(n_time_points)]
    z_value_sample = z_values[time_idx, :]
    z_value_samples.append(z_value_sample)
    subject_ids.append([i + 1] * n_time_points)
z_value_samples = np.vstack(z_value_samples)
subject_ids = np.concatenate(subject_ids, axis=0)

# 创建t-SNE对象
tsne = TSNE(n_components=2)

# 对数据进行降维
z_tsne = tsne.fit_transform(z_value_samples)

# 可视化数据
fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 9])
fig.set_size_inches(16, 10)
plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)

# 计算Silhouette指数
silhouette_avg = silhouette_score(z_tsne, subject_ids)

silhouette_values = silhouette_samples(z_tsne, subject_ids)

y_lower = 5
for i in range(1, n_subjects + 1):
    # Aggregate the silhouette scores for samples belonging to
    # subject i, and sort it
    ith_subject_silhouette_values = silhouette_values[subject_ids == i]
    ith_subject_silhouette_values.sort()
    y_upper = y_lower + 10

    ax1.fill_betweenx(
        np.linspace(y_lower, y_upper, ith_subject_silhouette_values.shape[0]), 
        0, ith_subject_silhouette_values,
        facecolor='black', edgecolor='black', alpha=0.7,
    )
    y_lower = y_upper + 5 

ax1.set_title(f"The silhouette index. avg={silhouette_avg:.2f}")
ax1.set_yticks([10 + 15 * i for i in range(int(n_subjects))])
ax1.set_yticklabels([i + 1 for i in range(int(n_subjects))])
ax1.set_xlabel("Silhouette values")
ax1.set_ylabel("Subject labels")

ax2.scatter(z_tsne[:, 0], z_tsne[:, 1], c=subject_ids, cmap='tab20b')
ax2.set_title('t-SNE Visualization of z Representation')
ax2.set_xlabel('Axis 1')
ax2.set_ylabel('Axis 2')

# set color bar
norm = Normalize(1, n_subjects)
cmap = plt.cm.tab20b
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax2, shrink=0.5, ticks=[1, n_subjects], label='subject id')

n = 0
while os.path.exists(f'./silhouette_{mode}_papaer_nc{args.n_components}_sub{n_subjects}_{n}.png'):
    n += 1
plt.savefig(f'./silhouette_{mode}_papaer_nc{args.n_components}_sub{n_subjects}_{n}.svg')

