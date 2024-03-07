from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import random
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os

# matplotlib.use('Agg')  # 使用非交互式后端
n_subjects = 20
n_time_points = 100
X = np.random.rand(n_time_points * n_subjects, 256)
subject_ids = [[i + 1] * n_time_points for i in range(n_subjects)]
subject_ids = np.concatenate(subject_ids, axis=0)

# 创建t-SNE对象
tsne = TSNE(n_components=2)

# 对数据进行降维
z_tsne = tsne.fit_transform(X)

# 可视化数据

fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 9])
fig.set_size_inches(16, 10)
plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)

# 计算Silhouette指数
silhouette_avg = silhouette_score(X, subject_ids)

silhouette_values = silhouette_samples(z_tsne, subject_ids)

y_lower = 5
for i in range(1, n_subjects + 1):
    # Aggregate the silhouette scores for samples belonging to
    # subject i, and sort it
    ith_subject_silhouette_values = silhouette_values[subject_ids == i]

    ith_subject_silhouette_values.sort()

    # size_subject_i = ith_subject_silhouette_values.shape[0]
    # y_upper = y_lower + size_subject_i
    # y_upper = y_lower + n_time_points * 0.01
    y_upper = y_lower + 10

    # color = cm.nipy_spectral(float(i) / n_subjects)
    ax1.fill_betweenx(
        np.linspace(y_lower, y_upper, ith_subject_silhouette_values.shape[0]),
        0,
        ith_subject_silhouette_values,
        facecolor='black',
        edgecolor='black',
        alpha=0.7,
    )

    # Label the silhouette plots with their subject numbers at the middle
    # ax1.text(-0.05, y_lower + 0.5 * size_subject_i, str(i))

   # Compute the new y_lower for next plot
    y_lower = y_upper +5  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various subjects.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_yticks([10 + 15 * i for i in range(int(n_subjects))])
ax1.set_yticklabels([i + 1 for i in range(int(n_subjects))])
ax1.set_ylabel("Cluster label")

# set color bar
norm = Normalize(1, n_subjects)
cmap = plt.cm.tab20b
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax2, shrink=0.5, ticks=[1, n_subjects], label='subject id')

ax2.scatter(z_tsne[:, 0], z_tsne[:, 1], c=subject_ids, cmap='tab20b')
ax2.set_title('t-SNE Visualization of Iris Dataset')
ax2.set_xlabel('Axis 1')
ax2.set_ylabel('Axis 2')
plt.show()