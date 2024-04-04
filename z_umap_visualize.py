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
import sys
import argparse
import umap
import umap.plot
from typing import Literal, Optional


def reduce_2d_visualize(data_path: Optional[str]=None, mode: Literal['test', 'train']=None,
                        method: Literal['tsne', 'umap']='umap',
                        z_value: Literal['latent_value', 'distribution']=None,
                        save: bool=False) -> None:
    if data_path is None:
        print('[ERROR]: data_path should be specified')
        sys.exit(1)
    if z_value is None:
        print('[ERROR]: z_value should be specified')
        sys.exit(1)

    z_value_samples = sio.loadmat(data_path)['sapmle_z_data']
    subject_ids = sio.loadmat(data_path)['subject_ids'][0, :]

    if method == 'tsne':
        reducer = TSNE(n_components=2)
    elif method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.001)
    else:
        print("[ERROR]: method should be set in ['tsne','umap']")
        sys.exit(1)
    z_embedding = reducer.fit_transform(z_value_samples)

    # 可视化数据
    fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 9])
    fig.set_size_inches(16, 10)
    plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)

    # 计算Silhouette指数
    silhouette_avg = silhouette_score(z_embedding, subject_ids)
    silhouette_values = silhouette_samples(z_embedding, subject_ids)

    y_lower = 5
    for i in range(1, 20 + 1):
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
    ax1.set_yticks([10 + 15 * i for i in range(20)])
    ax1.set_yticklabels([i + 1 for i in range(20)])
    ax1.set_xlabel("Silhouette values")
    ax1.set_ylabel("Subject IDs")

    ax2.scatter(z_embedding[:, 0], z_embedding[:, 1],
                c=subject_ids, cmap='tab20b', s=0.5)
    ax2.set_title('UMAP Visualization of z Representation')
    ax2.set_xlabel('Axis 1')
    ax2.set_ylabel('Axis 2')

    # set color bar
    norm = Normalize(1, 20)
    cmap = plt.cm.tab20b
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, shrink=0.5, ticks=[1, 20], label='subject id')
    # plt.show()

    if save:
        n = 0
        while os.path.exists(f'./image/2D_map_{mode}_{z_value}_{method}_beta9_{n}.svg'):
            n += 1
        plt.savefig(f'./image/2D_map_{mode}_{z_value}_{method}_beta9_{n}.svg')
    else:
        pass
#
if __name__ == "__main__":
    data_paths = ['./samples_for_scaling/test_z_distributions.mat',
                 './samples_for_scaling/train_z_distributions.mat',
                 './samples_for_scaling/test_z_latent_values.mat',
                 './samples_for_scaling/train_z_latent_values.mat',]
    # for data_path in data_paths:
    #     mode = None
    #     z_value = None
    #     if 'test' in data_path:
    #         mode = 'test'
    #     elif 'train' in data_path:
    #         mode = 'train'
    #     else:
    #         print('[ERROR]: mode not defined')
    #         sys.exit(1)
    #     if 'distribution' in data_path:
    #         z_value = 'distribution'
    #     elif 'values' in data_path:
    #         z_value = 'latent_value'
    #     else:
    #         print('[ERROR]: z_value mode not defined')
    #         sys.exit(1)
    #
    #     for method in ['umap', 'tsne']:
    #         reduce_2d_visualize(data_path, mode=mode, method=method, z_value=z_value, save=True)
    reduce_2d_visualize(data_paths[1], mode='train', method='tsne', z_value='distribution', save=True)
