import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# 生成一些随机数据
X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=0.6, random_state=0)

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=4, random_state=0)
cluster_labels = kmeans.fit_predict(X)

# 计算每个样本的轮廓系数
silhouette_values = silhouette_samples(X, cluster_labels)

# 获取每个类别的平均轮廓系数
cluster_silhouette_avg = []
for cluster in set(cluster_labels):
    mask = (cluster_labels == cluster)
    cluster_samples = silhouette_values[mask]
    silhouette_avg_cluster = np.mean(cluster_samples)
    cluster_silhouette_avg.append(silhouette_avg_cluster)


# 定义绘制带尖峰的条形图的函数
def plot_bar_with_peak(ax, x, y):
    ax.bar(x, y, color='blue')
    for i, val in enumerate(y):
        ax.plot([i, i], [0, val], color='red', linewidth=2)
        ax.plot(i, val, marker='^', color='red', markersize=10)


# 创建图形和子图
fig, ax = plt.subplots()

# 调用自定义绘制函数绘制带尖峰的条形图
plot_bar_with_peak(ax, range(len(cluster_silhouette_avg)), cluster_silhouette_avg)

# 设置图形属性
ax.set_xlabel('Cluster')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score for Each Cluster')

# 显示图形
plt.show()
