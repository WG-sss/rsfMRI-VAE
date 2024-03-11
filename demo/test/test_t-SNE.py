import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris  # 这里使用鸢尾花数据集作为示例

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 创建t-SNE对象
tsne = TSNE(n_components=2)

# 对数据进行降维
X_tsne = tsne.fit_transform(X)

# 可视化数据
plt.figure(figsize=(8, 6))
for i in range(len(np.unique(y))):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=f'Class {i}')
plt.title('t-SNE Visualization of Iris Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()

# 计算Silhouette指数
silhouette_avg = silhouette_score(X, y)
print("整体数据集的Silhouette指数:", silhouette_avg)
