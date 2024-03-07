from sklearn.decomposition import PCA
import numpy as np

# 假设您有一个 (N, M) 形状的矩阵
data = np.random.rand(100, 20)  # 这里假设有 100 个样本，每个样本有 20 个特征

# 创建 PCA 模型并拟合数据
pca = PCA(n_components=3)  # 指定要保留的主成分数量
pca.fit(data)

# 得到转换后的数据
transformed_data = pca.transform(data)

# 查看转换后的数据形状
print("转换后的数据形状：", transformed_data.shape)
