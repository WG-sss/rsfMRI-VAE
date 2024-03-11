import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 创建一些随机数据和相应的标注
x = np.random.rand(100)
y = np.random.rand(100)
labels = np.random.rand(100)

# 绘制散点图
plt.scatter(x, y, c=labels, cmap='viridis')

# 创建颜色映射对象
norm = Normalize(vmin=np.min(labels), vmax=np.max(labels))
cmap = plt.cm.viridis
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# 添加颜色条
plt.colorbar(sm, label='Subject')

plt.show()
