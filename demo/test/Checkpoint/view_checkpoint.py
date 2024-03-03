import torch

# 加载检查点文件
checkpoint = torch.load('checkpoint30.pth.tar')

# 查看检查点文件中保存的内容
# print(checkpoint.keys())
print(checkpoint['optimizer'])
