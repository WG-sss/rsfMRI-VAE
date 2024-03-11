import os
import csv
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# 定义数据集目录
data_dir = './data'

# 收集所有被试的 tranformed_fMRI_data.h5 文件路径
h5_files = []
for subject_dir in os.listdir(data_dir):
    subject_path = os.path.join(data_dir, subject_dir)
    if os.path.isdir(subject_path):
        for run_dir in os.listdir(subject_path):
            run_path = os.path.join(subject_path, run_dir)
            if os.path.isdir(run_path):
                h5_file_path = os.path.join(run_path, 'transformed_fMRI_data.h5')
                if os.path.isfile(h5_file_path):
                    h5_files.append(h5_file_path)

# 划分数据集为训练集、验证集和测试集
train_files, test_files = train_test_split(h5_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)  # 将训练集划分为训练集和验证集，此时训练集占总数据集的60%

print("训练集数量:", len(train_files))
print("验证集数量:", len(val_files))
print("测试集数量:", len(test_files))


# 定义保存路径
output_file = './split_datasets_pathes.csv'

# 保存训练集、验证集和测试集文件列表
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['train', 'valid', 'test']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for train_file, valid_file, test_file in zip(train_files, val_files, test_files):
        writer.writerow({'train': train_file, 'valid': valid_file, 'test': test_file})



