import csv
import os
from get_inference import *

data_paths = './demo/split_dataset_paths.csv'
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

val_paths = [val_dir for val_dir in val_paths if val_dir != '']
test_paths = [test_dir for test_dir in test_paths if test_dir != '']

checkpoint_file = './Checkpoint/checkpoint100-9.pth.tar'

h5_data_paths = train_paths[:20] + val_paths[:20] + test_paths[:20]
print(h5_data_paths)
for h5_path in h5_data_paths[:20]:
    parent_dir = os.path.dirname(h5_path) + '/'
    get_inference(data_path=h5_path, z_dir=parent_dir, resume_file=checkpoint_file,
                  img_dir=parent_dir, mode='encode')
