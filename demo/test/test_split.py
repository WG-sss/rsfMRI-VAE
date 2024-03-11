import csv

# 定义保存路径
data_pathes = '../split_dataset_paths.csv'

# 读取 CSV 文件
train_dirs = []
val_dirs = []
test_dirs = []

with open(data_pathes, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        train_dirs.append(row['train'])
        val_dirs.append(row['valid'])
        test_dirs.append(row['test'])

val_dirs = [val_dir for val_dir in val_dirs if val_dir != '']
test_dirs = [test_dir for test_dir in test_dirs if test_dir != '']

print(len(train_dirs), len(val_dirs), len(test_dirs))

for idx, path in enumerate(val_dirs):
    print(idx, path, type(path))

print("" == None)
print("" is None)

