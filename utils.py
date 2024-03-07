import os
import h5py
import torch
import csv
import torch.utils.data as data
import torch.multiprocessing
import scipy.io as sio
import random


# torch.multiprocessing.set_start_method('spawn')

class H5Dataset(data.Dataset):
    def __init__(self, H5Path):
        super().__init__()
        self.H5File = h5py.File(H5Path, 'r')
        self.LeftData = self.H5File['LeftData']
        self.RightData = self.H5File['RightData']
        self.LeftMask = self.H5File['LeftMask'][:]
        self.RightMask = self.H5File['RightMask'][:]

    def __getitem__(self, index):
        return (torch.from_numpy(self.LeftData[index, :, :, :]).float(),
                torch.from_numpy(self.RightData[index, :, :, :]).float())

    def __len__(self):
        return self.LeftData.shape[0]


class MatDataset(data.Dataset):
    def __init__(self, mat_data_path):
        super().__init__()
        self.z = sio.loadmat(mat_data_path)['z']

    def __getitem__(self, index):
        return torch.from_numpy(self.z[index, :]).float()

    def __len__(self):
        return self.z.shape[0]


def save_image_mat(img_l, img_r, result_path, idx):
    save_data = {}
    save_data['recon_L'] = img_l.detach().cpu().numpy()
    save_data['recon_R'] = img_r.detach().cpu().numpy()
    sio.savemat(os.path.join(result_path, 'img{}.mat'.format(idx)), save_data)


def load_dataset(data_path, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dir = data_path + '_train.h5'
    val_dir = data_path + '_val.h5'
    train_set = H5Dataset(train_dir)
    val_set = H5Dataset(val_dir)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader


def load_dataset_test(data_path, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_dir = data_path + '.h5'
    test_set = H5Dataset(test_dir)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader


def load_z_data_paths(data_paths='./split_dataset_paths.csv', mode='mix', subject_num=20):
    # 读取 CSV 文件
    train_paths = []
    val_paths = []
    test_paths = []

    with open(data_paths, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            train_paths.append(row['train'])
            val_paths.append(row['valid'])
            test_paths.append(row['test'])

    train_z_paths = [os.path.dirname(train_path) + '/save_z.mat' for train_path in train_paths if train_path != '']
    val_z_paths = [os.path.dirname(val_path) + '/save_z.mat' for val_path in val_paths if val_path != '']
    test_z_paths = [os.path.dirname(test_path) + '/save_z.mat' for test_path in test_paths if test_path != '']

    # choose subjects without repeatability
    if subject_num is not None:
        train_z_paths = random.sample(train_z_paths, subject_num)
        val_z_paths = random.sample(val_z_paths, subject_num)
        test_z_paths = random.sample(test_z_paths, subject_num)
        mix_z_paths = random.sample(train_z_paths, int(subject_num / 2)) + random.sample(test_z_paths,
                                                                                         int(subject_num / 2))
    else:
        # choose all data
        mix_z_paths = train_z_paths + test_z_paths + val_z_paths
        pass

    z_paths = []
    if mode == 'mix':
        z_paths = mix_z_paths
    elif mode == 'train':
        z_paths = train_z_paths
    elif mode == 'test':
        z_paths = test_z_paths
    else:
        print('[ERROR]: should choose a right mode')

    return z_paths
