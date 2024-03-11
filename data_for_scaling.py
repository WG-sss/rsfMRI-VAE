import scipy.io as sio
import csv
import random
import numpy as np
import os

def get_z_data(mode:str ='test', n_subjects: int=20, n_time_points: int=1200,
               n_components: int=2, z_mode: str='z_value') -> None:

    data_paths = './split_dataset_paths.csv'
    # loading csv file
    train_paths = []
    val_paths = []
    test_paths = []

    with open(data_paths, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_paths.append(row['train'])
            val_paths.append(row['valid'])
            test_paths.append(row['test'])

    train_z_paths = [os.path.dirname(train_path) + '/saved_z_beta9.mat' for train_path in train_paths if train_path != '']
    val_z_paths = [os.path.dirname(val_path) + '/saved_z_beta9.mat' for val_path in val_paths if val_path != '']
    test_z_paths = [os.path.dirname(test_path) + '/saved_z_beta9.mat' for test_path in test_paths if test_path != '']

    # choose subjects
    z_paths = []
    random.seed(1234)
    if mode == 'mix':
        mix_z_paths = random.sample(train_z_paths, int(n_subjects / 2)) + random.sample(test_z_paths, int(n_subjects / 2))
        z_paths = mix_z_paths
    elif mode == 'train':
        train_z_paths = random.sample(train_z_paths, n_subjects)
        z_paths = train_z_paths
    elif mode == 'test':
        test_z_paths = random.sample(test_z_paths, n_subjects)
        z_paths = test_z_paths
    else:
        print('[ERROR]: should choose a right mode')

    z_value_samples = []
    subject_ids = []
    save_z_data = {}
    z_loaded_values = 'z_latent_values' if z_mode == 'z_value' else 'z_distributions'
    for i, z_path in enumerate(z_paths):
        z_values = sio.loadmat(z_path)[z_loaded_values]
        time_idx = [i for i in range(n_time_points)]
        z_value_sample = z_values[time_idx, :]
        z_value_samples.append(z_value_sample)
        subject_ids.append([i + 1] * n_time_points)
    
    save_z_data["sapmle_z_data"] = np.vstack(z_value_samples)
    save_z_data["subject_ids"] = np.concatenate(subject_ids, axis=0)

    save_dir = './samples_for_scaling'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f'{mode}_{z_loaded_values}.mat'
    sio.savemat(os.path.join(save_dir, file_name), save_z_data)


if __name__ == "__main__":
    get_z_data(mode='test', z_mode="z_value")
    get_z_data(mode='test', z_mode="z_distribution")
    get_z_data(mode='train', z_mode="z_value")
    get_z_data(mode='train', z_mode="z_distribution")

