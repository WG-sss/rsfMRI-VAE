from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from get_inference import *
import scipy.io as sio
import os
import csv


def split_data(data_dir: str='/home/wanggao/Datasets/HCP_S1200/data') -> None:
    # 创建一个空字典来存储每个被试的路径
    subject_paths = {}

    # 遍历data目录下的所有目录
    for subject_id in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject_id)
        if not os.path.isdir(subject_dir):
            continue

        # 检查是否存在完整的两个run
        runs = [f for f in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, f))]
        if len(runs) != 4:
            print(f"{len(runs)}, don't have enough runs")
            continue

        # 检查是否存在save_z_beta9.mat文件
        mat_files = {}
        for run in runs:
            mat_file = os.path.join(subject_dir, run, 'saved_z_beta9.mat')
            if os.path.exists(mat_file):
                run_name = run.split('_')[1]
                mat_files[run_name] = mat_file

        # 检查是否每个run都存在mat文件
        if len(mat_files) == 4:
            subject_paths[subject_id] = mat_files

    # 将结果保存到CSV文件中
    csv_file = 'subjects_with_mat_path.csv'
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'REST1LR', 'REST1RL', 'REST2LR', 'REST2RL'])
        for subject_id, paths in subject_paths.items():
            row = [subject_id]
            row.extend(paths[run] for run in ['REST1LR', 'REST1RL', 'REST2LR', 'REST2RL'])
            writer.writerow(row)

    print(f'totally has {len(subject_dir)} subject and {len(subject_dir) * 4} mat file')
    print(f'Successfully saved the list of subjects with mat files to {csv_file}.')



def within_subject_eigen(data_paths: str='./subjects_with_mat_path.csv', subject_idx: str=1) -> None:
     
    # 读取 CSV 文件
    REST1 = []
    REST2 = []

    row = []
    with open(data_paths, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        row = list(reader)[subject_idx]

    run1_lr = sio.loadmat(row['REST1LR'])['z_latent_values'] # 1200 * 256 
    run1_rl = sio.loadmat(row['REST1RL'])['z_latent_values']
    run2_lr = sio.loadmat(row['REST2LR'])['z_latent_values']
    run2_rl = sio.loadmat(row['REST2RL'])['z_latent_values']

    satisfied = True
    if run1_lr.shape[1] !=256 or run1_lr.shape[0] != 1200:
        satisfied == False
    if run1_rl.shape[1] !=256 or run1_rl.shape[0] != 1200:
        satisfied == False
    if run2_lr.shape[1] !=256 or run2_lr.shape[0] != 1200:
        satisfied == False
    if run2_rl.shape[1] !=256 or run2_rl.shape[0] != 1200:
        satisfied == False

    if satisfied:
        REST1.append(run1_lr.T)
        REST1.append(run1_rl.T)
        REST2.append(run2_lr.T)
        REST2.append(run2_rl.T)

    REST1 = np.hstack(REST1)
    REST2 = np.hstack(REST2)

    return REST1, REST2


def concate_over_subject(data_paths: str='./subjects_with_mat_path.csv', subject_num:int =None) -> None:
     
    # 读取 CSV 文件
    REST1LR_paths = []
    REST1RL_paths = []
    REST2LR_paths = []
    REST2RL_paths = []

    group_z_latent_value_run1 = []
    group_z_latent_value_run2 = []

    rows = []
    with open(data_paths, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    for row in rows[:subject_num]:
        run1_lr = sio.loadmat(row['REST1LR'])['z_latent_values'] # 1200 * 256 
        run1_rl = sio.loadmat(row['REST1RL'])['z_latent_values']
        run2_lr = sio.loadmat(row['REST2LR'])['z_latent_values']
        run2_rl = sio.loadmat(row['REST2RL'])['z_latent_values']

        if run1_lr.shape[1] ==256 and run1_lr.shape[0] == 1200:
            group_z_latent_value_run1.append(run1_lr.T)
        else:
            print(f"{row['REST1LR']}'s shape not right.")
        if run1_rl.shape[1] ==256 and run1_rl.shape[0] == 1200:
            group_z_latent_value_run1.append(run1_rl.T)
        else:
            print(f"{row['REST1RL']}'s shape not right.")
        if run2_lr.shape[1] ==256 and run2_lr.shape[0] == 1200:
            group_z_latent_value_run2.append(run2_lr.T) # 1200 * 256
        else:
            print(f"{row['REST1LR']}'s shape not right.")
        if run2_rl.shape[1] ==256 and run2_rl.shape[0] == 1200:
            group_z_latent_value_run2.append(run2_rl.T)
        else:
            print(f"{row['REST2RL']}'s shape not right.")

    group_z_latent_value_run1 = np.hstack(group_z_latent_value_run1) # (256, 1200 * subject_num * 2)
    group_z_latent_value_run2 = np.hstack(group_z_latent_value_run2)

    # saving data
    # if not os.path.exists('./data/group_z_run1.mat'):
    #     sio.savemat('./data/group_z_run1.mat', {'group_z_run1': group_z_latent_value_run1})
    #     print('saved group_z_run1.mat')
    # if not os.path.exists('./data/group_z_run2.mat'):
    #     sio.savemat('./data/group_z_run2.mat', {'group_z_run2': group_z_latent_value_run2})
    #     print('saved group_z_run2.mat')

    return group_z_latent_value_run1, group_z_latent_value_run2


def eigen_micro_state(z_latent_values_matrix: np.ndarray=None, N: int=9, save_name: str=None) -> None:

    z_latent_values_matrix_next_time = np.hstack((z_latent_values_matrix[:, 1:], z_latent_values_matrix[:, 0:1]))
    z_latent_gradients = z_latent_values_matrix - z_latent_values_matrix_next_time
    U, P, _ = np.linalg.svd(z_latent_gradients)

    acc1 = P / np.sum(P)
    acc = np.zeros(256)
    for i in range(256):
        acc[i] = np.sum(acc1[:i + 1])

    plt.plot(acc)
    n = 0
    while os.path.exists(f'./accumulating_eigen_value_{n}.png'):
        n += 1
    plt.savefig(f'./accumulating_eigen_value{n}.png')

    eigen_micro_states = []
    for i in range(N):
        ith_eigen_micro_state = U[:, i:i+1] * P[i]
        eigen_micro_states.append(ith_eigen_micro_state)
    eigen_micro_states = np.hstack(eigen_micro_states)

    if not os.path.exists(save_name):
        sio.savemat(save_name, {'eigen_state': eigen_micro_states})
        print('saved eigen_micro_states')

    # return U, P, eigen_micro_states


def pearson_relation_matrix(A: np.ndarray=None, B: np.ndarray=None) -> None:
    """
    A is a matrix, whose dimention is (N, M), each column presents a variable
    return:
        a correlation coefficient matrix
    """
    if A is None:
        raise ValueError("Must give a matrix")

    # compute the correlation coefficient matrix
    A = A.T # 9 * 256
    B = B.T
    N = A.shape[0]
    correlation_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            correlation_matrix[i, j] = np.corrcoef(A[i, :], B[j, :])[0, 1]

    # figure
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='plasma', interpolation='nearest')
    plt.colorbar(shrink=0.5, ticks=[-0.8, 0, 0.8])
    plt.title('Latent Gradient space')
    plt.xlabel('run1')
    plt.ylabel('run2')
    plt.show()

    n = 0
    while os.path.exists(f'./correlation_coefficient_matrix_{n}.png'):
        n += 1
    plt.savefig(f'./correlation_coefficient_matrix_{n}.png')

    # return correlation_matrix

def generate_cortex_data(z: np.ndarray=None, 
                         resume_file: str='./Checkpoint/checkpoint100-9.pth.tar') -> None:
    if z is None:
        raise ValueError("Must give a z representation")

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetaVAE().to(device)

    if os.path.isfile(resume_file):
        print(f"==> Loading checkpoint: { resume_file }")
        checkpoint = torch.load(resume_file, map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"[ERROR] Checkpoint not found: { resume_file }")
        raise RuntimeError

    x_recon_l, x_recon_r = model.decode(z_latent_value.to(device))

    return x_recon_l.detach().cpu().numpy(), x_recon_r.detach().cpu().numpy()


def get_dteries_from_fmri():
    pass


if __name__ == "__main__":
    # split_data()
    # saved at ./data/group_z_run1 or 2.mat
    # concate_over_subject()
    # load group_z_run1 and 2.mat
    # print('start')
    # group_z_latent_run1 = sio.loadmat('./data/group_z_run1.mat')['group_z_run1'][:, :12000]
    # print('run1_data has been loaded')
    # print(group_z_latent_run1.shape)
    # eigen_micro_state(group_z_latent_run1, save_name='./data/eigen_state_run1_100.mat')
    # print('eigen state run 1 has been computed')
    # group_z_latent_run1 = None
    # #
    # #
    # group_z_latent_run2 = sio.loadmat('./data/group_z_run2.mat')['group_z_run2'][:, :12000]
    # print('run2_data has been loaded')
    # # saved ./data/eigen_state.mat
    # eigen_micro_state(group_z_latent_run2, save_name='./data/eigen_state_run2_100.mat')
    # print('eigen state run 2 has been computed')
    # group_z_latent_run2 = None

    # load eigen_state.mat
    eigen_micro_states_run1 = sio.loadmat('./data/eigen_state_run1.mat')['eigen_state']
    eigen_micro_states_run2 = sio.loadmat('./data/eigen_state_run2.mat')['eigen_state']
    print('eigen data loaded')
    pearson_relation_matrix(eigen_micro_states_run1, eigen_micro_states_run2)


