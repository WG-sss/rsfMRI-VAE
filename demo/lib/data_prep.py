import argparse
import numpy as np
import scipy.io as sio
import h5py
import os

parser = argparse.ArgumentParser(description='VAE for fMRI generation')
parser.add_argument('--time-points', type=int, default=1200, help='number of time points')
parser.add_argument('--img-size', type=int, default=192, help='size of geometric-reformatted image')
parser.add_argument('--fmri-path', default='./data', type=str, help='path of the input fMRI data')
parser.add_argument('--trans-path', default='./result', type=str,
                    help='path of the geometric reformatting transformation')
parser.add_argument('--output-path', default='./data', type=str, help='path of the output data for VAE inference')
print('here above')


# Transform the data
def GenerateData(args, left_trans_mat, right_trans_mat):
    # left_surf_data = np.zeros([args.time_points,1,args.img_size,args.img_size])
    # right_surf_data = np.zeros([args.time_points,1,args.img_size,args.img_size])

    fmri_file = os.path.join(args.fmri_path, 'fMRI.mat')
    fmri_data = sio.loadmat(fmri_file)['Normalized_fMRI']
    left_data = fmri_data[0:29696, :]  # 29696 * 1200
    right_data = fmri_data[29696:59412, :]  # 29716 * 1200
    print(
        f'Loading data the size of the left hemisphere is {left_data.shape}; the size of the right hemisphere is {right_data.shape}')
    # left (36864, 29696) * (29696, 1200) ==> Transform ==> (1200, 36864) ==> (1200, 192, 192)
    left_surf_data = np.expand_dims(left_trans_mat.dot(left_data).T.reshape((-1, args.img_size, args.img_size)), axis=1)
    print(left_surf_data.shape)
    # right (36864, 29716) * (29716, 1200) ==> Transform ==> (1200, 36864) ==> (1200, 192, 192)
    right_surf_data = np.expand_dims(right_trans_mat.dot(right_data).T.reshape((-1, args.img_size, args.img_size)),
                                     axis=1)
    print(right_surf_data.shape)
    print('here in generate data')

    return left_surf_data, right_surf_data


# Save the training data as hdf5 file
def SaveData(left_surf_data, right_surf_data, left_mask, right_mask, file_path):
    print(left_surf_data.shape)
    if os.path.isfile(file_path):
        print('Output Data Exists. Will delete it and generate a new one')
        os.system('rm -r ' + file_path)
    H5File = h5py.File(file_path, 'w')
    H5File['LeftData'] = left_surf_data.astype('float32')
    H5File['RightData'] = right_surf_data.astype('float32')
    H5File['LeftMask'] = left_mask.astype('float32')
    H5File['RightMask'] = right_mask.astype('float32')
    H5File.close()
    print('here in save_data')


if __name__ == "__main__":
    print('--------------------------here in main------------------------------------')
    args = parser.parse_args()
    # check data availability of the target folder
    if os.path.isdir(args.output_path):
        print('Target directory exists: ' + args.output_path)
    else:
        os.system('mkdir ' + args.output_path)
        print('Target directory does not exist and is created: ' + args.output_path)

    # Loading transformation data
    left_trans_mat = sio.loadmat(os.path.join(args.trans_path,'Left_fMRI2Grid_192_by_192_NN.mat'))['grid_mapping']
    print(f'The shape of the loaded left-transoformation file is: {left_trans_mat.shape}')
    right_trans_mat = sio.loadmat(os.path.join(args.trans_path,'Right_fMRI2Grid_192_by_192_NN.mat'))['grid_mapping']
    print(f'The shape of the loaded right-transoformation file is: {right_trans_mat.shape}')

    # Loading Brain Mask
    # Loading Regular_Grid_Left_Mask (192, 192)
    left_mask  = sio.loadmat(os.path.join(args.trans_path,'MSE_Mask.mat'))['Regular_Grid_Left_Mask']
    # Loading Regular_Grid_Right_Mask (192, 192)
    right_mask = sio.loadmat(os.path.join(args.trans_path,'MSE_Mask.mat'))['Regular_Grid_Right_Mask']

    # Generate the Left and Right Data
    file_path = os.path.join(args.output_path, 'transformed_fMRI_data.h5')  # 1200 * 192 * 192
    left_surf_data, right_surf_data = GenerateData(args, left_trans_mat, right_trans_mat)
    SaveData(left_surf_data, right_surf_data, left_mask, right_mask, file_path)
