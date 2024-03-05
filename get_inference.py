import torch
import numpy as np
import scipy.io as sio
import argparse
from utils import *
from fMRIVAE_Model import *
import torch.utils.data
from typing import Literal
import os


def get_inference(batch_size: int = 120,
             seed: int = 1,
             z_dim: int = 256,
             data_path: str = './demo/data/100408_REST1LR/val_fMRI_data.h5',
             z_dir: str = './demo/data/100408_REST1LR/z_distribution/',
             resume_file: str = './demo/checkpoint/checkpoint40-9.pth.tar',
             img_dir: str = './demo/data/100408_REST1LR/recon_img/',
             mode: Literal['decode', 'encode', 'inference'] = 'encode'):
    """
    description: 'VAE for fMRI generation')
    para:
        batch_size: input batch size for training (default: 120)
        seed: random seed (default: 1)
        z_dim: dimension of latent variables (default: 120)
        data_path: path to dataset, the data is a .h5 file containing (batch, 1, 192, 192) image
        z_dir: files directory to save z. Only Z files must be in this path, not other files.
        resume_file: checkpoint file name of saving model parameters to load
        img_dir: path to save reconstructed images
        mode: Mode to get data. Choose one of [encode, decode]
    retrun:
        encode: z
        decode: reconstruction image
        inference: encode and decode
    """

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BetaVAE(z_dim=z_dim, nc=1).to(device)
    if os.path.isfile(resume_file):
        print("==> Loading checkpoint: ", resume_file)
        checkpoint = torch.load(resume_file, map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('[ERROR] Checkpoint not found: ', resume_file)
        raise RuntimeError

    def encode() -> None:
        '''
        Encoder mode. 
        Image --> z
        '''
        test_loader = load_dataset_test(data_path, batch_size)

        print(f'Mode: Eecode \n Representation of Z will be saved at: {z_dir}')
        if not os.path.isdir(z_dir):
            os.system('mkdir ' + z_dir)

        for batch_idx, (x_l, x_r) in enumerate(test_loader):
            x_l = x_l.to(device)
            x_r = x_r.to(device)
            z_distribution = model.encode(x_l, x_r)
            mu = z_distribution[:, :z_dim].clone().detach()
            logvar = z_distribution[:, z_dim:].clone().detach()
            z = model.reparameterize(mu, logvar).detach()
            save_data = {'z': z}
            sio.savemat(z_dir + f'save_z{batch_idx}.mat', save_data)

    def decode() -> None:
        '''
        Decoder mode.
        z --> reconstructed image
        '''
        print(f'Mode: Decode \n Reconstructed images will be saved at: {img_dir}')
        if not os.path.isdir(z_dir):
            print(f'[ERROR] Dir does not exist: {z_dir}')
            raise RuntimeError
        if not os.path.isdir(img_dir):
            os.system('mkdir ' + img_dir)

        file_list = [f for f in os.listdir(z_dir) if f.split('_')[0] == 'save']
        for batch_idx, filename in enumerate(file_list):
            z = sio.loadmat(z_dir + f'save_z{batch_idx}.mat')['z']
            z = torch.tensor(z, device=device)

            # z = model.reparameterize(1, 1)
            # eps_z = z

            # test_save_Z_mat(z,batch_idx,z_dir)
            x_recon_L, x_recon_R = model.decode(z)
            # x_recon_L, x_recon_R = model.decode(torch.tensor(mu).to(device))
            save_image_mat(x_recon_L, x_recon_R, img_dir, batch_idx)

    if mode.lower() == 'encode':
        encode()
    elif mode.lower() == 'decode':
        decode()
    elif mode.lower() == 'inference':
        encode()
        decode()
    else:
        print('[ERROR] Selected mode: ' + mode + ' is not valid. \n Choose either [encode, decode, inference]')
