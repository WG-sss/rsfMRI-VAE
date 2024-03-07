import torch
import numpy as np
import scipy.io as sio
import argparse

from demo.lib.utils import *
# from utils import *
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
        mode: Mode to get data. Choose one of [encode, decode, inference]
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
        data_set_h5 = H5Dataset(data_path)
        data_loader = torch.utils.data.DataLoader(data_set_h5, batch_size=batch_size, shuffle=False)

        print(f'Mode: Eecode \n Representation of Z will be saved at: {z_dir}')
        if not os.path.isdir(z_dir):
            os.system('mkdir ' + z_dir)

        save_datas = []

        for batch_idx, (x_l, x_r) in enumerate(data_loader):
            z_distribution = model.encode(x_l.to(device), x_r.to(device))
            mu = z_distribution[:, :z_dim].clone().detach()
            logvar = z_distribution[:, z_dim:].clone().detach()
            z = model.reparameterize(mu, logvar).detach()
            save_datas.append(z.detach().numpy())
        save_z = {'z': np.concatenate(save_datas, axis=0)}
        sio.savemat(z_dir + f'save_z.mat', save_z)

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

        z_value_mat = MatDataset(z_dir + 'save_z.mat')
        z_value_loader = torch.utils.data.DataLoader(z_value_mat, batch_size=batch_size, shuffle=False)

        save_recon_img_l = []
        save_recon_img_r = []
        save_recon_img = {}
        for batch_idx, z in enumerate(z_value_loader):
            x_recon_l, x_recon_r = model.decode(z.to(device))
            save_recon_img_l.append(x_recon_l.detach().numpy())
            save_recon_img_r.append(x_recon_r.detach().numpy())

        save_recon_img = {'recon_L': np.concatenate(save_recon_img_l, axis=0)}
        save_recon_img = {'recon_R': np.concatenate(save_recon_img_r, axis=0)}
        sio.savemat(img_dir + '/recon_img.mat', save_recon_img)

    if mode.lower() == 'encode':
        encode()
    elif mode.lower() == 'decode':
        decode()
    elif mode.lower() == 'inference':
        encode()
        decode()
    else:
        print('[ERROR] Selected mode: ' + mode + ' is not valid. \n Choose either [encode, decode, inference]')
