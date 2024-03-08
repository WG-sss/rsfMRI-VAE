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
        z_dir: files directory to save z_latent_variables. Only Z files must be in this path, not other files.
        resume_file: checkpoint file name of saving model parameters to load
        img_dir: path to save reconstructed images
        mode: Mode to get data. Choose one of [encode, decode, inference]
    retrun:
        encode: z_latent_variables
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
        """
        Encoder mode.
        Image --> z_latent_variables
        """
        data_set_h5 = H5Dataset(data_path)
        data_loader = torch.utils.data.DataLoader(data_set_h5, batch_size=batch_size, shuffle=False)

        print(f'Mode: Eecode \n Representation of Z will be saved at: {z_dir}')
        if not os.path.isdir(z_dir):
            os.system('mkdir ' + z_dir)

        saved_z = {}
        z_distributions = []
        z_latent_values = []

        for batch_idx, (x_l, x_r) in enumerate(data_loader):
            z_distribution = model.encode(x_l.to(device), x_r.to(device))
            mu = z_distribution[:, :z_dim].clone().detach()
            logvar = z_distribution[:, z_dim:].clone().detach()
            z_latent_values = model.reparameterize(mu, logvar).detach()
            z_distributions.append(z_distribution.detach().numpy())
            z_latent_values.append(z_latent_values.detach().numpy())
        z_distributions = np.concatenate(z_distributions, axis=0)
        z_latent_values = np.concatenate(z_latent_values, axis=0)
        saved_z['z_distributions'] = z_distributions  # dimension: N * (256 + 256), mu + logvar
        saved_z['z_latent_values'] = z_latent_values  # dimension: N * 256
        sio.savemat(z_dir + f'saved_z_beta9.mat', saved_z)

    def decode() -> None:
        """
        Decoder mode.
        z_latent_variables --> reconstructed image
        """
        print(f'Mode: Decode \n Reconstructed images will be saved at: {img_dir}')
        if not os.path.isdir(z_dir):
            print(f'[ERROR] Dir does not exist: {z_dir}')
            raise RuntimeError
        if not os.path.isdir(img_dir):
            os.system('mkdir ' + img_dir)

        saved_z_mat = MatLatentSet(z_dir + 'saved_z_beta9.mat')
        z_latent_value_loader = torch.utils.data.DataLoader(saved_z_mat, batch_size=batch_size, shuffle=False)

        saved_recon_img_l = []
        saved_recon_img_r = []
        saved_recon_img = {}
        for batch_idx, z_latent_value in enumerate(z_latent_value_loader):
            x_recon_l, x_recon_r = model.decode(z_latent_value.to(device))
            saved_recon_img_l.append(x_recon_l.detach().numpy())
            saved_recon_img_r.append(x_recon_r.detach().numpy())

        saved_recon_img['recon_L'] = np.concatenate(saved_recon_img_l, axis=0)
        saved_recon_img['recon_R'] = np.concatenate(saved_recon_img_r, axis=0)
        sio.savemat(img_dir + '/recon_img_z.mat', saved_recon_img)

    if mode.lower() == 'encode':
        encode()
    elif mode.lower() == 'decode':
        decode()
    elif mode.lower() == 'inference':
        encode()
        decode()
    else:
        print('[ERROR] Selected mode: ' + mode + ' is not valid. \n Choose either [encode, decode, inference]')
