import torch
import numpy as np
import scipy.io as sio
import argparse
from utils import *
from fMRIVAE_Model import *
import torch.utils.data
import os

parser = argparse.ArgumentParser(description='VAE for fMRI generation')
parser.add_argument('--batch-size', type=int, default=120, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--zdim', type=int, default=256, metavar='N',
                    help='dimension of latent variables')
parser.add_argument('--data-path', default='./demo/data/100408_REST1LR/val_fMRI_data.h5', type=str, metavar='DIR',
                    help='path to dataset, which should be concatenated with either _train.h5 or _val.h5 to yield training or validation datasets')
parser.add_argument('--z-path', type=str, default='./demo/data/100408_REST1LR/z_distribution/',
                    help='path to saved z files. Only Z files must be in this path, not other files.')
parser.add_argument('--resume', type=str, default='./demo/checkpoint/checkpoint40-9.pth.tar',
                    help='checkpoint file name of saved model parameters to load')
parser.add_argument('--img-path', type=str, default='./demo/data/100408_REST1LR/recon_img/',
                    help='path to save reconstructed images')
parser.add_argument('--mode', type=str, default='encode',
                    help='Mode to get data. Choose one of [encode, decode]')

args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BetaVAE(z_dim=args.zdim, nc=1).to(device)
if os.path.isfile(args.resume):
    print("==> Loading checkpoint: ", args.resume)
    checkpoint = torch.load(args.resume, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
else:
    print('[ERROR] Checkpoint not found: ', args.resume)
    raise RuntimeError

if args.mode.lower() == 'encode':
    '''
    Encoder mode. 
    Image --> z
    '''
    test_loader = load_dataset_test(args.data_path, args.batch_size)

    print(f'Mode: Eecode \n Distribution of Z will be saved at: {args.z_path}')
    if not os.path.isdir(args.z_path):
        os.system('mkdir ' + args.z_path)

    for batch_idx, (xL, xR) in enumerate(test_loader):
        xL = xL.to(device)
        xR = xR.to(device)
        z_distribution = model.encode(xL, xR)
        mu = z_distribution[:, :args.zdim].clone().detach()
        logvar = z_distribution[:, args.zdim:].clone().detach()
        z = model.reparameterize(mu, logvar).detach()
        save_data = {'z': z}
        sio.savemat(args.z_path + f'save_z{batch_idx}.mat', save_data)


elif args.mode.lower() == 'decode':
    '''
    Decoder mode.
    z --> reconstructed image
    '''
    print(f'Mode: Decode \n Reconstructed images will be saved at: {args.img_path}')
    if not os.path.isdir(args.z_path):
        print(f'[ERROR] Dir does not exist: {args.z_path}')
        raise RuntimeError
    if not os.path.isdir(args.img_path):
        os.system('mkdir ' + args.img_path)

    file_list = [f for f in os.listdir(args.z_path) if f.split('_')[0] == 'save']
    for batch_idx, filename in enumerate(file_list):
        z = sio.loadmat(args.z_path + f'save_z{batch_idx}.mat')['z']
        z = torch.tensor(z, device=device)

        # z = model.reparameterize(1, 1)
        # eps_z = z

        # test_save_Z_mat(z,batch_idx,args.z_path)
        x_recon_L, x_recon_R = model.decode(z)
        # x_recon_L, x_recon_R = model.decode(torch.tensor(mu).to(device))
        save_image_mat(x_recon_L, x_recon_R, args.img_path, batch_idx)

else:
    print('[ERROR] Selected mode: ' + args.mode + ' is not valid. \n Choose either [encode, decode]')
