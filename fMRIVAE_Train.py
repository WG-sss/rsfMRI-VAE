from __future__ import print_function
import argparse
import torch
from utils import *
from fMRIVAE_Model import *
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image
# import matplotlib.pyplot as plt
import os
import scipy.io as sio
import torch.optim as optim
import csv

# >>>>>>>>>>>>>>>>>>>>>>>> parse argument >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
parser = argparse.ArgumentParser(description='VAE for fMRI data')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--zdim', type=int, default=256, metavar='N',
                    help='dimension of latent variables (default: 256)')
parser.add_argument('--vae-beta', default=10, type=float,
                    help='beta parameter for KL-term (default: 10)')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate (default : 1e-4)')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='Adam optimizer beta2')
parser.add_argument('--resume', default='./demo/checkpoint/', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-path', default='./demo/data/100408_REST1LR/', type=str, metavar='DIR',
                    help='path to dataset, which should be concatenated with either _train.h5 or _val.h5 to yield training or validation datasets')
parser.add_argument('--apply-mask', default=True, type=bool,
                    help='Whether apply a mask of the crtical surface to the MSe loss function')
parser.add_argument('--Output-path', default='./demo/Output_Temp/', type=str,
                    help='Path to save results')
parser.add_argument('--mother-path', default='./', type=str,
                    help='Path to mother folder')
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def save_image_mat(img_r, img_l, result_path):
    save_data = {}
    save_data['recon_L'] = img_l.detach().cpu().numpy()
    save_data['recon_R'] = img_r.detach().cpu().numpy()
    sio.savemat(result_path + 'save_img_mat.mat', save_data)
    print('image saved as mat')


# >>>>>>>>>>>>>>>>>>>>>>>> initialization >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
args = parser.parse_args()
start_epoch = 0
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"the model is deployed on {device}")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>>>>>>>>>>>>>>>>>> load splited paths >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def load_data_paths(data_paths='./split_dataset_paths.csv'):

    # 读取 CSV 文件
    train_dirs = []
    val_dirs = []
    test_dirs = []

    with open(data_paths, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_dirs.append(row['train'])
            val_dirs.append(row['valid'])
            test_dirs.append(row['test'])

    val_dirs = [val_dir for val_dir in val_dirs if val_dir != '']
    test_dirs = [test_dir for test_dir in test_dirs if test_dir != '']

    return train_dirs, val_dirs, test_dirs
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>>>>>>>>>>>>>>>>>> relevant paths and files >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
result_path = args.Output_path
log_path = args.mother_path + '/Log/'
checkpoint_path = args.mother_path + '/Checkpoint/'
figure_path = args.mother_path + '/Figure/'
# create folder
if not os.path.isdir(args.mother_path):
    os.system('mkdir ' + args.mother_path)
if not os.path.isdir(result_path):
    os.system('mkdir ' + result_path)
if not os.path.isdir(log_path):
    os.system('mkdir ' + log_path)
if not os.path.isdir(checkpoint_path):
    os.system('mkdir ' + checkpoint_path)
if not os.path.isdir(figure_path):
    os.system('mkdir ' + figure_path)
# create log name
rep = 0
stat_name = f'Zdim_{args.zdim}_Vae-beta_{args.vae_beta}_Lr_{args.lr}_Batch-size_{args.batch_size}'
while (os.path.isfile(log_path + stat_name + f'_Rep_{rep}.txt')):
    rep += 1
log_name = log_path + stat_name + f'_Rep_{rep}.txt'
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



# >>>>>>>>>>>>>>>>>>>>>>>> init model and optimizer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
model = BetaVAE(z_dim=args.zdim, nc=1).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# resume
if args.resume:
    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{args.resume}' (epoch { checkpoint['epoch']})")
    else:
        print(f"=> no checkpoint found at '{args.resume}'")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>>>>>>>>> loss function >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def loss_function(xL, xR, x_recon_L, x_recon_R, mu, logvar, beta, left_mask, right_mask):
    Image_Size = xL.size(3)

    beta /= Image_Size ** 2

    # print('====> Image_Size: {} Beta: {:.8f}'.format(Image_Size, beta))

    # R_batch_size=xR.size(0)
    # Tutorial on VAE Page-14
    # log[P(X|z)] = C - \frac{1}{2} ||X-f(z)||^2 // \sigma^2 
    #             = C - \frac{1}{2} \sum_{i=1}^{N} ||X^{(i)}-f(z^{(i)}||^2 // \sigma^2
    #             = C - \farc{1}{2} N * F.mse_loss(Xhat-Xtrue) // \sigma^2
    # log[P(X|z)]-C = - \frac{1}{2}*2*192*192//\sigma^2 * F.mse_loss
    # Therefore, vae_beta = \frac{1}{36864//\sigma^2}
    if left_mask is not None:
        MSE_L = F.mse_loss(x_recon_L * left_mask.detach(), xL * left_mask.detach(), reduction='mean')
        MSE_R = F.mse_loss(x_recon_R * right_mask.detach(), xR * right_mask.detach(), reduction='mean')
    else:  # left and right masks are None
        MSE_L = F.mse_loss(x_recon_L, xL, reduction='mean')
        MSE_R = F.mse_loss(x_recon_R, xR, reduction='mean')

    # KLD is averaged across batch-samples
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()

    return KLD * beta + MSE_L + MSE_R
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>>>>>>>>>>>>>>>>>> train and test func >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def train_save_image_mat(img_r, img_l, recon_r, recon_l, loss, recon_loss, N_Epoch, result_path):
    save_data = {}
    save_data['recon_L'] = recon_l.detach().cpu().numpy()
    save_data['recon_R'] = recon_r.detach().cpu().numpy()
    save_data['img_R'] = img_r.detach().cpu().numpy()
    save_data['img_L'] = img_l.detach().cpu().numpy()
    save_data['Loss'] = loss
    save_data['Recon_Loss'] = recon_loss
    sio.savemat(result_path + '/train_save_img_mat' + str(N_Epoch) + '.mat', save_data)

    print('train image saved as mat')


def test_save_image_mat(img_r, img_l, recon_r, recon_l, loss, recon_loss, N_Epoch, result_path):
    save_data = {}
    save_data['recon_L'] = recon_l.detach().cpu().numpy()
    save_data['recon_R'] = recon_r.detach().cpu().numpy()
    save_data['img_R'] = img_r.detach().cpu().numpy()
    save_data['img_L'] = img_l.detach().cpu().numpy()
    save_data['Loss'] = loss
    save_data['Recon_Loss'] = recon_loss
    sio.savemat(result_path + '/test_save_img_mat' + str(N_Epoch) + '.mat', save_data)

    print('test image saved as mat')


def _train(epoch, train_loader, left_mask, right_mask):
    model.train()
    train_loss = 0
    recon_loss = 0
    # KL_div_loss = 0
    # loss_trace = []

    for batch_idx, (xL, xR) in enumerate(train_loader):
        xL = xL.to(device)
        xR = xR.to(device)
        optimizer.zero_grad()
        x_recon_L, x_recon_R, mu, logvar = model(xL, xR)
        Recon_Error = loss_function(xL, xR, x_recon_L, x_recon_R, mu, logvar, 0, left_mask, right_mask)
        recon_loss += Recon_Error.item()
        
        loss = loss_function(xL, xR, x_recon_L, x_recon_R, mu, logvar, args.vae_beta, left_mask, right_mask)
        loss.backward()
        train_loss += loss.item()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(xL)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'Loss: {loss.item() / len(xL):.6f}\t'
                f'recon_Loss: {Recon_Error.item() / len(xL):.6f}\t'
                f'KLD: {xL.size(3) ** 2 * (loss.item() - Recon_Error.item()) / (args.vae_beta * len(xL)):.6f}'
            )


        # train_save_image_mat(xR, xL,x_recon_R,x_recon_L,loss.item()/len(xL),Recon_Error.item()/len(xL),epoch,result_path)

    stat_file = open(log_name, 'a+')
    stat_file.write(f'Epoch:{epoch} Average training loss: '
                f'{train_loss / batch_idx:.8f} '
                f'Average reconstruction loss: '
                f'{recon_loss / batch_idx:.8f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / batch_idx:.4f}')

    # loss_trace.append(recon_loss)
    # loss_trace.append(KL_div_loss)
    return train_loss / batch_idx, recon_loss / batch_idx


def _test(epoch, test_loader, left_mask, right_mask):
    model.eval()
    test_loss = 0
    recon_loss = 0
    with torch.no_grad():
        for i, (xL, xR) in enumerate(test_loader):
            xL = xL.to(device)
            xR = xR.to(device)
            x_recon_L, x_recon_R, mu, logvar = model(xL, xR)
            test_loss += loss_function(xL, xR, x_recon_L, x_recon_R, mu, logvar, args.vae_beta, left_mask,
                                       right_mask).item()
            recon_loss += loss_function(xL, xR, x_recon_L, x_recon_R, mu, logvar, 0, left_mask, right_mask).item()
            # if i == 0:
            # n = min(xL.size(0), 8)
            # img_len = xL.size(2)
            # left
            # comparisonL = torch.cat([xL[:n],x_recon_L.view(args.batch_size, 1, img_len, img_len)[:n]])
            # save_image(comparisonL.cpu(),figure_path+'reconstruction_left_epoch_' + str(epoch) + '.png', nrow=n)
            # right
            # comparisonR = torch.cat([xR[:n],x_recon_R.view(args.batch_size, 1, img_len, img_len)[:n]])
            # save_image(comparisonR.cpu(),figure_path+'reconstruction_right_epoch_' + str(epoch) + '.png', nrow=n)
            # test_save_image_mat(xR,xL,x_recon_R,x_recon_L,test_loss,recon_loss,epoch,result_path)

    test_loss /= i
    stat_file = open(log_name, 'a+')
    stat_file.write('Epoch:{epoch} Average validation loss: { test_loss:.8f}')
    print(f'====> Test set loss: {test_loss:.4f}')

def train(epoch, train_dirs):
# >>>>>>>>>>>>>>>>>>>>>>>> dataloader >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loss_trace = []
    recon_loss_trace = []
    for train_dir in train_dirs[:10]:

        train_set = H5Dataset(train_dir)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        
        # load the mask from the data loader 
        if args.apply_mask:
            print('Will apply a mask to the loss function')
            left_mask = torch.from_numpy(train_set.LeftMask).to(device)
            right_mask = torch.from_numpy(train_set.RightMask).to(device)
        else:
            print('Will not apply a mask to the loss function')
            left_mask = None
            right_mask = None

        train_loss, recon_loss = _train(epoch, train_loader, left_mask, right_mask)
        train_loss_trace.append(train_loss)
        recon_loss_trace.append(recon_loss)

    print('train one time over all train subjects')
    return  sum(train_loss_trace) / len(train_loss_trace), sum(recon_loss_trace) / len(recon_loss_trace)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def test(epoch, test_dirs):
# >>>>>>>>>>>>>>>>>>>>>>>> dataloader >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    for test_dir in test_dirs[:2]:

        test_set = H5Dataset(test_dir)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        
        # load the mask from the data loader 
        if args.apply_mask:
            print('Will apply a mask to the loss function')
            left_mask = torch.from_numpy(test_set.LeftMask).to(device)
            right_mask = torch.from_numpy(test_set.RightMask).to(device)
        else:
            print('Will not apply a mask to the loss function')
            left_mask = None
            right_mask = None

        _test(epoch, test_loader, left_mask, right_mask)
    print('train one time over all test subjects')
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    

def save_checkpoint(state, filename):
    torch.save(state, filename)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if __name__ == "__main__":
    train_dirs, val_dirs, test_dirs = load_data_paths()
    test(0, val_dirs)
    train_loss_record, recon_loss_record = [], []
    loss_record = {}
    for epoch in range(start_epoch + 1, args.epochs):
        train_loss, recon_loss = train(epoch, train_dirs)
        train_loss_record.append(train_loss)
        recon_loss_record.append(recon_loss)
        test(epoch, val_dirs)
        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path + '/checkpoint' + str(epoch) + str(args.vae_beta) + '.pth.tar')
        scheduler.step()

    loss_record['train_loss_record'] = train_loss_record
    loss_record['recon_loss_record'] = recon_loss_record
    sio.savemat('loss_record.mat', loss_record)

    import matplotlib.pyplot as plt
    plt.plot(range(1, args.epochs + 1), train_loss_record, label='train loss')
    plt.plot(range(1, args.epochs + 1), recon_loss_record, label='recon loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss.pdf')



