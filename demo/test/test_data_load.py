import torch.utils.data
import h5py
import torch
import torch.utils.data as data
import torch.multiprocessing


# torch.multiprocessing.set_start_method('spawn')

class H5Dataset(data.Dataset):
    def __init__(self, H5Path):
        super(H5Dataset, self).__init__()
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

datapath = '/Users/xiaowangzi/Projects/rsfMRI-VAE/demo/data/100408_REST1LR/transformed_fMRI_data.h5'

train_set = H5Dataset(datapath)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)

for batch_idx, (xL, xR) in enumerate(train_loader):
    print(batch_idx)
    print(xL.shape)
    print(xR.shape)
