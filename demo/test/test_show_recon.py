import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio

print(matplotlib._get_version())
img = sio.loadmat('../data/100408_REST1LR/recon_img/img0.mat')


recon_L = img['recon_L']
recon_R = img['recon_R']

plt.imshow(recon_L[0, 0, :, :], cmap='coolwram')
plt.show()