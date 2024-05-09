import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting

origin_data = nib.load('../data/test_origin_with_NaN/100408_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_preprocessed_withNan.dtseries.nii')
recon_data = nib.load('../data/test_origin_with_NaN/100408_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_reconstruction.dtseries.nii')

origin_dtseries = origin_data.get_fdata()
recon_dtseries = recon_data.get_fdata()
print(origin_dtseries.shape)

origin_point = origin_dtseries[:100, 8662]
recon_point = recon_dtseries[:100, 8662]

plt.plot(origin_point, c='black')
plt.plot(recon_point, c='red')
plt.show()

# 读取 dtseries.nii 文件
# img = nib.load('../data/100408_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_reconstruction_test.dtseries.nii')
# axes = [img.header.get_axis(i) for i in range(img.ndim)]
# print(len(axes[1].name)) # 96854
#
# for i in range(29696):
#     print(axes[1].name[i])
#
# print(axes[1].name[29697])
# for i in range(len(axes[1])):
#     # if axes[1][i][2] == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
#         print(axes[1][i])
# print(axes[1][0][2])
# 创建nilearn中的Nifti1Image对象
# dtseries = img.get_fdata()
# dtseries = np.asarray(dtseries)
# print(dtseries.shape)
# print(np.nanmax(dtseries), np.nanmin(dtseries))
# print(np.sum(np.abs(dtseries) < 3) / 1200)
# plt.plot(dtseries[:, :3])
# plt.show()

# 可视化数据
# plotting.plot_epi(dtseries)

# 显示图像
# plotting.show()

# data1 = nib.load("../data/rfMRI_REST1_LR_Atlas_hp2000_clean_vn.dscalar.nii")
# data2 = nib.load("../data/rfMRI_REST1_LR_Atlas_hp2000_clean_bias.dscalar.nii")
# data3 = nib.load("../data/rfMRI_REST1_LR_Atlas_stats.dscalar.nii")
#
# myelin_data1 = data1.get_fdata().T
# myelin_data2 = data2.get_fdata().T
# myelin_data3 = data3.get_fdata().T
# print('数据维度:', myelin_data1.shape)
# print('数据维度:', myelin_data2.shape)
# print('数据维度:', myelin_data3.shape)

# axes = [dteries.header.get_axis(i) for i in range(dteries.ndim)]
# 获取头文件信息
# zooms = dteries.header.get_zooms()
# print('像素尺寸:', zooms[:3])
# print('仿射矩阵:\n', dteries.affine)
# dt = zooms[-1]
# print('TR:', '{:.2f} P'.format(dt))

# # 获取影像对应的矩阵
# bold_data = dteries.get_fdata()
# print('数据维度:', bold_data.shape)
# print(axes)
# print(axes[1].affine)
#
# scalar, brain_model_axis = axes
#
# # cc_mask = (brain_model_axis.name == 'CIFTI_STRUCTURE_CORTEX_LEFT') + (
# #             brain_model_axis.name == 'CIFTI_STRUCTURE_CORTEX_RIGHT')  # cerebral cortex
# # ncc_mask = (1 - cc_mask).astype(bool)  # no cerebral cortex
#
# dteries_t = dteries.get_fdata()
# print(dteries_t.shape)
# # dteries_t[:, cc_mask] = (dteries_t[:, cc_mask] - dteries_t[:, cc_mask].mean(axis=0)) / dteries_t[:, cc_mask].std(
#     axis=0)  # scaling
# dteries_t[:, ncc_mask] = 0  # non cerebral cortex set to 0

# cii = nib.Cifti2Image(dataobj=dteries_t, header=dteries.header)
# cii.to_filename("demean.dtseries.nii")