import nibabel as nib

data1 = nib.load("../data/rfMRI_REST1_LR_Atlas_hp2000_clean_vn.dscalar.nii")
data2 = nib.load("../data/rfMRI_REST1_LR_Atlas_hp2000_clean_bias.dscalar.nii")
data3 = nib.load("../data/rfMRI_REST1_LR_Atlas_stats.dscalar.nii")

myelin_data1 = data1.get_fdata().T
myelin_data2 = data2.get_fdata().T
myelin_data3 = data3.get_fdata().T
print('数据维度:', myelin_data1.shape)
print('数据维度:', myelin_data2.shape)
print('数据维度:', myelin_data3.shape)

# axes = [dteries.header.get_axis(i) for i in range(dteries.ndim)]
# 获取头文件信息
# zooms = dteries.header.get_zooms()
# print('像素尺寸:', zooms[:3])
# print('仿射矩阵:\n', dteries.affine)
# dt = zooms[-1]
# print('TR:', '{:.2f} s'.format(dt))

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