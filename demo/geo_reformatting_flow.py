% 指定目录路径
directory = '~/Datasets/HCP_S1200/data/';

% 获取目录下所有子目录
subdirectories = dir(directory);
subdirectories = subdirectories([subdirectories.isdir]); % 仅保留目录项
subdirectories = subdirectories(~ismember({subdirectories.name}, {'.', '..'})); % 移除 '.' 和 '..' 目录

% 初始化存储所有 .nii 文件路径的单元数组
nii_paths = {};

% 遍历每个子目录
for i = 1:length(subdirectories)
    subdir_path = fullfile(directory, subdirectories(i).name);
    
    % 获取当前子目录下所有 .nii 文件的路径
    nii_files = dir(fullfile(subdir_path, '*.nii'));
    
    % 将每个 .nii 文件的完整路径添加到 nii_paths
    for j = 1:length(nii_files)
        nii_paths{end+1} = fullfile(subdir_path, nii_files(j).name);
    end
end

for k = 1:length(nii_paths)
    % preprocess_fMRI(nii_paths{k})
    geometric_reformatting(nii_paths{k})
end
