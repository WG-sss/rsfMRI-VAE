function eigen_state_reformatting(cii_output_filepath)
    %%  
    % backward projection from the VAE reconstruction to the cortex
    
    %% Configuration
    % batchsize = 120;
    batchsize = 512;
    addpath('./CIFTI_read_save');
    eigen_path = '../temp/';
    disp('start')
    % if isempty(inverse_transformation_path)
    %     disp('inverse is empty')
    %     inverse_transformation_path = './data/100408_REST1LR/';
    %     disp(inverse_transformation_path)
    % end
    % if isempty(cii_template_filepath)
    %     cii_template_filepath = './data/100408_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_preprocessed.dtseries.nii';
    % end
    % cii_output_filepath = './data/100408_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_preprocessed_2';
    
    inverse_transformation_path = './data/100408_REST1LR/';
    cii_template_filepath = './data/100408_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_preprocessed.dtseries.nii';

    %% load the inverse transformation matrix
    
    % load the inverse transformation matrix of geometric reformatting, as in JH's Recon_fMRI_Generator.m
    load([inverse_transformation_path 'Left_fMRI2Grid_192_by_192_NN.mat'], 'inverse_transformation');
    Left_inverse_transformation = inverse_transformation;
    load([inverse_transformation_path 'Right_fMRI2Grid_192_by_192_NN.mat'], 'inverse_transformation');
    Right_inverse_transformation = inverse_transformation;
    
    %% backward projection
    recon_dtseries = zeros(59412, 512);
    load([eigen_path 'eigen_recon_img_dis_4.mat'], 'eigen_L', 'eigen_R');
    % (9, 1, 192, 192) --permute & reshape & tranpose-->
    % (29696, 36864) * (36864,9) -> (29696, 9)
    corticalrecon_L = Left_inverse_transformation * double(reshape(permute(eigen_L,[1,2,4,3]),batchsize, [])');
    % (29716, 36864) * (36864,9) -> (29716, 9)
    corticalrecon_R = Right_inverse_transformation * double(reshape(permute(eigen_R,[1,2,4,3]),batchsize, [])');
    recon_dtseries(:, 1:batchsize) = [corticalrecon_L; corticalrecon_R];
    
    %% save the reconstruction back into cifti file
    % read in original data with fieldtrip toolbox
    % loaded in as a struc
    cii = ft_read_cifti(cii_template_filepath);
    % extract time-series data from left and right cortex (regions 1,2)
    cortex_dtseries = cii.dtseries((cii.brainstructure == 1 | cii.brainstructure == 2), :);
    
    % fill the normalized data into the correct index of the cifti data
    cortex_dtseries(~isnan(cortex_dtseries(:,1)), 1:512) = recon_dtseries;
    cortex_dtseries(~isnan(cortex_dtseries(:,1)), 512:end) = 0;
    cii.dtseries((cii.brainstructure == 1 | cii.brainstructure == 2), :) = cortex_dtseries;
    cii.dtseries(~(cii.brainstructure == 1 | cii.brainstructure == 2), :) = NaN;
    
    % save the preprocessed data
    ft_write_cifti(cii_output_filepath, cii, 'parameter', 'dtseries');

end
