%%  
% backward projection from the VAE reconstruction to the cortex
% we have replace NaN value with 0, because NaN can't be processed by VAE
% After reconstruction, we should replace back 0 with NaN for better
% visualization in wb_view.


cii_input_file = './data/test_origin_with_NaN/100408_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_preprocessed.dtseries.nii';
cii_output_file = './data/test_origin_with_NaN/100408_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_preprocessed_withNan';

%% save the reconstruction back into cifti file
% read in original data with fieldtrip toolbox
% loaded in as a struc
cii = ft_read_cifti(cii_input_file);
cii.dtseries(~(cii.brainstructure == 1 | cii.brainstructure == 2), :) = NaN;

% save the preprocessed data
ft_write_cifti(cii_output_file, cii, 'parameter', 'dtseries');
