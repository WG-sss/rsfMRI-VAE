function preprocessed_fMRI(cii_input_filepath)
  % preprocess fMRI data for the VAE
  % para: 
  %   -cii_input_filepath: the raw fMRI data path
  if ~exist(cii_input_filepath, 'file')
    fprintf('Error: The file %s does not exist.\n', cii_input_filepath);
    return;
  end

  %% Configuration
  addpath('./lib');
  addpath('./CIFTI_read_save');

  % check input path and create output path
  if endsWith(cii_input_filepath, '.dtseries.nii')
      % Remove the '.dtseries.nii' suffix
      base_path = erase(cii_input_filepath, '.dtseries.nii');
      % Append 'preprocessed' to the base path
      cii_output_filepath = [base_path, '_preprocessed']
  else
      error('Input path does not end with ''.dtseries.nii''');
  end

  %% test
  % cii_input_filepath = './data/100408_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii';
  % cii_output_filepath = './data/100408_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_preprocessed';

  %% for remote server
  % cii_input_filepath = '~/Datasets/HCP_S1200/data/100307/100307_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii';
  % cii_output_filepath = '~/Datasets/HCP_S1200/data/100307/100307_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_preprocessed';

  %% Preprocess
  % sampling frequency of HCP fMRI data
  Fs = 1/0.72; 

  % read in original data with fieldtrip toolbox
  % loaded in as a struc
  cii = ft_read_cifti(cii_input_filepath);

  % extract time-series data from left and right cortex (regions 1,2)
  cortex_dtseries = cii.dtseries((cii.brainstructure == 1 | cii.brainstructure == 2), :);
  cortex_nonan_dtseries = cortex_dtseries(~isnan(cortex_dtseries(:,1)), :); % 59412 dimensional

  % detrend and filter the data
  Normalized_cortex_nonan_dtseries = Detrend_Filter(cortex_nonan_dtseries,Fs);

  % fill the normalized data into the correct index of the cifti data
  cortex_dtseries(~isnan(cortex_dtseries(:,1)), :) = Normalized_cortex_nonan_dtseries;
  cii.dtseries((cii.brainstructure == 1 | cii.brainstructure == 2), :) = cortex_dtseries;

  % save the preprocessed data
  ft_write_cifti(cii_output_filepath, cii, 'parameter', 'dtseries');
end
