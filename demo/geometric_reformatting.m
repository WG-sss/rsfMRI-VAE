function geometric_reformatting(Preprocessed_fMRI_filepath)
  %%
  % Geometric reformatting to get the transformation from cortex to grid data
  % para:
  %   - Preprocessed_fMRI_filepath: as the name

  % TODO: merge ID and run information
  %% Configurations
  addpath('./lib');
  addpath('./CIFTI_read_save');
  
  % set the size of the output image
  im_size = 192;

  % check the validity of input parameter
  if ~exist(Preprocessed_fMRI_filepath, 'file')
    fprintf('Error: The file %s does not exist.\n', Preprocessed_fMRI_filepath);
    return;
  end
  
  % there is no suffix, attention.
  % Preprocessed_fMRI_filepath = '~/Datasets/HCP_S1200/data/100307/100307_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_preprocessed';
  
  % ------------------------- split the ID and other character -------------------------
  % Extract the parent directory and baseFileName of the preprocessed file
  [parentDir,  baseFileName, ~] = fileparts(Preprocessed_fMRI_filepath);

  % Split the base file name using underscore as delimiter
  parts = split(baseFileName, '_');
  ID = parts{1};
  run = [parts{3} parts{4}]; % for example 'REST1_LR'

  % creating result directory
  resultDirPath = fullfile(parentDir, [ID '_' run]);

  % Check if the 'result' directory already exists
  if ~exist(resultDirPath, 'dir')
      % Create the 'result' directory
      mkdir(resultDirPath);
      disp(['Created directory: ', resultDirPath]);
  else
      disp(['Directory already exists: ', resultDirPath]);
  end

  % -----------------------------------------------------------------------------------
  % -------------------------- read data and process ----------------------------------

  LSphere_filepath = './template/Q1-Q6_R440.L.sphere.32k_fs_LR.surf.gii';
  RSphere_filepath = './template/Q1-Q6_R440.R.sphere.32k_fs_LR.surf.gii';

  %% load coordanite values from spherical templates
  % read in left/right hemisphere spherical templates (gifti geometry format)
  lb = gifti(LSphere_filepath);
  rb = gifti(RSphere_filepath);

  %% load data
  % read in preprocessed data with fieldtrip toolbox
  cii = ft_read_cifti(Preprocessed_fMRI_filepath);

  % extract time-series data from left and right visual cortex (regions 1,2)
  % fMRI dimensions: num voxels x num time points
  fMRI = cii.dtseries((cii.brainstructure == 1 | cii.brainstructure == 2),:);
  Left_fMRI = cii.dtseries((cii.brainstructure == 1),:);
  Right_fMRI = cii.dtseries((cii.brainstructure == 2),:);


  % -----------------------------------------------------------------------------------
  % -------------------------- sample a data from fMRI --------------------------------
  % -------------------- as a template for the dimensional info  ----------------------
  % extract the first time point from fMRI
  Sample_Data = fMRI(:,1);
  Sample_Data_Left = Left_fMRI(:,1);
  Sample_Data_Right = Right_fMRI(:,1);

  %% get voxels in 59412 space
  % create logical vector corresponding to which voxels have nan value
  % voxels with nan value are not valid values
  voxel_valid = ~isnan(Sample_Data);
  voxel_valid_L = ~isnan(Sample_Data_Left);
  voxel_valid_R = ~isnan(Sample_Data_Right);

  % remove nan values from the sample data
  % Sample_Data_no_nan = Sample_Data(voxel_valid);
  Sample_Data_no_nan_Left = Sample_Data_Left(voxel_valid_L);
  Sample_Data_no_nan_Right = Sample_Data_Right(voxel_valid_R);

  % ---------------------------- save some useful information -----------------------------
  %% save Normalized_fMRI as a mat file
  Normalized_fMRI = fMRI(voxel_valid,:);
  save([resultDirPath '/fMRI.mat'], 'Normalized_fMRI');
  % ---------------------------------------------------------------------------------------


  % ---------------------------- Transform nii data to grid form --------------------------
  % calculate the azimuth and elevation coordinates for each voxel
  % do only for the valid voxels with non-nan values
  [L_az_nonan, L_el_nonan, R_az_nonan, R_el_nonan] = Dimension_Reduction_Surface(voxel_valid,lb,rb); 

  %% Mask generation
  % initialize mask vectors for left/right cortex
  Left_Mask = ones(length(Sample_Data_Left),1);
  Right_Mask = ones(length(Sample_Data_Right),1);
  % input a zero in rows that correspond to voxels with nan value
  Left_Mask(isnan(Sample_Data_Left))=0;
  Right_Mask(isnan(Sample_Data_Right))=0;

  % create a vector of ones to pass into Dimension_Reduction_Surface
  voxel_all = true(length(voxel_valid),1);

  % same calculation of azimuth and elevation coordinates for each voxel
  % but do for all the voxels (with/wihout nan values)
  [L_az, L_el, R_az, R_el] = Dimension_Reduction_Surface(voxel_all,lb,rb); 

  % transform angles and create grid (for data with nan values)
  [T_L_az, T_L_el, T_R_az, T_R_el, X, Y] = Create_Grid(im_size, L_az, L_el, R_az, R_el);

  % generate L/R masks for im_size x im_size grid
  [Regular_Grid_Left_Mask, Regular_Grid_Right_Mask] = Mask_Generation(im_size, Left_Mask, Right_Mask, T_L_az, T_L_el, T_R_az, T_R_el, X, Y);

  % save the masks for the im_size x im_size grid
  save([resultDirPath '/MSE_Mask.mat'],'Regular_Grid_Left_Mask','Regular_Grid_Right_Mask');

  %% grid mapping
  % transform angles and create grid (for data without nan values)
  [T_L_az_nonan, T_L_el_nonan, T_R_az_nonan, T_R_el_nonan, X, Y] = Create_Grid(im_size, L_az_nonan, L_el_nonan, R_az_nonan, R_el_nonan);

  % generate map for voxel data to 2D grid and its inverse map for L hemi
  % save the grid mapping and inverse grid mapping for L hemi
  [grid_mapping, inverse_transformation, ~, ~] = Geometric_Reformatting_fMRI2Grid_NN(im_size, T_L_az_nonan, T_L_el_nonan, X,Y, Sample_Data_no_nan_Left);
  save([resultDirPath '/Left_fMRI2Grid_',num2str(im_size),'_by_',num2str(im_size),'_NN.mat'],'grid_mapping','inverse_transformation')

  % generate map for voxel data to 2D grid and its inverse map for R hemi
  % save the grid mapping and inverse grid mapping for R hemi
  [grid_mapping, inverse_transformation, ~, ~] = Geometric_Reformatting_fMRI2Grid_NN(im_size, T_R_az_nonan, T_R_el_nonan, X,Y, Sample_Data_no_nan_Right);
  save([resultDirPath '/Right_fMRI2Grid_',num2str(im_size),'_by_',num2str(im_size),'_NN.mat'],'grid_mapping','inverse_transformation')


  % ---------------------------------------------------------------------------------------

  % ---------------------------------------------------------------------------------------




  % % -------------------------- Visualization ----------------------------------------------
  % %% visualization
  % % for visualization, plot the data for the first time point for L
  % fig = figure;
  % title('2D image of cortical pattern (L)');
  % imagesc(reshape(transformed_gridmap_L, im_size,im_size));
  % save(fig, '')
  % % 使用自定义调色板
  % colormap(slanCM('coolwarm'));
  % disp(['Loss rate of reformatting and inverse-reformatting procedures (L) is ',num2str(Loss_Rate_L),'%'])
  %
  % % for visualization, plot the data for the first time point for R
  % fig = figure;
  % title('2D image of cortical pattern (R)');
  % imagesc(reshape(transformed_gridmap_R, im_size,im_size))
  % save(fig, '')
  % colormap(slanCM('coolwarm'));
  % disp(['Loss rate of reformatting and inverse-reformatting procedures (R) is ',num2str(Loss_Rate_R),'%'])
  % % ---------------------------------------------------------------------------------------

  %% generate hdf5 format for Pytorch dataloader
  % data_prep.py driver
  % for remote server
  system(['/opt/homebrew/Caskroom/miniconda/base/envs/rsfMRI-VAE/bin/python ./lib/data_prep.py --fmri-path ' resultDirPath ' --trans-path ' resultDirPath ' --output-path ' resultDirPath]);
end
