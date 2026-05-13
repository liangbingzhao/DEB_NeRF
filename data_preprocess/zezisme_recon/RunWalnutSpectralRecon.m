function RunWalnutSpectralRecon(walnut_name)
% RunWalnutSpectralRecon — wrapper for ImageSpectralRecon (Pipeline 2: VMI + MD).
% Loads existing Low + High DICOMs from Reconstructions_mat/, runs material
% decomposition + virtual mono-energy synthesis, saves to same dir.
%
% Usage (from MATLAB or batch):
%   RunWalnutSpectralRecon('Walnut_1')
%   RunWalnutSpectralRecon('Walnut_2')

reset(gpuDevice());

% Add TIGRE toolbox + zezisme functions
addpath(genpath('/ibex/project/c2272/liangbing/cs300_project/SAX-NeRF/TIGRE-2.3/MATLAB'));
addpath(genpath(pwd));

% Project paths — exact match to RunWalnutRecon
project_root  = '/ibex/project/c2272/liangbing/cs300_project/data_preprocess';
data_dir_root = [project_root, '/Reconstructions_mat'];
save_path     = [project_root, '/Reconstructions_mat'];
cali_path     = [project_root, '/CalibrationTable'];

% Recon params — match WalnutSpectralRecon.m EXACTLY
recon_para.WalnutMD_Enable  = 1;
recon_para.WalnutVMI_Enable = 1;
recon_para.WalnutVMI_E      = 10:10:80;

input_dir  = [data_dir_root, '/', walnut_name, '/FDK_Dose_1_hann_TV_100_20'];
output_dir = [save_path,     '/', walnut_name, '/FDK_Dose_1_hann_TV_100_20'];
fprintf('=== RunWalnutSpectralRecon for %s ===\n', walnut_name);
fprintf('  input:  %s\n', input_dir);
fprintf('  output: %s\n', output_dir);
fprintf('=========================================\n');

ImageSpectralRecon(input_dir, output_dir, cali_path, recon_para);

fprintf('=== RunWalnutSpectralRecon DONE for %s ===\n', walnut_name);
end
