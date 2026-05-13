function RunWalnutRecon(walnut_name)
% RunWalnutRecon — wrapper for ReconAllEnergy with our project paths.
%   Replicates WalnutDataRecon.m but takes walnut name as argument and
%   uses Linux-style paths matching our data layout.
%
% Usage (from MATLAB or batch):
%   RunWalnutRecon('Walnut_1')
%   RunWalnutRecon('Walnut_2')
%   RunWalnutRecon('Walnut_3')

% Reset GPU and clear state
reset(gpuDevice());

% Add TIGRE toolbox
addpath(genpath('/ibex/project/c2272/liangbing/cs300_project/SAX-NeRF/TIGRE-2.3/MATLAB'));

% Add zezisme functions to path
addpath(genpath(pwd));

% Project paths — Linux absolute
project_root  = '/ibex/project/c2272/liangbing/cs300_project/data_preprocess';
data_dir_root = project_root;                                 % parent of Walnut_X
% Output suffix _mat indicates MATLAB pipeline (vs _python or original GT)
save_path     = [project_root, '/Reconstructions_mat'];

% Recon parameters — matching original WalnutDataRecon.m EXACTLY
recon_para.CaliTablePath     = [project_root, '/CalibrationTable'];
recon_para.NonUniformityCorr = 1;        % STEPC algorithm
recon_para.RingArtifactCorr  = 1;        % ring artifact correction
recon_para.recon_type        = 2;        % 1=FDK, 2=FDK+TV
recon_para.FDK_filter        = 'hann';   % per original
recon_para.TV_niter          = 100;
recon_para.TV_lambda         = 20;
recon_para.dose_ratio        = 1;        % full dose
recon_para.recon_Bin         = [1 1 1];  % [Low, High, Total] — all three
recon_para.nVoxel            = [1000;1000;300];
recon_para.sVoxel            = [50;50;15];   % mm
recon_para.is_write2dicom    = 1;             % save as DICOM

% Make output dir
if ~exist(save_path, 'dir')
    mkdir(save_path);
end

input_dir  = [data_dir_root, '/', walnut_name];
output_dir = [save_path,     '/', walnut_name];
fprintf('=== RunWalnutRecon for %s ===\n', walnut_name);
fprintf('  input:  %s\n', input_dir);
fprintf('  output: %s\n', output_dir);
fprintf('  TIGRE: %s\n', fileparts(which('FDK')));
fprintf('=================================\n');

ReconAllEnergy(input_dir, output_dir, recon_para);

fprintf('=== RunWalnutRecon DONE for %s ===\n', walnut_name);
end
