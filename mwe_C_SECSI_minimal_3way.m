% Minimal working example for coupled CP decomposition of three-way
% tensors.
%
% |----------------------------------------------------------------
% | (C) 2020 Mikus Grasis
% |
% |         __          __               ______            __
% |        / /   ____ _/ /____  _  __   /_  __/___  ____  / /____
% |       / /   / __ `/ __/ _ \| |/_/    / / / __ \/ __ \/ / ___/
% |      / /___/ /_/ / /_/  __/>  <     / / / /_/ / /_/ / (__  )
% |     /_____/\__,_/\__/\___/_/|_|    /_/  \____/\____/_/____/
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |
% |     Date authored: 05.07.2020
% |     Modifications:
% |     05.07.2020 - initial version (MG)
% |----------------------------------------------------------------
%
% Notation: number of tensors L, tensor order N, rank R, dimensions I
%
clear; close all; clc
rng(7, 'twister'); %<-- set seed for reproducibility

%% Dependencies
% addpath(genpath(fullfile('.', '_toolboxes', 'tensor_toolbox')))
addpath(genpath(fullfile('.', 'algorithm')))
addpath(genpath(fullfile('.', 'utils')))

%% Create Problem
L = 2;                  % number of tensors
N = 3;                  % tensor order
I = cell(1, L);
I{1} = [3, 4, 5];       % size of tensor 1
I{2} = [3, 4, 5];       % size of tensor 2 (must agree along coupled mode)
R = 3;                  % model rank
 
% Generate i.i.d. random factor matrices with coupling along 1-st mode
coupling.modes = 1;
coupling = parse_coupling(I, R, coupling); % verify dimensions

opts_fac.factors_type = 'gaussian';
F = generate_coupled_random_factors(I, R, coupling, opts_fac);

% Construct coupled CP-model tensors
Xt_0 = cell(1, L);
for l = 1:L
    Xt_0{l} = cp_construct(F{l});
end

%% Add Noise
sn = 42;                % snr in dB
Xt = cell(1, L);
for l = 1:L
    Xt{l} = Xt_0{l} + generate_noise_tensor(tensor_power(Xt_0{l}), I{l}, sn, 'gaussianIID');
end

%% Solve Problem
opts = [];

tic
[F_hat, infos] = C_SECSI_minimal_3way(Xt, R, coupling, opts);
toc

for l = 1:L
    fprintf('\nRelative squared error: %.6e\n', relative_squared_error(Xt_0{l}, cp_construct(F_hat{l})));
    
    fprintf('Factor error: %.6e\n', sum(cpderr_RSE(F{l}, F_hat{l})));

%     % show FMS score (requires tensor_toolbox on path)    
%     [F_l, lambda_l] = normalize_factors(F{l});
%     [F_hat_l, lambda_hat_l] = normalize_factors(F_hat{l});
%     fprintf('FMS: %1.3f\n', score(ktensor(lambda_hat_l, F_hat_l), ktensor(lambda_l, F_l)));
    
    str_rec_errs = sprintf('%1.2e, ', infos.rec_err(l, :));
    fprintf('Reconstruction errors: %s\n', str_rec_errs);
    
    [~, selected_SMD] = min(infos.rec_err(l, :));
    fprintf('Selected SMD: %d\n', selected_SMD);
end

%% EoF
disp('script finished successfully...')
