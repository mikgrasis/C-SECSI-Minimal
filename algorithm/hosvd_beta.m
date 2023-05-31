function [St, U, SIGMA] = hosvd_beta(Xt, R, opts)
% Computes the truncated higher-order singular value decomposition (HOSVD)
% of a tensor.
%
% |----------------------------------------------------------------
% | (C) 2021 TU Ilmenau, Communications Research Laboratory
% |
% |
% |  ______   ______     __   __     ______     ______     ______        ______   ______     ______     __         ______
% | /\__  _\ /\  ___\   /\ "-.\ \   /\  ___\   /\  __ \   /\  == \      /\__  _\ /\  __ \   /\  __ \   /\ \       /\  ___\
% | \/_/\ \/ \ \  __\   \ \ \-.  \  \ \___  \  \ \ \/\ \  \ \  __<      \/_/\ \/ \ \ \/\ \  \ \ \/\ \  \ \ \____  \ \___  \
% |    \ \_\  \ \_____\  \ \_\\"\_\  \/\_____\  \ \_____\  \ \_\ \_\       \ \_\  \ \_____\  \ \_____\  \ \_____\  \/\_____\
% |     \/_/   \/_____/   \/_/ \/_/   \/_____/   \/_____/   \/_/ /_/        \/_/   \/_____/   \/_____/   \/_____/   \/_____/
% |
% |
% |     Mikus Grasis
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |         Prof. Andre Lima Ferrer de Almeida
% |
% |     Date Authored: 16.07.2019
% |     Modifications:
% |     27.06.2022 - nicer handling of dimension ordering (MG)
% |     11.05.2022 - add support for subspace_estimation_opts (MG)
% |     05.04.2022 - remove 'opts.truncation_method' (MG)
% |     04.04.2022 - move subspace estimation to separate function (MG)
% |     29.11.2021 - new method 'EIGS' (MG)
% |     26.04.2021 - code review (MG)
% |     18.07.2020 - initial version (MG)
% |----------------------------------------------------------------
%
% [St, U] = hosvd_beta(Xt, R)
%
% computes the (truncated) higher-order singular value decomposition of Xt,
% such that Xt = St x_1 U{1} ... x_N U{N}.
%
% Inputs:
%   Xt      - tensor (MATLAB multidimensional array)
%   R       - multilinear rank R(1),...,R(N) for truncation
%   opts    - struct with options
%       opts.dim_order = 'descending'; % sequential | descending
%       opts.svd_method = 'EIG'; % SVD | SVDS | EIG | EIGS | RSVD
%       opts.subspace_estimation_opts - this is passed on to subspace_estimation.m
%
% Outputs:
%   St      - core tensor
%   U       - cell array with matrices of eigenvectors
%   SIGMA   - cell array with vectors of singular values
%
% References:
% [1] L. De Lathauwer, B. De Moor, and J. Vandewalle, "A Multilinear
%     Singular Value Decomposition, " SIAM J. Matrix Anal. Appl., vol. 21,
%     no. 4, pp. 1253-1278, 2000.
% [2] N. Halko, P. G. Martinsson, and J. A. Tropp, "Finding Structure with
%     Randomness: Probabilistic Algorithms for Constructing Approximate
%     Matrix Decompositions, " SIAM Rev., vol. 53, no. 2, pp. 217-288, Jan. 2011.
if ~isa(Xt, 'double')
    error('HOSVD_beta:input_check', 'input must be of type MATLAB double');
end

if nargin < 3, opts = []; end
if nargin < 2, R = []; end

opts.dim_order  = setparam(opts, 'dim_order', 'sequential'); % sequential | descending
opts.svd_method = setparam(opts, 'svd_method', 'EIG'); % SVD | SVDS | EIG | EIGS | RSVD
opts.subspace_estimation_opts = setparam(opts, 'subspace_estimation_opts', []);

I = size(Xt);
N = length(I);

% check multilinear ranks for truncation
if isempty(R) % no rank specified == full HOSVD
    R = size(Xt);
elseif eq(length(R), 1) % we can also pass just one rank that is used for all dimensions
    R = ones(N, 1) * R;
elseif ne(length(R), N) % the normal case (multilinear rank for truncation)
    error('HOSVD_beta:input_check', 'multilinear ranks must match number of dimensions')
end

% compute SVDs of unfoldings
SIGMA = cell(N, 1);
U = cell(N, 1);
for n = 1:N
    Xunf = unfolding(Xt, n, 1);
    [U{n}, SIGMA{n}] = subspace_estimation(Xunf, R(n), opts.svd_method, opts.subspace_estimation_opts);
end

% choose dimension ordering for core tensor computation
if strcmp(opts.dim_order, 'descending')
    [~, dim_idx] = sort(I, 'descend'); % start with largest dimension
else
    dim_idx = 1:N;
end

% compute core tensor
St = Xt;
for n = 1:N
    St = nmode_product(St, U{dim_idx(n)}', dim_idx(n));
end
end
