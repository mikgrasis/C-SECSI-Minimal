function [St, U, SIGMA] = st_hosvd_beta(Xt, R, opts)
% Computes sequentially truncated higher-order singular value decomposition
% (ST-HOSVD) of a tensor.
%
% |----------------------------------------------------------------
% | (C) 2022 TU Ilmenau, Communications Research Laboratory
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
% |
% |     Date Authored: 11.04.2022
% |     Modifications:
% |     08.09.2022 - include explicit dimension ordering (MG)
% |     20.08.2022 - update handling of dimension orderings (MG)
% |     11.04.2022 - initial version (MG)
% |----------------------------------------------------------------
%
% [St, U] = st_hosvd_beta(Xt, R)
%
% computes the sequentially truncated higher-order singular value
% decomposition (ST-HOSVD) of Xt, such that Xt \approx St x_1 U{1} ... x_N U{N}.
%
% Inputs:
%   Xt      - tensor (MATLAB multidimensional array)
%   R       - multilinear rank R(1),...,R(N) for truncation
%   opts    - struct with options
%   	opts.dim_order = 'descending'; % sequential | descending | ascending | random
%       opts.svd_method = = 'EIG'; % SVD | SVDS | EIG | EIGS | RSVD
% Outputs:
%   St      - core tensor
%   U       - cell array with matrices of eigenvectors
%   SIGMA   - cell array with vectors of singular values
%
% References:
% [1] N. Vannieuwenhoven, R. Vandebril, and K. Meerbergen, "A New Truncation
%   Strategy for the Higher-Order Singular Value Decomposition," SIAM J. Sci.
% Comput., vol. 34, no. 2, pp. A1027-A1052, Jan. 2012.
%
if ~isa(Xt, 'double')
    error('ST_HOSVD_beta:input_check', 'input must be of type MATLAB double');
end

if nargin < 3, opts = []; end
if nargin < 2, R = []; end

opts.dim_order = setparam(opts, 'dim_order', 'sequential'); % sequential | descending | ascending | random
opts.svd_method = setparam(opts, 'svd_method', 'EIG'); % SVD | SVDS | EIG | EIGS | RSVD
opts.subspace_estimation_opts = setparam(opts, 'subspace_estimation_opts', []);
opts.subspace_estimation_opts.randomization_method = setparam(opts.subspace_estimation_opts, 'randomization_method', []);

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

% determine dimension ordering
if strcmp(opts.dim_order, 'sequential')
    dim_idx = 1:N;
    
elseif strcmp(opts.dim_order, 'descending')
    [~, dim_idx] = sort(I, 'descend');
    
elseif strcmp(opts.dim_order, 'ascending')
    [~, dim_idx] = sort(I, 'ascend');
    
elseif strcmp(opts.dim_order, 'random')
    dim_idx = randperm(N);
    
elseif isnumeric(opts.dim_order) % explicit dimension ordering
    assert(eq(length(opts.dim_order), N))
    dim_idx = opts.dim_order;
        
else
    error('ST_HOSVD:input_check', 'unknown dimension ordering')
end

% compute products regarding the size of unfoldings
I_breve = nan(1, N);
for n = 1:N
    I_breve(dim_idx(n)) = prod(R(dim_idx(1:n-1))) * prod(I(dim_idx(n+1:N))); % each unfolding is of size I(n) x I_breve(n)
end

U = cell(1, N); SIGMA = cell(1, N);
Yt = Xt;
for n = 1:N
    % adjust sampling factor (to match amount of data to first unfolding)
    if strcmp(opts.subspace_estimation_opts.randomization_method, 'sampling_factor')
        if n > 1
            prev_unfolding_size = I(dim_idx(n-1)) * I_breve(dim_idx(n-1));
            fprintf('previous unfolding size: %d\n', prev_unfolding_size)
            curr_unfolding_size = I(dim_idx(n)) * I_breve(dim_idx(n));
            fprintf('current unfolding size: %d\n', curr_unfolding_size)
            opts.subspace_estimation_opts.sampling_factor = min(1, opts.subspace_estimation_opts.sampling_factor * prev_unfolding_size / curr_unfolding_size);
        end
    elseif strcmp(opts.subspace_estimation_opts.randomization_method, 'delta')
        % TODO
    end
    
    
    Y_unf = unfolding(Yt, dim_idx(n), 1);
    [U{dim_idx(n)}, SIGMA{dim_idx(n)}] = subspace_estimation(Y_unf, R(dim_idx(n)), opts.svd_method, opts.subspace_estimation_opts);
    Yt = nmode_product(Yt, U{dim_idx(n)}', dim_idx(n));
end
St = Yt;

end
