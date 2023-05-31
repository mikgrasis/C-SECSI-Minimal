function [St, U, SIGMA] = coupled_ST_HOSVD(Xt, R, coupling, opts)
% Computes the coupled sequentially truncated higher-order singular value
% decomposition (C-ST-HOSVD) of L tensors.
%
% |----------------------------------------------------------------
% | (C) 2022 Mikus Grasis, Alla Manina
% |
% |
% |  ______   ______     __   __     ______     ______     ______        ______   ______     ______     __         ______
% | /\__  _\ /\  ___\   /\ "-.\ \   /\  ___\   /\  __ \   /\  == \      /\__  _\ /\  __ \   /\  __ \   /\ \       /\  ___\
% | \/_/\ \/ \ \  __\   \ \ \-.  \  \ \___  \  \ \ \/\ \  \ \  __<      \/_/\ \/ \ \ \/\ \  \ \ \/\ \  \ \ \____  \ \___  \
% |    \ \_\  \ \_____\  \ \_\\"\_\  \/\_____\  \ \_____\  \ \_\ \_\       \ \_\  \ \_____\  \ \_____\  \ \_____\  \/\_____\
% |     \/_/   \/_____/   \/_/ \/_/   \/_____/   \/_____/   \/_/ /_/        \/_/   \/_____/   \/_____/   \/_____/   \/_____/
% |
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |
% |     Date Authored: 13.04.2022
% |     Last Modifications:
% |     12.05.2022 - add support for subspace_estimation_opts (MG)
% |     13.04.2022 - initial version (MG)
% |----------------------------------------------------------------
%
% Inputs:
%   Xt      - Cell with tensors. Each tensor Xt{l}, l = 1,...,L is a MATLAB
%             multidimensional array. Size along coupled dimensions must be equal.
%   R       - a) Cell with multilinear rank R{l}(1),...,R{l}(N(l)) for truncation.
%                Truncation rank along coupled dimension(s) must be equal.
%             b) One rank value R for (R,...,R) truncation.
%   coupling - coupling (cf. parse_coupling.m)
%   opts    - struct with options
%           opts.svd_method = 'EIG'; % SVD | SVDS | EIG | EIGS | RSVD
%           opts.dim_order - ordering of the computations % sequential | descending | ascending | random
%           opts.fast  - append coupled unfoldings using allocation
%           opts.subspace_estimation_opts - this is passed on to subspace_estimation.m
%
% Outputs:
%   St      - Cell with core tensors, each core tensor St{l}, l = 1,...,L
%               is of size R{l}(1) x ... x R{l}(N(l)).
%   U       - Cell with L sets of matrices of singular vectors.
%   SIGMA   - Cell with n-mode singular values. Each SIGMA{l}, l = 1,...,L
%               is a cell array with N vectors of singular values.
%
% Notation:
%   number of tensors L, order N, rank R, dimensions I
%
% References:
% [1] N. Vannieuwenhoven, R. Vandebril, and K. Meerbergen, "A New Truncation
%   Strategy for the Higher-Order Singular Value Decomposition," SIAM J. Sci.
% Comput., vol. 34, no. 2, pp. A1027-A1052, Jan. 2012.

%% Preliminaries
if nargin < 4
    opts = [];
end
opts.dim_order = setparam(opts, 'dim_order', 'descending'); % sequential | descending | ascending | random
opts.fast = setparam(opts, 'fast', 1); % 0 | 1
opts.svd_method = setparam(opts, 'svd_method', 'EIG'); % SVD | SVDS | EIG | EIGS | RSVD
opts.subspace_estimation_opts = setparam(opts, 'subspace_estimation_opts', []);

L = numel(Xt);

% query order of input tensors
N = nan(1, L);
for l = 1:L
    N(l) = length(size(Xt{l}));
end

% check if same tensor order for all
same_order = false;
if eq(length(unique(N)), 1)
    same_order = true;
end
assert(same_order, 'coupled_ST_HOSVD:input_check', 'so far only implemented for tensors of same order')

% query size of input tensors
same_size = true;
I = cell(1, L);
for l = 1:L
    I{l} = size(Xt{l});
    if ~all(eq(I{l}, I{1}))
        same_size = false;
    end
end
% check if same tensor dimensions for all
assert(same_size, 'coupled_ST_HOSVD:input_check', 'so far only implemented for tensors of equal dimensions')

% handle different rank input formats
if isscalar(R) % one rank that is used for all tensors in all dimensions
    R_old = R;
    R = cell(1, L);
    for l = 1:L
        R{l} = ones(1, N(l)) * R_old;
    end
elseif isvector(R) && ~iscell(R) && same_order % one multilinear rank that is used for all tensors
    assert(eq(length(R), N(1)), 'coupled_HOSVD:input_check', 'multilinear ranks must match number of dimensions')
    R_old = R;
    R = cell(1, L);
    for l = 1:L
        R{l} = R_old;
    end
elseif iscell(R) % the normal case: L multilinear ranks for truncation
    assert(eq(numel(R), L), 'coupled_HOSVD:input_check', 'multilinear ranks must match number of tensors')
    for l = 1:L
        assert(eq(length(R{l}), N(l)), 'coupled_HOSVD:input_check', 'multilinear ranks must match number of dimensions')
    end
else
    error('oh-oh!')
end

% convert coupling to format 1 (binary vector)
output_format = 1;
coupling = parse_coupling(I, R, coupling, output_format); % this also verifies the dimensions


%% Determine dimension ordering
N = N(1); % so far, we only work with N_1 = ... = N_L only

if strcmp(opts.dim_order, 'descending')
    % start with largest dimension
    [~, dim_idx] = sort(I{1}, 'descend');

elseif strcmp(opts.dim_order, 'ascending')
    % start with smallest dimension
    [~, dim_idx] = sort(I{1}, 'ascend');

elseif strcmp(opts.dim_order, 'sequential')
    % go thru modes sequentially 1,...,N
    dim_idx = 1:N;

elseif strcmp(opts.dim_order, 'random')
    % go thru modes randomly
    dim_idx = randperm(N);

elseif isnumeric(opts.dim_order) % explicit dimension ordering
    assert(eq(length(opts.dim_order), N))
    dim_idx = opts.dim_order;

else
    error('coupled_ST_HOSVD:input_check', 'unknown dimension ordering')
end

%% Compute products regarding the size of unfoldings
if opts.fast
    I_breve = cell(1, L);
    for l = 1:L
        for n = 1:N
            I_breve{l}(dim_idx(n)) = prod(R{l}(dim_idx(1:n-1))) * prod(I{l}(dim_idx(n+1:N))); % each unfolding is of size I{l}(n) x I_breve{l}(n)
        end
    end
end

%% Coupled ST-HOSVD
Yt = Xt;
U = cell(1, L); SIGMA = cell(1, L);
for n = 1:N
    if coupling(dim_idx(n))
        % concatenate n-mode unfoldings
        if opts.fast
            % compute size of concatenated unfoldings (number of columns)
            sum_I_bar_L_n = 0;
            for l = 1:L
                sum_I_bar_L_n = sum_I_bar_L_n + I_breve{l}(dim_idx(n));
            end
            % issue warning for large concatenated unfoldings
            warn_size = 1e8 * 1.25;
            if gt(I{1}(dim_idx(n)) * sum_I_bar_L_n, warn_size)
                warning('Xt_unf_n_all will use more than %1.2f GB in memory', warn_size * 8 / 1e9) % 1 entry = 64 Bit = 8 Byte
            end
            % concatenate n-mode unfoldings
            Yt_unf_n_all = nan(I{1}(dim_idx(n)), sum_I_bar_L_n);
            block_start = 1;
            for l = 1:L
                block_length = I_breve{l}(dim_idx(n));
                block_end = block_start + block_length - 1;
                Yt_unf_n_all(:, block_start:block_end) = unfolding(Yt{l}, dim_idx(n), 1);
                block_start = block_end + 1;
                if eq(l, L)
                    assert(eq(block_start, sum_I_bar_L_n + 1), 'oh-oh!'); % just checking...
                end
            end
        else
            Yt_unf_n_all = [];
            for l = 1:L
                Yt_unf_n_all = [Yt_unf_n_all, unfolding(Yt{l}, dim_idx(n), 1)]; %#ok<AGROW>
            end
        end
        % compute SVD of concatenated n-mode unfoldings
        [U{1}{dim_idx(n)}, SIGMA{1}{dim_idx(n)}] = subspace_estimation(Yt_unf_n_all, R{1}(dim_idx(n)), opts.svd_method, opts.subspace_estimation_opts);
        for l = 2:L
            U{l}{dim_idx(n)} = U{1}{dim_idx(n)};
            SIGMA{l}{dim_idx(n)} = SIGMA{1}{dim_idx(n)};
        end
        % multiply core tensor
        for l = 1:L
            Yt{l} = nmode_product(Yt{l}, U{l}{dim_idx(n)}', dim_idx(n));
        end
    else
        for l = 1:L
            % compute SVD of n-mode unfolding for each tensor individually
            [U{l}{dim_idx(n)}, SIGMA{l}{dim_idx(n)}] = subspace_estimation(unfolding(Xt{l}, dim_idx(n), 1), R{l}(dim_idx(n)), opts.svd_method, opts.subspace_estimation_opts);
            % multiply core tensor
            Yt{l} = nmode_product(Yt{l}, U{l}{dim_idx(n)}', dim_idx(n));
        end
    end
end
St = Yt;
end
