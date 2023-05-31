function [St, U, SIGMA] = coupled_HOSVD(Xt, R, coupling, opts)
% Computes the coupled higher-order singular value decomposition of a set
% of L tensors.
%
% |----------------------------------------------------------------
% | (C) 2021 Mikus Grasis, Alla Manina
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
% |         Prof. Andre Lima Ferrer de Almeida
% |
% |     Date Authored: 22.03.2020
% |     Modifications:
% |     12.05.2022 - add support for subspace_estimation_opts (MG)
% |     13.04.2022 - move subspace estimation to subspace_estimation.m (MG)
% |     02.09.2021 - code review (MG)
% |     04.08.2021 - coupling moved into separate input argument (MG)
% |     17.05.2021 - SIGMA: added fix for degenerate case (AM)
% |     21.04.2021 - code review (MG)
% |     11.03.2021 - bugfix output of n-mode singular values (AM)
% |     05.07.2020 - initial version (MG)
% |----------------------------------------------------------------
%
% Inputs:
%   Xt      - Cell with tensors. Each tensor Xt{l}, l = 1,...,L is a MATLAB
%             multidimensional array. Size along coupled dimensions must match.
%   R       - a) Cell with multilinear rank R{l}(1),...,R{l}(N(l)) for truncation.
%                Truncation rank along coupled dimension(s) must be equal.
%             b) one rank value R for R,...,R truncation
%   coupling - coupling (cf. parse_coupling.m)
%   opts    - struct with options
%       opts.svd_method = = 'EIG'; % SVD | SVDS | EIG | EIGS | RSVD
%       opts.dim_order - ordering of the computations for core tensor
%       opts.fast  - append coupled unfoldings using allocation
%       opts.subspace_estimation_opts - this is passed on to subspace_estimation.m
%
% Outputs:
%   St      - Cell with core tensors. Each core tensor St{l}, l = 1,...,L
%             is of size R{l}(1) x ... x R{l}(N(l)).
%   U       - Cell with L sets of matrices of singular vectors.
%   SIGMA   - Cell with n-mode singular values. Each SIGMA{l}, l = 1,...,L
%             is a cell array with N vectors of singular values.
%
% Notation:
%   number of tensors L, order N, dimensions I, rank R

%% Preliminaries
if nargin < 4
    opts = [];
end
opts.dim_order = setparam(opts, 'dim_order', 'descending'); % descending | sequential
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
assert(same_order, 'coupled_HOSVD:input_check', 'so far only implemented for tensors of same order')

% query size of input tensors
I = cell(1, L);
for l = 1:L
    I{l} = size(Xt{l});
end

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

% compute products regarding the size of unfoldings
if opts.fast
    I_bar = cell(1, L);
    for l = 1:L
        for n = 1:N(l)
            I_bar{l}(n) = prod(I{l}) / I{l}(n); % each unfolding is of size I{l}(n) x I_bar{l}(n)
        end
    end
end

%% Coupled HOSVD
U = cell(1, L);
SIGMA = cell(1, L);
for n = 1:N
    if coupling(n)
        % concatenate n-mode unfoldings
        if opts.fast
            % compute size of concatenated unfoldings (number of columns)
            sum_I_bar_L_n = 0;
            for l = 1:L
                sum_I_bar_L_n = sum_I_bar_L_n + I_bar{l}(n);
            end
            % issue warning for large concatenated unfoldings
            warn_size = 1e8 * 1.25;
            if gt(I{1}(n) * sum_I_bar_L_n, warn_size)
                warning('Xt_unf_n_all will use more than %1.2f GB in memory', warn_size * 8 / 1e9) % 1 entry = 64 Bit = 8 Byte
            end
            % concatenate n-mode unfoldings
            Xt_unf_n_all = nan(I{1}(n), sum_I_bar_L_n);
            block_start = 1;
            for l = 1:L
                block_length = I_bar{l}(n);
                block_end = block_start + block_length - 1;
                Xt_unf_n_all(:, block_start:block_end) = unfolding(Xt{l}, n, 1);
                block_start = block_end + 1;
                if eq(l, L)
                    assert(eq(block_start, sum_I_bar_L_n + 1), 'oh-oh!'); % just checking...
                end
            end
        else
            Xt_unf_n_all = [];
            for l = 1:L
                Xt_unf_n_all = [Xt_unf_n_all, unfolding(Xt{l}, n, 1)]; %#ok<AGROW>
            end
        end
        % compute SVD of concatenated n-mode unfoldings
        [U{1}{n}, SIGMA{1}{n}] = subspace_estimation(Xt_unf_n_all, R{1}(n), opts.svd_method, opts.subspace_estimation_opts);
        for l = 2:L
            U{l}{n} = U{1}{n};
            SIGMA{l}{n} = SIGMA{1}{n};
        end

    else
        for l = 1:L
            % compute SVD of n-mode unfolding for each tensor individually
            [U{l}{n}, SIGMA{l}{n}] = subspace_estimation(unfolding(Xt{l}, n, 1), R{l}(n), opts.svd_method, opts.subspace_estimation_opts);
        end
    end
end
St = cell(1, L);
for l = 1:L
    if strcmp(opts.dim_order, 'descending')  % start with largest dimension (faster)
        [~, idx_descending] = sort(I{l}, 'descend');
        St{l} = nmode_product(Xt{l}, U{l}{idx_descending(1)}', idx_descending(1));
        for n = 2:N(l)
            St{l} = nmode_product(St{l}, U{l}{idx_descending(n)}', idx_descending(n));
        end
    else
        St{l} = nmode_product(Xt{l}, U{l}{1}', 1);
        for n = 2:N(l)
            St{l} = nmode_product(St{l}, U{l}{n}', n);
        end
    end
end
end
