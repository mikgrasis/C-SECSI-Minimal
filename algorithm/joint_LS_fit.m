function F_hat_n = joint_LS_fit(Xt, F_hat, n, method)
% Compute joint LS fit for L 3-way tensors.
%
% |----------------------------------------------------------------
% | (C) 2022 TU Ilmenau, Communications Research Laboratory
% |
% |     Mikus Grasis
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |
% |     Date authored: 08.09.2022
% |     Modifications:
% |     08.09.2022 - initial version (MG)
% |----------------------------------------------------------------
%
% Inputs:
%   Xt      - cell array with L tensors
%   F_hat   - 1 x L cell array with 1 x 3 cell of matrices (per cell)
%   n       - mode for factor to be fitted
%   method  - string with method
%
% Outputs:
%   F_hat   - updated cell array with matrices
if nargin < 4
    method = 'fast';
end

L = numel(Xt);
N = 3;

I = cell(1, L);
for l = 1:L
    I{l} = size(Xt{l});
    % check inputs: dimension along joint mode must agree
    assert(eq(I{l}(n), I{1}(n)), 'oh-oh!')
end

% compute products regarding the size of unfoldings
I_bar = cell(1, L);
for l = 1:L
    for k = 1:N
        I_bar{l}(k) = prod(I{l}) / I{l}(k); % each unfolding is of size I{l}(n) x I_bar{l}(n)
    end
end

% let m and o denote the remaining two modes but n and let m < o
modes_not_n = 1:3;
modes_not_n(n) = [];
m = modes_not_n(1);
o = modes_not_n(2);

R = size(F_hat{1}{n}, 2);

fast_append = 1;
if strcmp(method, 'pinv')
    % concatenate n-mode unfoldings
    if fast_append
        % compute size of concatenated unfoldings (number of columns)
        sum_I_bar_L_n = 0; %#ok<*UNRCH>
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
        % concatenate Khatri-Rao products
        KRP_minus_n_all = nan(sum_I_bar_L_n, R);
        block_start = 1;
        for l = 1:L
            block_length = I_bar{l}(n);
            block_end = block_start + block_length - 1;
            KRP_minus_n_all(block_start:block_end, :) = krp(F_hat{l}{o}, F_hat{l}{m});
            block_start = block_end + 1;
        end
    else
        Xt_unf_n_all = [];
        KRP_minus_n_all = [];
        for l = 1:L
            Xt_unf_n_all = [Xt_unf_n_all, unfolding(Xt{l}, n, 1)];
            KRP_minus_n_all = [KRP_minus_n_all; krp(F_hat{l}{o}, F_hat{l}{m})];
        end
    end
    
    F_hat_n = Xt_unf_n_all * pinv(KRP_minus_n_all).';
    
elseif strcmp(method, 'fast')
    GRAM = zeros(R, R);
    MTTKRP = zeros(I{1}(n), R);
    for l = 1:L
        MTTKRP = MTTKRP + unfolding(Xt{l}, n, 1) * krp(F_hat{l}{o}, F_hat{l}{m});
        
        GRAM_l = ones(R, R);
        for k = 1:N
            if eq(k, n)
                continue
            end
            GRAM_l = GRAM_l .* (F_hat{l}{k}' * F_hat{l}{k});
        end
        GRAM = GRAM + GRAM_l;
    end
    F_hat_n = MTTKRP / GRAM; % MTTKRP * inv(GRAM)
end
end
