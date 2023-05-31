function [F_hat, infos] = C_SECSI_minimal_3way(Xt, R, coupling, opts)
% Minimal version of Coupled SECSI for 3-way tensors.
%
% |----------------------------------------------------------------
% | (C) 2020 TU Ilmenau, Communications Research Laboratory
% |
% |     Mikus Grasis
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |         Prof. Andre Lima Ferrer de Almeida
% |
% |     Date authored: 05.07.2020
% |     Modifications:
% |     28.08.2022 - make coupling a separate input argument (MG)
% |     05.07.2020 - initial version (MG)
% |-----------------------------------------------------------------
%
%   -one heuristic only 'REC PS'
%   -no special cases (two-slab, two-component, rank-1)
%   -no customization
%
% 'REC PS' solves *all* SMDs possible but then, tests only combinations of
% estimates originating from the same SMD (PS = paired solutions). Often
% close to BM but a lot faster, especially for N > 3.
%
% Inputs:
%   Xt       - L x 1 cell with tensors Xt{l}, l = 1,...,L. Each tensor Xt{l}
%              is a MATLAB multidimensional array. Size along coupled dimensions must agree.
%	R        - CP rank for coupled CP decomposition
%   coupling - coupling (cf. parse_coupling.m)
%   opts     - struct with options
%
% Outputs:
%   F_hat    - L x N cell with factor matrices for L tensors
%   infos    - struct for diagnostic information
%
% References:
%   [1] A. Manina, M. Grasis, L. Khamidullina, A. Korobkov, J. Haueisen,
%       and M. Haardt, "Coupled CP Decomposition of EEG and MEG Magnetometer
%       and Gradiometer Measurements via the Coupled SECSI Framework,"
%       in 2021 55th Asilomar Conference on Signals, Systems, and Computers,
%       2021, pp. 1661-1667.
%
% Notation:
%   order N, rank R, dimensions I{l}(1),...,I{l}(N), number of tensors L
if nargin < 4
    opts = [];
end
infos = [];

%% Input Checks
L = numel(Xt);
N = length(size(Xt{1}));

I = cell(1, L);
for l = 1:L
    I{l} = size(Xt{l});
end

% Convert coupling to format 1 (binary vector)
output_format = 1;
coupling = parse_coupling(I, R, coupling, output_format); % this also verifies the dimensions

% Check for complex-valued data
real_valued = 1;
for l = 1:L
    if ~isreal(Xt{l})
        real_valued = 0;
        break
    end
end

%% Step1: Estimate Tucker Cores and n-Mode Subspaces
[St, U] = coupled_ST_HOSVD(Xt, R, coupling, opts);

%% Initialization
num_SMDs = 6;
F_hat = cell(1, L);
for l = 1:L
    F_hat{l} = cell(num_SMDs, N);
end

%% Let's Go...
num_good_modes = 0;
for n = 1:3
    I_n = nan(1, L);
    for l = 1:L, I_n(l) = I{l}(n); end
    if any(I_n < R)
        continue % skip mode if dimension I{l}(n) is smaller than rank we are looking for
    end
    
    % Let m and o denote the remaining two modes but n and let m < o
    modes_not_n = 1:3;
    modes_not_n(n) = [];
    m = modes_not_n(1);
    o = modes_not_n(2);
    
    %% Step 2: n-Mode Core Tensors
    St_n = cell(1, L);
    for l = 1:L
        St_n{l} = nmode_product(St{l}, U{l}{n}, n);
    end
    
    % Arrange 'St_n{l}' in slices of size R x R
    for l = 1:L
        St_n{l} = permute(St_n{l}, [modes_not_n, n]); % move n-mode to the last dimension...
    end
    
    %% Step 3: Compute condition numbers of each slice
    conds = cell(1, L);
    for l = 1:L
        I{l}(n) = size(St_n{l}, 3);
        conds{l} = zeros(1, I{l}(n));
        for k = 1:I{l}(n)
            conds{l}(k) = cond(St_n{l}(:, :, k));
        end
    end
    
    % Determine slice wih the best conditioning number
    p = zeros(1, L);
    for l = 1:L
        [~, p(l)] = min(conds{l});
        if rank(St_n{l}(:, :, p(l))) < R
            warning('SECSI:rankdeficiency', 'Rank-deficient mode detected. Skipping (this warning is only displayed once)...');
            warning('off', 'SECSI:rankdeficiency');
            continue
        end
    end
    num_good_modes = num_good_modes + 1; % number of non rank-deficient modes
    
    %% Step 4: Pivoting
    St_n_rhs = cell(1, L);
    St_n_lhs = cell(1, L);
    for l = 1:L
        St_n_rhs{l} = St_n{l};
        St_n_lhs{l} = St_n{l};
        for k = 1:I{l}(n)
            St_n_rhs{l}(:, :, k) = St_n_rhs{l}(:, :, k) / St_n{l}(:, :, p(l));
            St_n_lhs{l}(:, :, k) = (St_n{l}(:, :, p(l)) \ St_n_lhs{l}(:, :, k)).';
        end
    end
    
    %% Step 5: Solve Symmetric SMD-Problem
    if find(coupling)
        num_slices_in_mode_n = 0;
        for l = 1:L
            num_slices_in_mode_n = num_slices_in_mode_n + I{l}(n);
        end
    end
    
    D_rhs = cell(1, L); D_lhs = cell(1, L);
    T_rhs = cell(1, L); T_lhs = cell(1, L);
    if ismember(find(coupling), m)
        %%% append slices
        St_n_rhs_all = nan(R, R, num_slices_in_mode_n);
        block_start = 1;
        for l = 1:L
            block_end = block_start + I{l}(n) - 1;
            St_n_rhs_all(:, :, block_start:block_end) = St_n_rhs{l};
            block_start = block_start + I{l}(n);
        end
        
        %%% solve RHS problem jointly
        if real_valued
            [D_rhs, T_rhs] = jointdiag(St_n_rhs_all);
        else
            [D_rhs, T_rhs] = jointdiag_c(St_n_rhs_all);
        end
    else
        %%% solve RHS problem separately
        if real_valued
            for l = 1:L
                [D_rhs{l}, T_rhs{l}] = jointdiag(St_n_rhs{l});
            end
        else
            for l = 1:L
                [D_rhs{l}, T_rhs{l}] = jointdiag_c(St_n_rhs{l});
            end
        end
    end
    if ismember(find(coupling), o)
        %%% append slices
        St_n_lhs_all = nan(R, R, num_slices_in_mode_n);
        block_start = 1;
        for l = 1:L
            block_end = block_start + I{l}(n) - 1;
            St_n_lhs_all(:, :, block_start:block_end) = St_n_lhs{l};
            block_start = block_start + I{l}(n);
        end
        
        %%% solve LHS problem jointly
        if real_valued
            [D_lhs, T_lhs] = jointdiag(St_n_lhs_all);
        else
            [D_lhs, T_lhs] = jointdiag_c(St_n_lhs_all);
        end
    else
        %%% solve LHS problem separately
        if real_valued
            for l = 1:L
                [D_lhs{l}, T_lhs{l}] = jointdiag(St_n_lhs{l});
            end
        else
            for l = 1:L
                [D_lhs{l}, T_lhs{l}] = jointdiag_c(St_n_lhs{l});
            end
        end
    end
    
    %% Step 6: Obtain two sets of three estimates (paired solutions)
    %%% First estimate: from the transform matrices
    for l = 1:L
        if ismember(m, find(coupling))
            F_hat{l}{2*n-1, m} = U{l}{m} * T_rhs;
        else
            F_hat{l}{2*n-1, m} = U{l}{m} * T_rhs{l};
        end
        
        if ismember(o, find(coupling))
            F_hat{l}{2*n, o} = U{l}{o} * T_lhs;
        else
            F_hat{l}{2*n, o} = U{l}{o} * T_lhs{l};
        end
    end
    
    %%% Second estimate: the diagonals of the matrices
    % D_rhs_k = T_m^{-1) * St_n_rhs_k * T_m
    % and
    % D_lhs_k = T_l^{-1) * St_n_lhs_k * T_l
    % provide an estimate for the rows of the matrix F^{(n)}
    k_rhs_joint = 0;
    k_lhs_joint = 0;
    for l = 1:L
        F_n_rhs_l = zeros(I{l}(n), R);
        F_n_lhs_l = zeros(I{l}(n), R);
        for k = 1:I{l}(n)
            if ismember(m, find(coupling))
                k_rhs_joint = k_rhs_joint + 1;
                F_n_rhs_l(k, :) = diag(D_rhs(:, :, k_rhs_joint));
            else
                F_n_rhs_l(k, :) = diag(D_rhs{l}(:, :, k));
            end
            if ismember(o, find(coupling))
                k_lhs_joint = k_lhs_joint + 1;
                F_n_lhs_l(k, :) = diag(D_lhs(:, :, k_lhs_joint));
            else
                F_n_lhs_l(k, :) = diag(D_lhs{l}(:, :, k));
            end
        end
        F_hat{l}{2*n-1, n} = F_n_rhs_l;
        F_hat{l}{2*n, n} = F_n_lhs_l;
    end
    
    %%% Third estimate: LS fit of missing estimate
    % Given two factor matrices, the remaining one is found via a pseudo inverse
    for l = 1:L
        F_hat{l}{2*n-1, o} = unfolding(Xt{l}, o) / krp_Nd(F_hat{l}(2*n-1, [o + 1:N, 1:o - 1])).';
        F_hat{l}{2*n, m} = unfolding(Xt{l}, m) / krp_Nd(F_hat{l}(2*n, [m + 1:N, 1:m - 1])).';
    end
    
    if ismember(o, find(coupling))
        for l = 2:L
            % Align estimates according to the coupled mode
            idx_perm = find_column_permutation(F_hat{1}{2*n-1, o}, F_hat{l}{2*n-1, o}, 'angle');
            F_hat{l}{2*n-1, o} = F_hat{l}{2*n-1, o}(:, idx_perm);
            F_hat{l}{2*n-1, n} = F_hat{l}{2*n-1, n}(:, idx_perm);
            F_hat{l}{2*n-1, m} = F_hat{l}{2*n-1, m}(:, idx_perm);
            
            % Fix sign ambiguity
            for j = find(diag(F_hat{1}{2*n-1, o}' * F_hat{l}{2*n-1, o}) < 0)
                F_hat{1}{2*n-1, o}(:, j) = F_hat{1}{2*n-1, o}(:, j) * -1; % resolve sign ambiguity
                F_hat{1}{2*n-1, n}(:, j) = F_hat{1}{2*n-1, n}(:, j) * -1; % move sign into some other factor
            end
        end
        F_hat_o = joint_LS_fit(Xt, cellfun(@(c) c(2*n-1, :), F_hat, 'uni', 0), o);
        for l = 1:L
            F_hat{l}{2*n-1, o} = F_hat_o;
            
            % Update component weights (LS-fit)
            gamma_hat = krp_Nd(F_hat{l}(2*n-1, :), [], 1) \ Xt{l}(:);
            F_hat{l}{2*n-1, o} = F_hat_o * diag(gamma_hat);
        end
    end
    % Compute joint LS-fit for the coupled mode
    if ismember(m, find(coupling))
        for l = 2:L
            % Align estimates according to the coupled mode
            idx_perm = find_column_permutation(F_hat{1}{2*n, m}, F_hat{l}{2*n, m}, 'angle');
            F_hat{l}{2*n, m} = F_hat{l}{2*n, m}(:, idx_perm);
            F_hat{l}{2*n, n} = F_hat{l}{2*n, n}(:, idx_perm);
            F_hat{l}{2*n, o} = F_hat{l}{2*n, o}(:, idx_perm);
            
            % Fix sign ambiguity
            for j = find(diag(F_hat{1}{2*n, m}' * F_hat{l}{2*n, m}) < 0)
                F_hat{1}{2*n, m}(:, j) = F_hat{1}{2*n, m}(:, j) * -1; % resolve sign ambiguity
                F_hat{1}{2*n, n}(:, j) = F_hat{1}{2*n, n}(:, j) * -1; % move sign into some other factor
            end
        end
        F_hat_m = joint_LS_fit(Xt, cellfun(@(c) c(2*n, :), F_hat, 'uni', 0), m);
        for l = 1:L
            F_hat{l}{2*n, m} = F_hat_m;
            
            % Update component weights (LS-fit)
            gamma_hat = krp_Nd(F_hat{l}(2*n, :), [], 1) \ Xt{l}(:);
            F_hat{l}{2*n, m} = F_hat{l}{2*n, m} * diag(gamma_hat);
        end
    end
end
if num_good_modes == 0
    warning('SECSI:couldnotdecompose', 'Could not decompose the tensor: too many rank deficiencies. Returning NaN.');
    for l = 1:L
        for n = 1:N
            F_hat{l}{n} = nan(I(n), R);
        end
    end
    return
end

%% Step 7: Select final estimate
% We'll select the set of estimates with the smallest reconstruction error for each tensor
num_est = 2 * num_good_modes;
rec_err = inf(L, num_est);
for l = 1:L
    F_hat{l} = F_hat{l}(1:num_est, :);
    if num_est > 1
        for curr_est = 1:num_est
            Xt_rec_l = cp_construct(F_hat{l}(curr_est, :));
            rec_err(l, curr_est) = norm(Xt{l}(:)-Xt_rec_l(:))^2;
        end
        [~, best] = min(rec_err(l, :));
        F_hat{l} = F_hat{l}(best, :);
    end
end
infos.rec_err = rec_err;
end
