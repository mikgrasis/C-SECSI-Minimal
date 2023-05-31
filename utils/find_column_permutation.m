function idx = find_column_permutation(A_ref, A_perm, method, shared)
% Fix permutation ambiguity of factor matrices.
%
% |----------------------------------------------------------------
% | (C) 2018 Mikus Grasis
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
% |     Date authored:  July 2018
% |-----------------------------------------------------------------
%
% Wrapper function for various permutation functions.
%
% Basic usage:
% idx = find_column_permutation(A{n}, A_perm{n}, 'angle');
%   -> A_perm{n}(:, idx) == A{n}
%
% Inputs:
%   A_ref   - factor matrix with reference permutation
%   A_perm  - permuted factor matrix
%   method  - string with method to apply for processing
%               'euclid' | 'scalProd' | 'angle' | 'nion'
%   shared  - indicate shared rows
%
% Output:
%   idx     - permutation vector to restore reference
if nargin < 3
    method = 'nion';
end

switch method
    case 'euclid' % Gujral (SamBaTen)
        % don't use, produces wrong results...
        if nargin < 4
            shared = 1:size(A_ref, 1);
        end
        addpath(genpath(fullfile(return_repository_root(), 'Tensor_Decomposition', 'online_decomposition', 'SamBaTen')))

        [~, idx] = match_and_permute(A_ref, A_perm, shared);

    case 'scalProd' % Gujral (SamBaTen)
        % computes all combinations of scalar products, seems to work OK
        addpath(genpath(fullfile(return_repository_root(), 'Tensor_Decomposition', 'online_decomposition', 'SamBaTen')))
        if nargin < 4
            shared = 1:size(A_ref, 1);
        end
        F = size(A_perm, 2);
        idx = fixPermTurbo(A_ref, A_perm, F, shared);

    case 'angle' % from SECSI
        [~, idx] = fix_permutation_by_angle(A_ref, A_perm);
        idx = idx{1};

    case 'nion' % from SDT_RLST
        % works well, so far not for sparse matrices...
        addpath(genpath(fullfile(return_repository_root(), 'Tensor_Decomposition', 'online_decomposition', 'SDT_RLST')));

        [~, P] = solve_perm_scale(A_ref, A_perm);
        [idx, ~] = find(P);
        idx = idx';
end
end
