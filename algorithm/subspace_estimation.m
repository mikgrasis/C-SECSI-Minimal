function [U, SIGMA, info, V] = subspace_estimation(X, R, method, opts)
% Compute dominant left singular vectors and singular values of a matrix.
%
% |----------------------------------------------------------------
% | (C) 2022 TU Ilmenau, Communications Research Laboratory
% |
% |     Mikus Grasis, Alla Manina
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |
% |     Date authored: 04.04.2022
% |     Last Modifications:
% |     18.06.2022 - include V as an output (MG)
% |     10.05.2022 - updated sampling routine (MG)
% |     29.04.2022 - added sampling routine (initial version) (AM)
% |     04.04.2022 - initial version (MG)
% |----------------------------------------------------------------
%
% Inputs:
%   X       - matrix of size I x J
%   R       - rank R for truncation
%   method  - method for computation: SVD | SVDS | EIG | EIGS | RSVD
%   opts    - struct with options
%       opts.randomization_method       - [] | 'delta' | 'sampling_factor' | 'v_idx' | 'euclidean' [2] | 'projection' [3]
%       opts.Delta             - fraction of cols to add to I, i.e., K = I + Delta * I
%       opts.sampling_factor   - fraction of how many columns we are taking
%       opts.v_idx             - indices of selected columns to use
%
% Outputs:
% 	U       - matrix of left singular vectors, size: I X R
%   SIGMA   - diagonal matrix of singular values, size: R x R
%   info    - struct with output information
%       info.v_idx  - indices of the selected sampled columns
% 	V       - matrix of right singular vectors, size: J X R
%
% Note1: Computing V is in some cases expensive and often not required.
% Hence, V in some cases in only computed when the function is called with
% four output arguments. (That's why we made it the last output argument!)
%
% !!! Note2: In the cases, where sampling is applied, the computation of
% the V matrices has so far not been considered and is likely to cause
% unexpected results!!! !!!
%
% References:
%   [1] L. De Lathauwer, B. De Moor, and J. Vandewalle, "A multilinear
%       singular value decomposition," SIAM J. Matrix Anal. Appl., vol. 21,
%       no. 4, pp. 1253-1278, 2000.
%   [2] P. Drineas and M. W. Mahoney, "A randomized algorithm for a
%       tensor-based generalization of the singular value decomposition,"
%       Linear Algebra Appl., vol. 420, pp. 553-571, 2007.
%   [3] G. Zhou, A. Cichocki, and S. Xie, "Decomposition of big tensors with
%       low multilinear rank," arXiv Prepr. arXiv1412.1885, 2014.
if nargin < 4
    opts = [];
end
opts.randomization_method = setparam(opts, 'randomization_method', []);

if nargin < 3
    method = 'EIG';
end

[I, J] = size(X);
if isempty(R)
    R = min(I, J);
end

%%% run sampling
info.v_idx = [];
if ~isempty(opts.randomization_method)
    % determine the number of columns
    if isfield(opts, 'Delta')
        K = I + round(opts.Delta * I);
        if K > J
            K = J;
            disp('subspace_estimation: less than K cols available, using all available cols...')
        end
    elseif isfield(opts, 'sampling_factor')
        K = round(J * opts.sampling_factor);
    end
    
    % apply randomization method
    if strcmp(opts.randomization_method, 'v_idx')
        v_idx = opts.sampling.v_idx;
        X = X(:, v_idx);
        info.v_idx = v_idx;
        
    elseif strcmp(opts.randomization_method, 'euclidean')
        W = sum(abs(X).^2, 1);
        v_idx = randsample(J, K, 'true', W); % with replacement (why?)
        X = X(:, v_idx);
        info.v_idx = v_idx;
        
    elseif strcmp(opts.randomization_method, 'delta') || strcmp(opts.randomization_method, 'sampling_factor')
        v_idx = randsample(J, K); % sample without replacement
        X = X(:, v_idx);
        info.v_idx = v_idx;
        
    elseif strcmp(opts.randomization_method, 'projection')
        omega = randn(J, K);
        X = X * omega;
        
    else
        error('subspace_estimation:sampling', 'unknown method')
    end
end

%%% compute SVD of non-/sampled columns
switch method
    case 'SVD'
        if size(X, 2) > size(X, 1) % X is a wide matrix
            [V, SIGMA, U] = svd(X', 0); % faster?!
        else % X is a tall matrix
            [U, SIGMA, V] = svd(X, 0);
        end
        % truncate, store singular values in RxR matrix
        %U = U(:, 1:min(R, size(U, 2))); % @Alla - do we need this?
        U = U(:, 1:R);
        SIGMA = SIGMA(1:R, 1:R);
        V = V(:, 1:R);
        
    case 'SVDS' % works but does not make too much sense? (need sparse input and handling!)
        [U, SIGMA, V] = svds(X, R);
        
    case 'EIG'
        if size(X, 2) > size(X, 1) % X is a wide matrix
            [U, LAMBDA] = eig(X * X');
            [~, idx_EV] = sort(diag(LAMBDA), 'descend');
            
            % truncate, store singular values in RxR matrix
            U = U(:, idx_EV(1:R));
            SIGMA = sqrt(LAMBDA(idx_EV(1:R), idx_EV(1:R)));
            if nargout == 4
                V = X' * U / SIGMA;
            end
            
        else % X is a tall matrix
            [V, LAMBDA] = eig(X' * X);
            [~, idx_EV] = sort(diag(LAMBDA), 'descend');
            
            % truncate, store singular values in RxR matrix
            V = V(:, idx_EV(1:R));
            SIGMA = sqrt(LAMBDA(idx_EV(1:R), idx_EV(1:R)));
            U = X * V / SIGMA;
        end
        
    case 'EIGS'
        if size(X, 2) > size(X, 1) % X is a wide matrix
            [U, LAMBDA] = eigs(X * X', R);
            SIGMA = sqrt(LAMBDA);
            if nargout == 4
                V = X' * U / SIGMA;
            end
        else % X is a tall matrix
            [V, LAMBDA] = eigs(X' * X, R);
            SIGMA = sqrt(LAMBDA);
            U = X * V / SIGMA;
        end
        
    case 'RSVD'
        % randomized SVD from [2] (implementation A. Liutkus)
        rootDir = return_repository_root();
        addpath(genpath(fullfile(rootDir, 'Matrix_Computations', 'rsvd')))
        [U, SIGMA, V] = rsvd(X, R);
        
    otherwise
        error('subspace_estimation:method', 'method not supported')
        
end
end
