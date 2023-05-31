function X = krp_Nd(A, skip_dims, reverse)
% Computes repeated Khatri-Rao (column-wise Kronecker) product of N
% matrices.
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
% |     Date authored: 13.08.2019
% |     Last Modifications:
% |     01.07.2020 - bugfix for skipped dimensions (MG)
% |-----------------------------------------------------------------
%
% X = krp_Nd(A)
%
% computes the Khatri-Rao product of all matrices given in the length-N
% cell array A, viz.,
%       X = A{1} \krp \cdots \krp \A{N}.
% All matrices in A are required to have the same number of columns.
%
% Inputs: A         - cell array of N matrices
%         skip_dims - vector with dimensions to skip
%         reverse   - switch to reverse order of Khatri-Rao product
%                       default: false
% Output: X         - Khatri-Rao product of matrices
if nargin < 3, reverse = false; end
if nargin < 2, skip_dims = []; end
if ~iscell(A), A = {A}; end

N = length(A);

set_N = 1:N;
set_N(skip_dims) = [];  % remove dimensions to be skipped
if reverse
    set_N = fliplr(set_N);
end

if N == 1
    X = A{1};
else
    X = A{set_N(1)};
    for curr_n = 2:length(set_N)
        n = set_N(curr_n);
        X = krp(X, A{n});
    end
end
end
