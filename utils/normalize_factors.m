function [A, lambda] = normalize_factors(A)
% Normalize columns of factor matrices to unit norm and absorb scaling into
% lambda.
%
% |----------------------------------------------------------------
% | (C) 2021 Mikus Grasis
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |         Prof. Andre Lima Ferrer de Almeida
% |
% |     Date authored: September 2017
% |     Modifications:
% |     10.04.2021 - code review (MG)
% |----------------------------------------------------------------
%
% Usage:
% [A, lambda] = normalize_factors(A)
%
% normalizes the columns of the factors in A to unit norm and absorbs
% scaling into lambda.
%
% Inputs: A      - set of factor matrices A{n}, n = 1,...,N
%                  each factor matrix has size I(n) x R
%
% Output: A      - normalized factors A{n}, n = 1,...,N
%         lambda - component weights vector of size R x 1
%
% Notation: order N, rank R, dimensions I(1),...,I(N)
if ~iscell(A)
    A = {A};
end
N = numel(A);
R = size(A{1}, 2);

% normalize columns and store scaling in lambda
lambda = ones(R, 1);
for n = 1:N
    for r = 1:R
        sc = norm(A{n}(:, r), 2);
        A{n}(:, r) = A{n}(:, r) ./ sc;
        lambda(r) = lambda(r) * sc;
    end
end
end
