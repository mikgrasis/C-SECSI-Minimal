function A = apply_scaling(A, lambda, opts)
% Apply component weights to factors of CP model.
%
% |----------------------------------------------------------------
% | (C) 2019 Mikus Grasis
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |         Prof. Andre Lima Ferrer de Almeida
% |
% |     Date authored: June 2019
% |     Modifications:
% |     12.08.2022 - code review (MG)
% |----------------------------------------------------------------
%
% Usage:
%   A = apply_scaling(A, lambda, 'first')
%
% applies scaling in lambda to first factor matrix in A.
%
% Inputs:
%   A       - factor matrices A{n}, n = 1,...,N each factor matrix is of size I_n x R
%   lambda  - component weights (vector of size R x 1)
%
% Output:
%   A       - factor matrices A{n}, n = 1,...,N with absorbed component weights
if nargin < 3
    opts = [];
end
opts.scaling = setparam(opts, 'scaling', 'first');
N = length(A);

if strcmp(opts.scaling, 'even') % apply scaling evenly across modes
    for n = 1:N
        A{n} = A{n} * diag(lambda.^(1 / N));
    end

elseif strcmp(opts.scaling, 'first')
    A{1} = A{1} * diag(lambda);

elseif strcmp(opts.scaling, 'none')
    % when comparing factor errors, the scaling does not matter

else
    error('oh-oh')

end
end
