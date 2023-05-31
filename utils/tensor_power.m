function P = tensor_power(Xt)
% Computes average power of tensor entries.
%
% |----------------------------------------------------------------
% | (C) 2019 Mikus Grasis
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |         Prof. Andre Lima Ferrer de Almeida
% |
% |     Last Modifications: 26.4.2019
% |         -headed added
% |         -diffenrentiate between ktensor, tensor, ttensor, and sptensor
% |         -added case for sparse tensor
% |         -added error case for invalid datatype
% |         -replaced prod(size(Xt)) by numel(Xt) for double
% |     Date Authored: August 2018
% |----------------------------------------------------------------
%
% Inputs: Xt - tensor Xt can be a Matlab double array or one of the following
%              types from Tensor Toolbox: ktensor, tensor, ttensor, or sptensor
% Output: P  - power (i.e., sample variance over tensor entries)
%
% Notation:
%   order N, rank R, dimensions I(1),...,I(N)
%#ok<*PSIZE>

if isa(Xt, 'tensor') || isa(Xt, 'ktensor') || isa(Xt, 'ttensor')
    P = norm(Xt)^2 / prod(size(Xt));
elseif isa(Xt, 'sptensor')
    if nnz(Xt) == 0
        P = 0;
    else
        P = norm(Xt)^2 / nnz(Xt);
    end
elseif isa(Xt, 'double')
    P = Xt(:)' * Xt(:) / numel(Xt);
else
    error('tensor_power: tensor should be double, ktensor, tensor, ttensor, or sptensor')
end
end
