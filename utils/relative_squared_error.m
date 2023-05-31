function RSE = relative_squared_error(Xt, Xt_hat)
% Compute relative squared error (RSE) of two tensors.
%
% |----------------------------------------------------------------
% | (C) 2019 Mikus Grasis
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |         Prof. Andre Lima Ferrer de Almeida
% |
% |     Last modifications: 19.09.2019
% |         - formatted according to coding rules
% |     12.07.2019
% |         - special case for sptensor/ttensor added
% |     05.06.2019
% |         - facelift on variable names and header
% |     Date authored: 21.04.2019
% |----------------------------------------------------------------
%
% RSE = relative_squared_error(Xt, Xt_hat)
%
% Computes the relative squared reconstruction error (RSE) of tensor Xt
% given its reconstruction Xt_hat. Input tensors must be of same type, e.g.,
% tensor, ktensor, sptensor, or multidim-array.
%
%   RSE = ||Xhat-X||_F^2 / ||X||_F^2
%
% Inputs:
%   Xt - original tensor
%   Xt_hat - approximated tensor
%
% Output:
% 	RSE - relative squared error
if ~strcmp(class(Xt), class(Xt_hat))
    warning('relative_error: first input is of type %s, second input is of type %s', class(Xt), class(Xt_hat));
end

if isa(Xt, 'double') && isa(Xt_hat, 'double')

    % MATLAB
    RSE = norm(Xt_hat(:)-Xt(:))^2 / norm(Xt(:))^2;

elseif isa(Xt, 'sptensor') && isa(Xt_hat, 'ttensor')
    fprintf('It''s fine I''ll use the SptTtDiffNorm function from Malik et al.!\n')
    %   [1] O. A. Malik, S. Becker. Low-Rank Tucker Decomposition of Large
    %       Tensors Using TensorSketch. Advances in Neural Information
    %       Processing Systems 32, pp. 10117-10127, 2018.

    % For sparse Tucker tensors there is an optimized routine for computing
    % the norm ||X-[G;V]||_F where X is an sptensor and [G;V] is a Tucker
    % tensor
    RSE = SptTtDiffNorm(Xt, Xt_hat.core, Xt_hat.U)^2 / norm(Xt)^2;

else
    % Tensor Toolbox
    RSE = norm(Xt_hat-Xt)^2 / norm(Xt)^2;
end
end
