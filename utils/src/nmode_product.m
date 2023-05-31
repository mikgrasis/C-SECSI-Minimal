function Xt_out = nmode_product(Xt_in, A, n)
% Computes the n-mode product of a tensor and a matrix.
%
% |----------------------------------------------------------------
% | (C) 2006 TU Ilmenau, Communications Research Laboratory
% |
% |     Martin Weis
% |
% |     Advisors:
% |        Dipl.-Ing. Giovanni Del Galdo
% |        Univ.-Prof. Dr.-Ing. Martin Haardt
% |
% |     Date authored: 06.20.2006
% |     Last modifications:
% |     22.08.2021 - use Matlab column ordering (MG)
% |     03.07.2019 - edits on notation (MG)
% |----------------------------------------------------------------
%
% Xt_out = nmode_product(Xt_in, A, n)
%
% Computes the n-mode product of the tensor Xt_in and the matrix A.
% This means that all n-mode vectors of Xt_in are multiplied from the left
% with A.
%
% Inputs: Xt_in - tensor
%         A     - matrix
%         n     - dimension
%
% Output: Xt_out - Xt_out = (Xt_in  x_n  A)
I_in = size(Xt_in);

% compute dimensions of Xt_out
I_out = I_in;
I_out(n) = size(A, 1);

% compute n-mode product
ordering = 1;
Xt_out = iunfolding(A*unfolding(Xt_in, n, ordering), n, I_out, ordering);

end
