function Xt = cp_construct(A, lambda)
% CP_CONSTRUCT   Build an N-D CP tensor from its loading matrices
%
% Syntax:
%    Xt = CP_CONSTRUCT(A) where A is a cell array of length N
%    containing the loading matrices for each of the modes n = 1, ..., N.
%    The size of each loading matrix should be [I(n), R] where R is the
%    number of components (i.e., the rank of the tensor) and Xt will be of
%    size [I(1), ..., I(N)].
%
% Author:
%    Florian Roemer, Communications Resarch Lab, TU Ilmenau
%
% Date:
%    Dec 2007
% Notation: order N, rank R, dimensions I(1),...,I(N)
if nargin < 2
    R = size(A{1}, 2);
    lambda = ones(R, 1);
end

N = length(A);

if N == 3
    % faster for N = 3
    Xt = iunfolding(A{3}*diag(lambda)*krp(A{1}, A{2}).', 3, [size(A{1}, 1), size(A{2}, 1), size(A{3}, 1)]);
elseif N == 4
    % faster for N = 4
    Xt = iunfolding(A{4}*diag(lambda)*krp_Nd(A(1:3)).', 4, [size(A{1}, 1), size(A{2}, 1), size(A{3}, 1), size(A{4}, 1)]);
elseif N == 1
    % we want to be robust enough to catch this
    Xt = A{1} * lambda;
else
    I = zeros(1, N);
    [I(1), R] = size(A{1});

    for n = 2:N
        if size(A{n}, 2) ~= R
            error('The number of columns must agree for all the factors');
        end
        I(n) = size(A{n}, 1);
    end

    Xt = iunfolding(A{N}*diag(lambda)*krp_Nd(A(1:N-1)).', N, I);
end
end
