function X_n = unfolding(Xt, n, ordering)
% Computes the n-mode unfolding of a tensor.
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
% |     Date authored: 04.23.2007
% |     Last modifications:
% |     03.07.2019 - minor edits on notation (MG)
% |     13.06.2019 - elaborated on description of column orderings (MG)
% |----------------------------------------------------------------
%
% X_n = unfolding(Xt, n)
%
% computes a matrix out that contains all n-mode vectors
% of the given tensor T, with an ordering as defined by de Lathauwer.
%
% X_n = unfolding(Xt, n, ordering)
%
% computes the matrix of n-mode vectors with the following ordering:
%
%   ordering = 1: forward column ordering (MATLAB)
%       indices of n-mode vectors go faster with increasing index
%   ordering = 2: reverse column ordering (Python, C++)
%       indices of n-mode vectors go slower with increasing index
%   ordering = 3: reverse cyclical, bc (de Lathauwer)
%       indices go slower with I_n+1, ... I_N, I_1, ... I_n-1
%   ordering = 4: forward cyclical, fc (flipped de Lathauwer)
%       indices go slower with I_n-1, ... I_1, I_N, ... I_n+1
%
% Inputs: Xt       - tensor (double)
%         n        - dimension
%         ordering - defines the ordering of the n-mode vectors (optional, defaults to 3)
%
% Output: X_n      - n-mode unfolding matrix of Xt

% Get dimensions
I = size(Xt);
dimension = length(I);

% Make singletons at the end of Xt possible
if n > dimension
    I = [I, ones(1, n-dimension)];
    dimension = n;
end

% Set standard Lathauwer unfolding
if nargin == 2
    ordering = 3;
end

% Permute tensor Xt for reshape - command
switch ordering
    case 1
        Xt = permute(Xt, [n, 1:(n - 1), (n + 1):dimension]); % indices go faster with increasing index
    case 2
        Xt = permute(Xt, [n, fliplr([1:(n - 1), (n + 1):dimension])]); % indices go slower with increasing index
    case 3
        Xt = permute(Xt, [n, fliplr(1:(n - 1)), fliplr((n + 1):dimension)]); % Lathauwer: indices go slower with I_n+1, ... I_N, I_1, ... I_n-1
    case 4
        Xt = permute(Xt, [n, fliplr([fliplr(1:(n - 1)), fliplr((n + 1):dimension)])]); % flipped Lathauwer
    otherwise
        disp('Error: unknown ordering for n--mode vectors');
        return
end

% Compute n-mode unfolding
X_n = reshape(Xt, [I(n), prod(I) ./ I(n)]);

end
