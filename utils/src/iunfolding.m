function Xt = iunfolding(X_n, n, I, ordering)
% Reconstructs a tensor out of its n-mode unfolding.
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
% |     Modifications:
% |     13.06.2019 - elaborated on description of column orderings (MG)
% |     03.07.2019 - minor edits on notation (MG)
% |----------------------------------------------------------------
%
% Xt = iunfolding(X_n, n, I, ordering)
%
% reproduces the origin tensor Xt out of its n-mode unfolding
% Matrix X_n, produced by the function unfolding (type 'help unfolding'
% to get additional informations). Therefore, the dimensions of the
% origin tensor have to be given by the vector sizes. The optional
% parameter order is the ordering used by the unfolding command. If order
% is not given, the function assumes ordering = 3 (Lathauwer unfolding).
% Otherwise the following ordering of the columns is assumed:
%
%   ordering = 1: forward column ordering (MATLAB, column-major)
%       indices of n - mode vectors go faster with increasing index
%   ordering = 2: reverse column ordering (Tensorly, row-major)
%       indices of n - mode vectors go slower with increasing index
%   ordering = 3: reverse cyclical, bc (de Lathauwer)
%       indices go slower with I_n+1, ... I_N, I_1, ... I_n-1
%   ordering = 4: forward cyclical, fc (flipped de Lathauwer)
%       indices go slower with I_n-1, ... I_1, I_N, ... I_n+1
%
% Inputs: X_n      - matrix with the n-mode vectors of a tensor Xt
%         n        - dimension
%         sizes    - vector containg the size of Xt
%         ordering - defines the ordering of the n-mode vectors (optional)
%
% Output: Xt       - reproduced tensor

% get dimension
N = length(I);

% make singletons at the end of Xt possible
if n > N
    I = [I, ones(1, n-N)];
    N = n;
end

% set standard Lathauwer unfolding
if nargin == 3
    ordering = 3;
end

% get permutation vector
switch ordering
    case 1
        permute_vec = [n, 1:(n - 1), (n + 1):N]; % indices go faster with increasing index
        [~, ipermute_vec] = sort(permute_vec);
    case 2
        permute_vec = [n, fliplr([1:(n - 1), (n + 1):N])]; % indices go slower with increasing index
        [~, ipermute_vec] = sort(permute_vec);
    case 3
        permute_vec = [n, fliplr(1:(n - 1)), fliplr((n + 1):N)]; % Lathauwer: indices go slower with I_n+1, ... I_N, I_1, ... I_n-1
        [~, ipermute_vec] = sort(permute_vec);
    case 4
        permute_vec = [n, fliplr([fliplr(1:(n - 1)), fliplr((n + 1):N)])]; % flipped Lathauwer
        [~, ipermute_vec] = sort(permute_vec);
    otherwise
        disp('Error: unknown ordering for n--mode vectors');
        return
end

% get origin tensor
Xt = permute(reshape(X_n, I(permute_vec)), ipermute_vec);

end
