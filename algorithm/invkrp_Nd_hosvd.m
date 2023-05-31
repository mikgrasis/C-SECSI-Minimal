function F_hat = invkrp_Nd_hosvd(X, I, usehooi)
% Invert an N-fold Khatri-Rao product via Least Squares Khatri-Rao
% Factorization (LSKRF).
%
% |----------------------------------------------------------------
% | (C) 2019 TU Ilmenau, Communications Research Laboratory
% |
% |     Florian Roemer
% |
% |     Advisors:
% |        Dipl.-Ing. Giovanni Del Galdo
% |        Univ.-Prof. Dr.-Ing. Martin Haardt
% |
% |     Date authored: Mar 2009
% |     Modifications:
% |     19.09.2019 - facelift on formatting and variable names (MG)
% |-----------------------------------------------------------------
%
% F_hat = invkrp_Nd_hosvd(X, I, usehooi)
%
% decomposes the matrix X into F_1 \krp F_2 \krp ... \krp F_N, where
% size(F_n) = [I(n), R] and size(X) = [prod(I), R].
%
% Inputs:
%   X       - matrix (double)
%   I       - number of rows of individual matrices
%   usehooi - true / {false}: Use HOOI for optimal truncated Tucker?
%
% Output:
%   F_hat   - factors as cell array of length N, where F_hat{n} is I(n) x R
if prod(I) ~= size(X, 1)
    error('X should be of size [PROD(I), R]!');
end
if nargin < 3
    usehooi = false;
end

N = length(I);
R = size(X, 2);

F_hat = cell(1, N);

if N == 1
    F_hat{1} = X;
else
    if any(isnan(X))
        for n = 1:N
            F_hat{n} = nan(I(n), R);
        end
    else
        for n = 1:N
            F_hat{n} = zeros(I(n), R);
        end
        for r = 1:R
            X_r = X(:, r);
            Xt_r = reshape(X_r, I(end:-1:1));
            [St, U, SD] = hosvd_CRL(Xt_r); %#ok<ASGLU>
            if usehooi && (N > 2)
                [U_cut, Xt_r] = opt_dimred(Xt_r, 1);
                St_cut = core_tensor(Xt_r, U_cut);
            else
                [St_cut, U_cut] = cuthosvd(St, U, 1);
            end
            U_cut = U_cut(end:-1:1);
            for n = 1:N
                F_hat{n}(:, r) = U_cut{n};
            end
            F_hat{1}(:, r) = F_hat{1}(:, r) * St_cut;
        end
    end
end
end
