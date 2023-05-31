function R_cor = correlation_matrix(d, rho, sType)
% Prepare correlation matrix for synthetic data.
%
% |----------------------------------------------------------------
% | (C) 2018 Mikus Grasis
% |
% |     Advisors:
% |         Univ. Prof. Dr.-Ing. Martin Haardt
% |         Prof. Andre Lima Ferrer de Almeida
% |
% |     Date authored: August 2018
% |     Last Modifications: 01.06.2019
% |         -added fantasy method from CRL papers
% |         -header and code formatting
% |-----------------------------------------------------------------
%
% R_cor = correlation_matrix(5, 0.9, 'onesMain')
%
% returns 5 x 5 correlation matrix with ones on main diagonal and rho
% elsewhere.
%
% Inputs:   d       - size of matrix (d x d)
%           rho     - correlation coeff
%           sType   - string with method to use: 'onesMain' | 'toeplitz' | 'fantasy'
% Output:
%           R_cor   - correlation matrix

% Default method
if nargin < 3
    sType = 'onesMain';
end

% Correlation matrix
switch sType
    case 'onesMain'
        % correlation matrix with ones on main diagonal and rho elsewhere
        R_cor = rho * ones(d, d);
        for r = 1:d
            R_cor(r, r) = 1;
        end
    case 'toeplitz'
        % toeplitz matrix with decreasing exponents on diagonals
        pow = 0:d - 1;
        ex = toeplitz(pow, pow);
        R_cor = (rho * ones(d, d)).^ex;
    case 'fantasy'
        % correlation matrix from CRL papers
        R_cor = (1 - rho) * eye(d) + rho / d * ones(d, d);
end
end
