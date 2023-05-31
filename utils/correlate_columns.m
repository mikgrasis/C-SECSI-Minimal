function A = correlate_columns(A, opts)
%function A = correlate_columns(A, opts)
% Correlates cols of matrix with the specified method and options.
%
% |----------------------------------------------------------------
% | (C) 2018 Mikus Grasis
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |
% |     Date authored: August 2018
% |----------------------------------------------------------------
%
% Example 1 (one matrix):
%   opts.sType = 'toeplitz';
%   opts.rho = 0.9;
%   A_corr = correlate_columns(A, opts);
%
% Example 2 (several matrices):
%   A = generate_random_factors([2, 3, 4], 2, 'gaussian');
%   opts.sType = 'toeplitz';
%   opts.rho = [0, 0.5, 0.9];
%   A_corr = correlate_columns(A, opts);
%
% Input:
%   A       - matrix/cell of size I x R or I(n) x R
%   opts    - struct for options
%       opts.rho    - correlation parameter
%       opts.sType  - method/structure of correlation matrix, e.g., onesMain | toeplitz |�fantasy
%
%   Note that 'fantasy' simply right multiplies A with the specified correlation matrix.
%
% Output:
%   A - matrix/cell with correlated columns
%
% Reference:
%   https://de.mathworks.com/matlabcentral/answers/101802-how-can-i-generate-two-correlated-random-vectors-with-values-drawn-from-a-normal-distribution
%
opts.rho = setparam(opts, 'rho', 0.9);
opts.sType = setparam(opts, 'sType', 'fantasy');

if ~iscell(A)
    A = {A};
end
N = length(A);
assert(eq(length(opts.rho), N), 'oh-oh')
for n = 1:N
    if opts.rho(n)
        R = size(A{n}, 2); % number of cols

        % generate correlation matrix
        R_corr = correlation_matrix(R, opts.rho(n), opts.sType);

        % apply correlation matrix
        if strcmp(opts.sType, 'fantasy')
            A{n} = A{n} * R_corr; % note: just multiplying does not result in matrices A{n} having the specified correlation between columns!
        else
            % use R matrix from Cholesky decomposition to apply correlation to factor matrix
            A{n} = A{n} * chol(R_corr);
        end
    end
end
if eq(N, 1)
    A = A{1};
end
end
