function [relsqerr, P, D, Uest] = cpderr_RSE(U, Uest)
% Hacked version of cpderr.m from Tensorlab 3.0. Modified to compute RSE
% rather than RE.

%CPDERR_RSE Errors between factor matrices in a CPD.
%   [relsqerr, P, D, Uest] = CPDERR_RSE(U, Uest) computes the relative difference in
%   *squared* Frobenius norm between the factor matrix U{n} and the estimated factor
%   matrix Uest{n} as
%
%      relerr(n) = norm(U{n} - Uest{n} * P * D{n}, 'fro')^2 / norm(U{n}, 'fro')^2
%
%   in which the matrices P and D{n} are a permutation and scaling matrix
%   such that the estimated factor matrix Uest{n} is optimally permuted and
%   scaled to fit U{n}. The optimally permuted and scaled version is
%   returned as fourth output argument. If size(Uest{n},2) > size(U{n},2),
%   then P selects size(U{n},2) rank-one terms of Uest that best match
%   those in U. If size(Uest{n},2) < size(U{n},2), then P pads the rank-one
%   terms of Uest with rank-zero terms. Furthermore, it is important to
%   note that the diagonal matrices D{n} are not constrained to multiply to
%   the identity matrix. In other words, relerr(n) returns the relative
%   error between U{n} and Uest{n} independently from the relative error
%   between U{m} and Uest{m}, where m ~= n.
%
%   See also cpdgen, lmlraerr.

%   Authors: Laurent Sorber      (Laurent.Sorber@cs.kuleuven.be)
%            Nico Vervliet       (Nico.Vervliet@esat.kuleuven.be)
%            Otto Debals         (Otto.Debals@esat.kuleuven.be)
%            Marc Van Barel      (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
% Version History:
% - 2016/03/02   OD      Added Uest output argument
% - 2016/02/07   NV      Fixed error for any size(U{n},1) == 1

% check the factor matrices U and Uest
if ~iscell(U), U = {U};
end
if ~iscell(Uest), Uest = {Uest};
end
if length(U) ~= length(Uest)
    error('cpderr:U', 'length(U) should equal length(Uest).');
end
R = size(U{1}, 2);
Rest = size(Uest{1}, 2);
if any(cellfun('size', U, 2) ~= R) || any(cellfun('size', Uest, 2) ~= Rest)
    error('cpderr:U', 'size(U(est){n},2) should be the same for all n.');
end
if any(cellfun('size', U(:), 1) ~= cellfun('size', Uest(:), 1))
    error('cpderr:U', 'size(U{n},1) should equal size(Uest{n},1).');
end

% compute the congruence between each pair of rank-one terms
N = length(U);
C = ones(Rest, R);
for n = 1:N
    Un = bsxfun(@rdivide, U{n}, sqrt(dot(U{n}, U{n}, 1)));
    Uestn = bsxfun(@rdivide, Uest{n}, sqrt(dot(Uest{n}, Uest{n}, 1)));
    C = C .* abs(Uestn'*Un);
end

% compute the permutation matrix
P = zeros(Rest, R);
for r = 1:R
    [Cr, i] = max(C, [], 1);
    [~, j] = max(Cr);
    P(i(j), j) = 1;
    C(i(j), :) = 0;
    C(:, j) = 0;
end

% compute the scaling matrices and relative errors
D = cell(1, N);
relsqerr = zeros(1, N);
for n = 1:N
    Uestn = Uest{n} * P;
    D{n} = diag(conj(dot(U{n}, Uestn, 1)./dot(Uestn, Uestn, 1)));
    D{n}(~isfinite(D{n})) = 1;
    Uestnd = Uestn * D{n};
    if nargout > 3, Uest{n} = Uestnd; end
    relsqerr(n) = norm(U{n}-Uestnd, 'fro')^2 / norm(U{n}, 'fro')^2;
end
if nargout > 3 && numel(Uest) == 1, Uest = Uest{1}; end
