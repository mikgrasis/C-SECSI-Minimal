function [F_hat, ordering] = fix_permutation_by_angle(F_ref, F_hat)
% Rearranges column ordering of a set of factor matrices F{n}, n = 1,...,N,
% that define a tensor of order N.
%
% Inputs:
%   F_hat     - cell array with estimated factors F_hat{n}, n = 1,...,N
%   F_ref     - cell array with original factors F_ref{n}, n = 1,...,N
%
% Outputs:
%   F_hat     - cell array with rearranged factors F_hat{n}, n = 1,...,N
%   ordering  - ordering of the columns
%
% Reference
%
% Mikus Grasis, CRL, September 2017 extracted from SECSI.m by Florian Roemer
if ~iscell(F_hat)
    F_hat = {F_hat};
    F_ref = {F_ref};
end

N = length(F_hat);

for n = 1:N
    assert(eq(size(F_ref{n}, 1), size(F_hat{n}, 1)), 'dimension mismatch: matrices must have equal number of rows');
    assert(eq(size(F_ref{n}, 2), size(F_hat{n}, 2)), 'dimension mismatch: matrices must have equal number of cols');
end

temp = cell(1, N);
ordering = cell(1, N);

for n = 1:N
    % normalize factors
    temp{n} = F_hat{n} * diag(1./sqrt(sum(abs(F_hat{n}).^2, 1)));
    %F_hat{n} = F_hat{n} * diag(1./sqrt(sum(abs(F_hat{n}).^2, 1)));
    F_ref{n} = F_ref{n} * diag(1./sqrt(sum(abs(F_ref{n}).^2, 1)));
    R_n = size(temp{n}, 2);

    % fix permutation
    fromassign = 1:R_n;
    toassign = 1:R_n;
    ordering{n} = zeros(1, R_n);
    for r = 1:R_n
        v = abs(temp{n}(:, toassign)' * F_ref{n}(:, fromassign));
        %v = abs(F_hat{n}(:, toassign)' * F_ref{n}(:, fromassign));
        [~, max_v] = max(v(:));
        [mx, my] = ind2sub([R_n - r + 1, R_n - r + 1], max_v);
        ordering{n}(fromassign(my)) = toassign(mx);
        toassign = toassign([1:mx - 1, mx + 1:end]);
        fromassign = fromassign([1:my - 1, my + 1:end]);
    end
    
    % output
    F_hat{n} = F_hat{n}(:, ordering{n});
end
