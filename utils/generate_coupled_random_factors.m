function F = generate_coupled_random_factors(I, R, coupling, opts_fac)
% Generate set of random factors for coupled tensor experiments.
%
% |----------------------------------------------------------------
% | (C) 2021 Mikus Grasis
% |
% |  ______   ______     __   __     ______     ______     ______        ______   ______     ______     __         ______
% | /\__  _\ /\  ___\   /\ "-.\ \   /\  ___\   /\  __ \   /\  == \      /\__  _\ /\  __ \   /\  __ \   /\ \       /\  ___\
% | \/_/\ \/ \ \  __\   \ \ \-.  \  \ \___  \  \ \ \/\ \  \ \  __<      \/_/\ \/ \ \ \/\ \  \ \ \/\ \  \ \ \____  \ \___  \
% |    \ \_\  \ \_____\  \ \_\\"\_\  \/\_____\  \ \_____\  \ \_\ \_\       \ \_\  \ \_____\  \ \_____\  \ \_____\  \/\_____\
% |     \/_/   \/_____/   \/_/ \/_/   \/_____/   \/_____/   \/_/ /_/        \/_/   \/_____/   \/_____/   \/_____/   \/_____/
% |
% |
% |     Advisors:
% |         Univ.-Prof. Dr.-Ing. Martin Haardt
% |         Prof. Andre Lima Ferrer de Almeida
% |
% |     Date authored:  20.05.2021
% |     Modifications:
% |     11.08.2021 - include opts_fact.select_tensor for correlation (MG)
% |     19.06.2021 - works for coupled Tucker (MG)
% |     20.05.2021 - initial version (MG)
% |----------------------------------------------------------------
%
% Usage:
%   I{1} = [2, 3, 4]; I{2} = [2, 4, 5]; R = 3;
%   coupling = [1, 0, 0];
%   opts_fac.factors_type = 'complexGaussian';
%   F = generate_coupled_random_factors(I, R, coupling, opts_fac)
%
% generates a cell array with L sets of N(l) random matrices of size
% I{l}(n) x R{l}(n) for n = 1,...,N(l).
%
% Examples:
% 1) Simple Mode (Binary Vector)
%     I = [];
%     I{1} = [4, 8, 7];
%     I{2} = [4, 7, 9];
%     I{3} = [4, 3, 2];
%
%     R = [];
%     R{1} = [3, 2, 1];
%     R{2} = [1, 2, 3];
%     R{3} = [4, 3, 2];
%
%     coupling = [1, 0, 0];
%     opts_fac.factors_type = 'complexGaussian';
%     F = generate_coupled_random_factors(I, R, coupling, opts_fac);
%
% 2) Intermediate (Modes Vector + Components)
%     I = [];
%     I{1} = [4, 8, 7];
%     I{2} = [4, 3];
%
%     R = [];
%     R{1} = [3, 2, 1];
%     R{2} = [3, 7];
%
%     coupling = struct;
%     coupling.modes = [1]; %#ok<NBRAK>
%     coupling.components = {[1, 1, 0]};
%     opts_fac.factors_type = 'complexGaussian';
%
%     F = generate_coupled_random_factors_beta(I, R, coupling, opts_fac);
%
% 3) Advanced (Flexible Components)
%     I = [];
%     I{1} = [4, 8, 7];
%     I{2} = [4, 3];
%     I{3} = [4, 5, 7, 6];
%     L = numel(I);
%
%     R = 3;
%
%     coupling = cell(1, L);
%     coupling{1} = {[1, 0, 0], [], [1, 1, 1]};
%     coupling{2} = {[0, 1, 0], []};
%     coupling{3} = {[1, 1, 0], [], [1, 0, 1], []};
%
%     opts_fac.factors_type = 'complexGaussian';
%     F = generate_coupled_random_factors(I, R, coupling, opts_fac);
%
% Inputs:
%   I   - dimensions
%   R   - ranks
%       -> a) one rank for all (as in CP-model)
%       -> b) cell array with one set of n-ranks for each tensor
%           note: coupling must match number of components (n-ranks)!
%   coupling - specification of coupling, can be vector/struct/cell (cf. parse_coupling.m)
%   opts_fac - struct with options
%       opts_fac.factors_type - type of factor matrices to generate (cf. generate_random_factors.m)
%       opts_fac.select_tensor - select tensors where to add correlation (binary vector of length L, defaults to ones)
%
% Output:
%   F  - cell array with coupled factors F{l}{n}, n = 1,...,N(l), l = 1,...,L
%
% Notation:
%   number of factor sets L, tensor orders N(1),...,N(L), dimensions I{l}(1),...,I{l}(N(l))
%   ranks R{l}(1),...,R{l}(N(l))

%% Handle Inputs
if nargin < 4
    opts_fac = struct;
end
opts_fac.factors_type = setparam(opts_fac, 'factors_type', 'gaussian');

L = numel(I); % number of factor sets
opts_fac.select_tensor = setparam(opts_fac, 'select_tensor', ones(1, L));

% get order of involved tensors
N = nan(1, L);
for l = 1:L
    N(l) = length(I{l});
end

% expand rank in case provided as scalar
if isscalar(R)
    R_old = R;
    R = cell(1, L);
    for l = 1:L
        R{l} = ones(1, N(l)) * R_old;
    end
    clear R_old
end

%% Parse Coupling
output_format = 3;
coupling = parse_coupling(I, R, coupling, output_format);

%% Factor Generation
F = cell(1, L);

% determine max mode length and max number of columns in each mode
I_max = zeros(1, max(N));
R_max = zeros(1, max(N));
for n = 1:max(N) % look at one mode at a time, until largest
    relevant_l = find(ge(N, n)); % see what tensors have this many modes at all
    for l = relevant_l
        if gt(I{l}(n), I_max(n))
            I_max(n) = I{l}(n);
        end
        if gt(R{l}(n), R_max(n))
            R_max(n) = R{l}(n);
        end
    end
end

% generate 'master' factors to copy for coupled modes
F_copy = generate_random_factors(I_max, R_max, opts_fac.factors_type, opts_fac);

% generate intial random factors for all modes
for l = 1:L
    opts_fac_l = opts_fac;
    % this is a quick fix and might cause all kinds of messy constellations
    % TODO: maybe better generate all factors without correlation, then
    % apply correlation afterwards (e.g., at the end this function or
    % elaborate the add_correlation function to handle such cases)
    if ~opts_fac.select_tensor(l)
        opts_fac_l.rho = zeros(1, N(l));
    end
    F{l} = generate_random_factors(I{l}, R{l}, opts_fac.factors_type, opts_fac_l);
end

% copy coupled modes from master factors
for n = 1:max(N)
    relevant_l = find(ge(N, n));

    % go thru components and copy from master factors
    for l = relevant_l
        for r = 1:R{l}
            if ~isempty(coupling{l}{n}) % no coupling, no action
                if coupling{l}{n}(r)
                    F{l}{n}(:, r) = F_copy{n}(:, r);
                end
            end
        end
    end
end
end
