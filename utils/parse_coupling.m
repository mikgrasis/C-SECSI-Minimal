function coupling = parse_coupling(I, R, coupling, output_format)
% Checks and converts specification of coupling.
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
% |
% |     Date authored:  22.05.2021 (MG)
% |     Modifications:
% |     04.09.2021 - bugfix components verification for intermediate case (MG)
% |     04.09.2021 - bugfix on upgrade intermediate -> advanced (MG)
% |     22.05.2021 - initial version (MG)
% |----------------------------------------------------------------
%
% Examples:
% 1) Simple Mode (Binary Vector)
%   I{1} = [2, 3, 4]; I{2} = [2, 4, 5]; R = 3;
%   coupling = [1, 0, 0]; output_format = 1;
%   parse_coupling(I, R, coupling, output_format);
%
% checks the validity of the dimensions and the specified coupling.
%
%   output_format = 3;
%   coupling = parse_coupling(I, R, coupling, output_format)
%
% checks the validity of the dimensions and the specified coupling and
% outputs the coupling according to output format 3 (flexible components).
%
% 2) Intermediate (Modes Vector + Components)
%   XXX
%
% 3) Advanced (Flexible Components)
%   XXX
%
% Conversion Rules:
%   Advanced -> Intermediate/Basic
%       If there is agreement between any two tensors, the mode counts as
%       coupled.
%
%       Example 1:
%           coupling = cell(1, L);
%           coupling{1} = {[1, 0, 0], [], [1, 1, 1]};
%           coupling{2} = {[0, 1, 0], [], [1, 1, 1]};
%           coupling{3} = {[1, 1, 0], [], [1, 0, 1]};
%       converts to
%           coupling.modes = [1, 3];
%           coupling.components = {[1, 1, 0], [1, 0, 1]};
%       in intermediate mode, or
%           coupling = [1, 0, 1];
%       in basic mode.
%
%       Example 2:
%           coupling = cell(1, L);
%           coupling{1} = {[1, 0, 0], [], [1, 1, 1]};
%           coupling{2} = {[0, 1, 0], []};
%           coupling{3} = {[1, 1, 0], [], [1, 0, 1], []};
%       converts to
%           coupling.modes = [1, 3];
%           coupling.components = {[1, 1, 0], [1, 0, 1]};
%       in intermediate mode. Conversion to basic mode is not possible in
%       this example, since ne(N(1), N(2), N(3)).
%
%   Advanced -> Intermediate
%       If there is agreement between any two components, the component
%       counts as coupled.
%
%       Example 1:
%           coupling = cell(1, L);
%           coupling{1} = {[1, 0, 0], [], []};
%           coupling{2} = {[0, 1, 0], [], []};
%           coupling{3} = {[1, 1, 0], [], []};
%       converts to
%           coupling.modes = [1];
%           coupling.components = [1, 1, 0];
%       in intermediate mode, or
%           coupling = [1, 0, 0];
%       in basic mode.
%
%       Example 2:
%           coupling = cell(1, L);
%           coupling{1} = {[1, 0, 0], [], []};
%           coupling{2} = {[0, 1, 0], [], []};
%       converts to
%           coupling.modes = [];
%           coupling.components = [0, 0, 0];
%       in intermediate mode, or
%           coupling = [0, 0, 0];
%       in basic mode.
%
%   Advanced/Intermediate -> Basic
%       If any component is coupled, the mode counts as coupled.
%
% Inputs:
%   I   - dimensions
%   R   - ranks
%       -> a) one rank for all (as in CP-model)
%       -> b) cell array with one set of n-ranks for each tensor
%           note: coupling must match number of components (n-ranks)!
%   coupling - coupling
%       output_format - flag for output format
%       1) simple (binary vector)
%       2) intermediate (struct)
%       3) advanced (cell)
%
% Output:
%   coupling - (reformatted) specification (vector/struct/cell)
%
% Notation:
% number of factor sets L, orders N(1),...,N(L),
%   dimensions I{l}(1),...,I{l}(N(l)), ranks R{l}(1),...,R{l}(N(l))

%% Handle Inputs
if nargin < 4
    output_format = [];
end

% number of factor sets
L = numel(I);

% order of involved tensors
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

% check if same tensor order for all
same_order = true;
for l = 2:L
    if ne(length(I{l}), length(I{1}))
        same_order = false;
        break
    end
end

%% Parse Coupling
if isnumeric(coupling) && eq(length(coupling), N(1))
    %% Easy mode: Binary Vector
    assert(same_order, 'easy mode: all tensors must be of the same order')
    % verify dimensions
    for n = find(coupling)
        for l = 2:L
            assert(eq(I{l}(n), I{1}(n)), 'easy mode: size in coupled dimension must be equal for all tensors');
        end
    end

    % format output
    if output_format == 3
        % convert: basic -> advanced
        coupling_old = coupling;
        coupling = cell(1, L);
        for l = 1:L
            coupling{l} = cell(1, N(l));
            for n = 1:N(l)
                if coupling_old(n)
                    coupling{l}{n} = ones(1, R{l}(n));
                end
            end
        end
        clear coupling_old
    elseif output_format == 2
        % convert: basic -> intermediate
        coupling_old = coupling;
        coupling = struct;
        coupling.modes = find(coupling_old);
        clear coupling_old
    elseif output_format == 1
        % do nothing
    end


elseif isfield(coupling, 'modes')
    %% Intermediate Mode: Modes Vector + Components
    num_coupled_modes = length(coupling.modes);
    if isfield(coupling, 'components')
        assert(eq(numel(coupling.components), num_coupled_modes), 'intermediate: components must be cell of equal length as coupling.modes')
    end
    % verify dimensions
    for c = 1:num_coupled_modes
        for l = 2:L
            assert(eq(I{l}(coupling.modes(c)), I{1}(coupling.modes(c))), 'intermediate: size in coupled dimension must be equal for all tensors');
        end
    end
    % verify components
    if isfield(coupling, 'components')
        for c = 1:num_coupled_modes
            for l = 1:L
                assert(eq(length(coupling.components{c}), R{l}(coupling.modes(c))), 'intermediate: the number of specified components must match the number of columns')
            end
        end
    end

    % format output
    if output_format == 3
        % convert: intermediate -> advanced
        coupling_old = coupling;
        coupling = cell(1, L);
        for l = 1:L
            coupling{l} = cell(1, N(l));
            for c = 1:num_coupled_modes
                if isfield(coupling_old, 'components')
                    coupling{l}{coupling_old.modes(c)} = coupling_old.components{c};
                else
                    coupling{l}{coupling_old.modes(c)} = ones(1, R{l}(coupling_old.modes(c)));
                end
            end
        end
        clear coupling_old
    elseif output_format == 2
        % do nothing
    elseif output_format == 1
        % convert: intermediate -> basic
        assert(same_order, 'intermediate -> basic: all tensors must be of the same order')
        coupling_old = coupling;
        coupling = zeros(1, N(1));
        coupling(coupling_old.modes) = 1;
        if isfield(coupling, 'components')
            warning('intermediate -> basic: component information will be lost!')
        end
        clear coupling_old
    end


elseif iscell(coupling)
    %% Advanced Mode: Flexible Components (one cell array per tensor/factor set)
    % verify input format and number of components
    assert(eq(numel(coupling), L), 'advanced: coupling must be cell with one cell for each tensor!')
    for l = 1:L
        assert(eq(numel(coupling{l}), N(l)), 'advanced: for each tensor, coupling must be one cell of length N(l)')
        for n = 1:N(l)
            if any(coupling{l}{n}) % no coupling, no action
                assert(eq(length(coupling{l}{n}), R{l}(n)), 'advanced: in each coupled mode, the number of components must match')
            end
        end
    end
    % verify dimensions for coupled components
    for n = 1:max(N)
        relevant_l = find(ge(N, n));
        first_occurence = true;
        for l = relevant_l
            if any(coupling{l}{n})
                if first_occurence
                    first_occurence = false;
                    l_first = l;
                else
                    assert(eq(I{l}(n), I{l_first}(n)), 'advanced: in each coupled mode, the dimensions must agree')
                end
            end
        end
    end

    % format output
    if output_format == 3
        % do nothing
    elseif output_format == 2
        %%% convert: advanced -> intermediate
        coupling_old = coupling;
        coupling = struct;
        % Note: Yes, there still is some flaws to this logic, e.g., the
        % both steps could be merged into one, and downgrading to
        % intermediate only makes sense if the number of components agrees
        % for the coupled modes. However, it works...

        % step 1 - detect coupled modes
        coupled_modes = zeros(1, max(N));
        for n = 1:max(N)
            % see what tensors have this many modes at all
            relevant_l = find(ge(N, n));

            % out of those, check for largest number of components in current mode
            R_max = 0;
            for l = relevant_l
                if gt(R{l}(n), R_max)
                    R_max = R{l}(n);
                end
            end

            % go thru components and find agreements
            for r = 1:R_max
                first_occurence = true;
                for l = relevant_l
                    if any(coupling_old{l}{n}) % no coupling, no action
                        if coupling_old{l}{n}(r) && first_occurence
                            first_occurence = false;
                        elseif coupling_old{l}{n}(r)
                            coupled_modes(n) = 1;
                        end
                    end
                end
            end
        end
        coupling.modes = find(coupled_modes);
        num_coupled_modes = length(coupling.modes);

        % step 2 - go thru the coupled modes and save coupled components
        coupling.components = cell(1, length(coupling.modes));
        for curr_coupled_mode = 1:num_coupled_modes
            n = coupling.modes(curr_coupled_mode);

            relevant_l = find(ge(N, n));
            R_max = 0;
            for l = relevant_l
                if gt(R{l}(n), R_max)
                    R_max = R{l}(n);
                end
            end

            % go thru components and mark them as coupled where appropriate
            coupling.components{curr_coupled_mode} = zeros(1, R_max);
            for r = 1:R_max
                first_occurence = true;
                for l = relevant_l
                    if any(coupling_old{l}{n}) % no coupling, no action
                        if coupling_old{l}{n}(r) && first_occurence
                            first_occurence = false;
                        elseif coupling_old{l}{n}(r)
                            coupling.components{curr_coupled_mode}(r) = 1;
                        end
                    end
                end
            end
        end
        clear coupling_old
    elseif output_format == 1
        assert(same_order, 'advanced -> basic: all tensors must be of the same order')
        % convert advanced -> basic
        coupling_old = coupling;
        coupling = zeros(1, N(1));
        for n = 1:N(1)
            % check for largest number of components in current mode
            R_max = 0;
            for l = 1:L
                if gt(R{l}(n), R_max)
                    R_max = R{l}(n);
                end
            end

            % go thru components and find agreements
            for r = 1:R_max
                first_occurence = true;
                for l = 1:L
                    if le(r, R{l}(n)) && any(coupling_old{l}{n}) % no coupling, no action
                        if coupling_old{l}{n}(r) && first_occurence
                            first_occurence = false;
                        elseif coupling_old{l}{n}(r)
                            coupling(n) = 1;
                        end
                    end
                end
            end
        end
    end
else
    error('that''s not how it works!')
end
end
