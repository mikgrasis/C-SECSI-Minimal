function [A_hat, lambda_hat, infos] = coupled_CPD_algorithms_v2(Xt, R, coupling, method, opts, VERBOSE)
% Collection of algorithms for the coupled CP-decomposition.
%
% |----------------------------------------------------------------
% | (C) 2021 Mikus Grasis, Alla Manina
% |
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
% |     Date authored: 24.03.2020
% |     Modifications:
% |     14.09.2022 - new method 'C-SECSI-Beta' (MG)
% |     13.09.2022 - new method 'C-SECSI-Minimal' (MG)
% |     07.09.2022 - copy+pasta friendly parameter descriptions (MG)
% |     12.08.2021 - move coupling into separate input (AM/MG)
% |     26.05.2021 - code review (MG)
% |----------------------------------------------------------------
%
% [A_hat, lambda_hat] = coupled_CPD_algorithms_v2(Xt, R, coupling, 'C-ALS-PV')
%
% fits L coupled rank-R CP-models to the input tensors Xt{l}, l = 1,...,L
% using the 'C-ALS-PV' algorithm with default parameters.
%
% Inputs:
%   Xt       - Cell with tensors Xt{l}, l = 1,...,L. Input formats are converted
%              according to the specified algorithm. Typical formats are:
%              MATLAB multidimensional array and Tensor Toolbox 'tensor'.
%   R        - rank of CP-model
%   coupling - coupling (cf. parse_coupling.m)
%   method   - string indicating which algorithm to use
%   opts     - struct with parameters for the called algorithm
%   VERBOSE  - switch for verbosity
%
% Outputs:
%   A_hat       - 1 x L cell of 1 x N cells of (normalized) factor matrices
%   lambda_hat  - 1 x L cell with estimated component weight vectors
%   infos       - struct with information
%
% Note: The following methods work for 3rd order tensors only.
%   'C-TALS-PV', 'C-SECSI-Minimal'
%
% Compatible algorithms (not exhaustive):
%   For L = 2 dense data third-order tensors
%       algs = {'C-Tucker-CP', 'CPD_v2', 'C-ALS-PV', 'C-TALS-PV', 'C-SECSI-5', 'C-SECSI', 'C-SECSI-Beta', 'C-SECSI-Minimal', 'CCP_NLS', 'CCP_MINF'}
%   For L dense data third-order tensors
%       algs = {'C-Tucker-CP', 'CPD_v2', 'C-SECSI', 'C-SECSI-Beta', 'C-SECSI-Minimal', 'CCP_NLS', 'CCP_MINF'}
%
% Collected algorithms:
% %%% 0) 'C-Tucker-CP'   - apply Tucker compression, decompose then decompress [Bro 1998]
% if(0)
%     algorithms{end + 1} = 'C-Tucker-CP';
%     str_legend{end + 1} = 'C-ST-HOSVD/C-SECSI';
%     temp = [];
%     temp.outerAlg = 'C-ST-HOSVD';     % cf. coupled_Tucker_algorithms_v2.m
%     temp.innerAlg = 'C-SECSI';        % cf. coupled_CPD_algorithms_v2.m
%     temp.outerAlgOpts = []; % will be passed to outerAlg, e.g., temp.outerAlgOpts.printitn = 0;
%     temp.innerAlgOpts = []; % will be passed to innerAlg, e.g., temp.innerAlgOpts.printitn = 0;
%     opts{end + 1} = temp;
% end
%
% %%% 1) 'CPD_v2'     - call any algorithm using CPD_algorithms_v2 wrapper individually (without the coupling)
% if(0)
%     algorithms{end + 1} = 'CPD_v2';
%     str_legend{end + 1} = 'SECSI';
%     temp = [];
%     temp.cp_alg = 'SECSI';
%     temp.cp_alg_opts = []; % this is passed on to the selected CP method, e.g., opts.cp_alg_opts.diag.DisplayWarnings = 0;
%     opts{end + 1} = temp;
% end
%
% %%% 2) 'C-ALS-PV'     - Plain-Vanilla ALS algorithm
% if(0)
%     algorithms{end + 1} = 'C-ALS-PV';
%     str_legend{end + 1} = 'C-ALS';
%     temp = [];
%     temp.MAXIT = 1000;
%     temp.delta = 1e-8;
%     temp.termination = 'all'; % all | any
%     temp.fast == 2;     %  0  - pinv(A_hat_minus_n_all.')
%                         %  1  - A_hat_minus_n_all / GRAM
%                         % (2) - MTTKRP via summation (best)
%     opts{end + 1} = temp;
% end
%
% %%% 2b) 'C-TALS-PV'    - Plain-vanilla ALS algorithm for third order tensors legacy-version from thesis of K. Naskovska
% if(0)
%     algorithms{end + 1} = 'C-TALS-PV';
%     str_legend{end + 1} = 'C-ALS';
%     temp = [];
%     temp.MAXIT = 1000;
%     opts{end + 1} = temp;
% end
%
% %%% 3a) 'C-SECSI-5'    - legacy implementation of Coupled-SECSI from Kristina [1]
% if(0)
%     algorithms{end + 1} = 'C-SECSI-5';
%     str_legend{end + 1} = 'C-SECSI (REC PS)';
%     temp = [];
%     temp.heuristic = 'REC PS'; % BM | (REC PS) | CON PS | RES
%     temp.usehooi = false;
%     temp.overridecf2 = false;
%     temp.whichsmds = [];    % 'all' | 'bestcond'
%     temp.selsmds = [];      % 'all' | 'bestres'
%     temp.selfinal = [];     % 'bm' | 'ps'
%     temp.solveperm = [];    % 'cp amp' | 'ang loadvec'
%     temp.bestn_cond = [];
%     temp.bestn_res = [];
%     temp.diag.DisplayWarnings = 1;
%     opts{end + 1} = temp;
% end
%
% %%% 3b) 'C-SECSI-JEVD'    - legacy implementation of Coupled-SECSI from Kristina [1]
% if(0)
%     algorithms{end + 1} = 'C-SECSI-JEVD';
%     str_legend{end + 1} = 'C-SECSI (REC PS)';
%     temp = [];
%     temp.heuristic = 'REC PS'; % BM | (REC PS) | CON PS | RES
%     temp.usehooi = false;
%     temp.overridecf2 = false;
%     temp.whichsmds = [];    % 'all' | 'bestcond'
%     temp.selsmds = [];      % 'all' | 'bestres'
%     temp.selfinal = [];     % 'bm' | 'ps'
%     temp.solveperm = [];    % 'cp amp' | 'ang loadvec'
%     temp.bestn_cond = [];
%     temp.bestn_res = [];
%     temp.diag.DisplayWarnings = 1;
%     opts{end + 1} = temp;
% end
%
% %%% 4a) 'C-SECSI-Lt'  - Coupled-SECSI for the arbitrary L number of tensors
% if(0)
%     algorithms{end + 1} = 'C-SECSI-Lt';
%     str_legend{end + 1} = 'C-SECSI (REC PS)';
%     temp = [];
%     temp.heuristic = 'REC PS'; % BM | (REC PS) | CON PS | RES
%     temp.usehooi = false;
%     temp.overridecf2 = false;
%     temp.overridecf2 = false;
%     temp.whichsmds = [];    % 'all' | 'bestcond'
%     temp.selsmds = [];      % 'all' | 'bestres'
%     temp.selfinal = [];     % 'bm' | 'ps'
%     temp.solveperm = [];    % 'cp amp' | 'ang loadvec'
%     temp.bestn_cond = [];
%     temp.bestn_res = [];
%     temp.diag.DisplayWarnings = 1;
%     opts{end + 1} = temp;
% end
%
% %%% 4b) 'C-SECSI-Beta'  - development version of Coupled-SECSI [2] based on C-SECSI-Lt
%     % works for L tensors
%     % provides options to choose dimensionality reduction
% if(0)
%     algorithms{end + 1} = 'C-SECSI-Beta';
%     str_legend{end + 1} = 'C-SECSI (REC PS)';
%     temp = [];
%     temp.heuristic = 'REC PS'; % BM | (REC PS) | CON PS | RES
%     temp.subspace_estimation = 'C-ST-HOSVD'; % e.g., C-HOSVD | C-ST-HOSVD (see coupled_Tucker_algorithms_v2.m)
%     temp.subspace_estimation_opts = []; % this passed on to coupled_Tucker_algorithms_v2.m
%     temp.subspace_estimation_opts.dim_order = 'descending';
%     temp.subspace_estimation_opts.svd_method = 'EIG';  %  EIG | SVD | RSVD | SVDS
%     opts{end + 1} = temp;
% end
%
% %%% 5) 'C-SECSI-Minimal'  - minimal version of Coupled-SECSI [2] (REC PS only)
%     % works for L tensors
%     % should work for any coupling (one, two, three modes....)
% if(0)
%     algorithms{end + 1} = 'C-SECSI-Minimal';
%     str_legend{end + 1} = 'C-SECSI (REC PS)';
%     temp = [];
%     temp.subspace_estimation = 'C-ST-HOSVD'; C-HOSVD | C-ST-HOSVD (see coupled_Tucker_algorithms_v2.m)
%     temp.subspace_estimation_opts = [];
%     % temp.subspace_estimation_opts is passed on to coupled_Tucker_algorithms_v2.m
%     %        e.g., temp.subspace_estimation_opts.svd_method = 'EIG'; %  EIG | SVD | RSVD | SVDS
%     opts{end + 1} = temp;
% end
%
% %%% 6) 'CCP_NLS'     - non-linear least squares
% if(0)
%     algorithms{end + 1} = 'CCP_NLS';
%     str_legend{end + 1} = 'C-CPD-NLS';
%     temp = [];
%     opts{end + 1} = temp;
% end
%
% %%% 7) 'CCP_MINF'    -
% if(0)
%     algorithms{end + 1} = 'CCP_MINF';
%     str_legend{end + 1} = 'C-CPD-MINF';
%     temp = [];
%     opts{end + 1} = temp;
% end
%
% References:
% C-SECSI
%   [1] K. Naskovska and M. Haardt, "Extension of the Semi-Algebraic
%       Framework for Approximate CP Decompositions via Simultaneous Matrix
%       Diagonalization to the Efficient Calculation of Coupled CP
%       Decompositions," in Proc. of 50th Asilomar Conf. on Signals,
%       Systems, and Computers, 2016.
% 'CCP_NLS' and 'CCP_MINF'
%   [2] L. Sorber, M. Van Barel, and L. De Lathauwer, "Structured Data
%       Fusion," IEEE J. Sel. Top. Signal Process., vol. 9, no. 4,
%       pp. 586-600, 2015.
%   [3] N. Vervliet, O. Debals, L. Sorber, M. Van Barel, and L. De
%       Lathauwer, "Tensorlab 3.0," 2016.
%
% Contribution Checklist:
%   To add new algorithms, you may want to follow the following procedure,
%   so you do not forget anything and to save you troubles if it does not
%   work out...
%       0) think of what use case your data is
%       1) add a new test case in test_CPD_algorithms.m in the appropriate
%          section (e.g., dense data)  ->  algs = {'NEW'};
%       2) add new algorithm to this function (at the end or where-ever
%          else, it's up to you...)
%       3) if everything works out fine, add new algorithm to the list of
%          collected algorithms and to what use case it fits
%       4) add the (default) options and a reference
%       5) finally, think of what other use cases your algorithm might fit
%          and add those to the list and also the corresponding test cases
%       6) make sure the tests runs and commit! good luck! ;)
rootDir = return_repository_root();
if nargin < 6
    VERBOSE = 0;
end
if nargin < 5
    opts = [];
end
opts.preserve_path = setparam(opts, 'preserve_path', 1);

L = length(Xt); % number of tensors
I = cell(1, L);
for l = 1:L
    I{l} = size(Xt{l});
end

%% Run (Coupled) CP Decomposition
%% Coupled Tucker-CP
if strcmp(method, 'C-Tucker-CP')
    if opts.preserve_path
        oldPath = path;
    end

    for l = 1:L
        if isa(Xt{l}, 'double')
            % just move on...
        elseif isa(Xt{l}, 'tensor') || isa(Xt{l}, 'sptensor')
            Xt{l} = double(Xt{l});
        else
            error('datatype ''%s'' is not supported by method ''C-SECSI-Beta''', class(Xt{l}));
        end
    end
    if VERBOSE, disp(opts); end

    infos = [];
    infos.opts = opts;

    % default options for Coupled Tucker-CP
    opts.outerAlg = setparam(opts, 'outerAlg', 'C-ST-HOSVD');
    opts.innerAlg = setparam(opts, 'innerAlg', 'C-SECSI');
    R_Tucker = cell(1, L);
    N = nan(1, L);
    for l = 1:L
        N(l) = length(I{l});
        R_Tucker{l} = ones(1, N(l)) * R;
    end
    opts.compressionRank = setparam(opts, 'compressionRank', R_Tucker);

    % outer algorithm: apply Tucker compression
    if ~isfield(opts, 'outerAlgOpts')
        opts.outerAlgOpts = [];
    end
    [Gt_hat, V_hat, infos_Tucker] = coupled_Tucker_algorithms_v2(Xt, opts.compressionRank, coupling, opts.outerAlg, opts.outerAlgOpts);

    % inner algorithm: decompose core tensor
    if ~isfield(opts, 'innerAlgOpts')
        opts.innerAlgOpts = [];
    end
    [B_hat, lambda_hat, infos_CP] = coupled_CPD_algorithms_v2(Gt_hat, R, coupling, opts.innerAlg, opts.innerAlgOpts);

    % obtain factor matrices of CP-model via projection matrices of Tucker model
    A_hat = cell(1, L);
    for l = 1:L
        B_hat{l} = apply_scaling(B_hat{l}, lambda_hat{l});
        A_hat{l} = cellfun(@(A, B) A*B, V_hat{l}, B_hat{l}, 'UniformOutput', 0);
        [A_hat{l}, lambda_hat{l}] = normalize_factors(A_hat{l}); % fix scaling (do we even need this?)
    end

    % handle infos
    infos.opts_Tucker = infos_Tucker.opts;
    f = fieldnames(rmfield(infos_Tucker, 'opts'));
    for curr_field = 1:length(f)
        infos.(f{curr_field}) = infos_Tucker.(f{curr_field});
    end

    infos.opts_CP = infos_CP.opts;
    f = fieldnames(rmfield(infos_CP, 'opts'));
    for curr_field = 1:length(f)
        infos.(f{curr_field}) = infos_CP.(f{curr_field});
    end


    %% CPD_v2 (no coupling!)
elseif strcmp(method, 'CPD_v2')
    if opts.preserve_path
        opts.cp_alg_opts.preserve_path = 1;
        opts.preserve_path = 0;
    end
    if VERBOSE, disp(opts); end

    % parse coupling
    parse_coupling(I, R, coupling); % verify dimensions

    % by default, we'll use SECSI to decompose the tensors individually
    opts.cp_alg = setparam(opts, 'cp_alg', 'SECSI-Beta');
    if ~isfield(opts, 'cp_alg_opts')
        opts.cp_alg_opts = [];
    end

    % decompose tensors individually using algorithm specified in opts.cp_alg
    A_hat = cell(1, L);
    lambda_hat = cell(1, L);
    infos = struct;
    for l = 1:L
        [A_hat{l}, lambda_hat{l}, infos_l] = CPD_algorithms_v2(Xt{l}, R, opts.cp_alg, opts.cp_alg_opts, VERBOSE);
        infos.individual(l) = infos_l;
    end
    infos.opts = opts;

    infos.bins = 1:6;           % for histogram only
    infos.bin_labels = 1:6;     % for histogram only


    %% C-ALS-PV
elseif strcmp(method, 'C-ALS-PV')
    if opts.preserve_path
        oldPath = path;
    end
    addpath(genpath(fullfile(rootDir, 'Tensor_Decomposition', 'Coupled_CP')))
    addpath(genpath(fullfile(rootDir, '_toolboxes', 'Tensor_Toolbox_CRL_beta')))

    % parse coupling
    output_format = 1;
    coupling = parse_coupling(I, R, coupling, output_format);

    % check input
    for l = 1:L
        if isa(Xt{l}, 'double')
            % just move on...
        elseif isa(Xt{l}, 'tensor') || isa(Xt{l}, 'sptensor')
            Xt{l} = double(Xt{l});
        else
            error('datatype ''%s'' is not supported by method ''ALS''', class(Xt{l}));
        end
    end
    if VERBOSE, disp(opts); end

    % default options for ALS
    opts.MAXIT = setparam(opts, 'MAXIT', 1000);
    opts.delta = setparam(opts, 'delta', 1e-8);
    opts.fast  = setparam(opts, 'fast', 2);
    opts.coupled_modes = coupling;

    % decompose using plain-vanilla ALS algorithm
    [A_hat, inno, rec_err] = C_ALS_PV(Xt, R, opts);

    lambda_hat = cell(1, L);
    for l = 1:L
        [A_hat{l}, lambda_hat{l}] = normalize_factors(A_hat{l});
    end

    infos.inno = inno;
    infos.rec_err = rec_err;


    %% C-TALS-PV
elseif strcmp(method, 'C-TALS-PV')
    if opts.preserve_path
        oldPath = path;
    end
    addpath(genpath(fullfile(rootDir, 'Tensor_Decomposition', 'Coupled_CP', 'Coupled_SECSI')))
    addpath(genpath(fullfile(rootDir, '_toolboxes', 'Tensor_Toolbox_CRL')))

    % parse coupling
    output_format = 2;
    coupling = parse_coupling(I, R, coupling, output_format);

    % check input
    if gt(L, 2)
        error('coupled_CPD_algorithms:ALS_PV', 'this method currently supports only two tensors')
    end
    if gt(length(find(coupling.modes)), 1)
        error('coupled_CPD_algorithms:ALS_PV', 'works for one coupled mode only')
    end
    for l = 1:L
        if isa(Xt{l}, 'double')
            % just move on...
        elseif isa(Xt{l}, 'tensor') || isa(Xt{l}, 'sptensor')
            Xt{l} = double(Xt{l});
        else
            error('datatype ''%s'' is not supported by method ''ALS''', class(Xt{l}));
        end
    end
    if VERBOSE, disp(opts); end

    % default options for ALS
    opts.MAXIT = setparam(opts, 'MAXIT', 1000);

    % decompose using plain-vanilla ALS algorithm
    [Factors1, Factors2, inno1, inno2, recerr1, recerr2] = plainvanilla_Ctals_Rd(Xt{1}, Xt{2}, R, coupling.modes, [], [], opts.MAXIT);

    A_hat = cell(1, L);
    lambda_hat = cell(1, L);
    [A_hat{1}, lambda_hat{1}] = normalize_factors(Factors1);
    [A_hat{2}, lambda_hat{2}] = normalize_factors(Factors2);

    infos.inno1 = inno1;
    infos.inno2 = inno2;
    infos.recerr1 = recerr1;
    infos.recerr2 = recerr2;

    infos.opts = opts;
    infos.coupling = coupling;


    %% C-SECSI
elseif strcmp(method, 'C-SECSI')
    error('coupled_CPD_algorithms:C_SECSI', 'C-SECSI has been renamed to C-SECSI-5, please update your scripts!')

    %% C-SECSI-5
elseif strcmp(method, 'C-SECSI-5')
    if opts.preserve_path
        oldPath = path;
    end
    addpath(genpath(fullfile(rootDir, '_toolboxes', 'Tensor_Toolbox_CRL')))
    addpath(genpath(fullfile(rootDir, 'Tensor_Decomposition', 'Coupled_CP', 'Coupled_SECSI')))
    addpath(genpath(fullfile(rootDir, 'Simultaneous_Matrix_Diagonalization')))

    % parse coupling
    output_format = 2;
    coupling = parse_coupling(I, R, coupling, output_format);

    % check input
    if gt(L, 2)
        error('coupled_CPD_algorithms:C_SECSI_5', 'this method currently supports only two tensors')
    end
    if gt(length(find(coupling.modes)), 1)
        error('coupled_CPD_algorithms:C_SECSI_5', 'works for one coupled mode only')
    end
    for l = 1:L
        if isa(Xt{l}, 'double')
            % just move on...
        elseif isa(Xt{l}, 'tensor') || isa(Xt{l}, 'sptensor')
            Xt{l} = double(Xt{l});
        else
            error('datatype ''%s'' is not supported by method ''C-SECSI-5''', class(Xt{l}));
        end
    end
    if VERBOSE, disp(opts); end

    % default options for SECSI (TODO: check if these apply for C_SECSI_5)
    opts.heuristic = setparam(opts, 'heuristic', 'REC PS'); % BM | REC PS | CON PS | RES
    opts.usehooi = setparam(opts, 'usehooi', false); % true | false
    opts.overridecf2 = setparam(opts, 'overridecf2', false); % true | false
    opts.whichsmds = setparam(opts, 'whichsmds', []); % all | bestcond
    opts.selsmds = setparam(opts, 'selsmds', []); % all | bestres
    opts.selfinal = setparam(opts, 'selfinal', []); % bm | ps
    opts.solveperm = setparam(opts, 'solveperm', []); % 'cp amp' | 'ang loadvec'
    opts.bestn_cond = setparam(opts, 'bestn_cond', []);
    opts.bestn_res = setparam(opts, 'bestn_res', []);
    %     if ~isfield(opts, 'diag')
    %         opts.diag = [];
    %     end
    %     opts.diag.DisplayWarnings = setparam(opts.diag, 'DisplayWarnings', 1);

    % decompose using SECSI
    [Factors1, Factors2] = C_SECSI_5(Xt{1}, Xt{2}, R, coupling.modes, opts.heuristic, ...
        'usehooi', opts.usehooi, ...
        'overridecf2', opts.overridecf2, ...
        'whichsmds', opts.whichsmds, ...
        'selsmds', opts.selsmds, ...
        'selfinal', opts.selfinal, ...
        'solveperm', opts.solveperm, ...
        'bestn_cond', opts.bestn_cond, ...
        'bestn_res', opts.bestn_res);
    %'bestn_res', opts.bestn_res), ...
    %'DisplayWarnings', opts.diag.DisplayWarnings);

    A_hat = cell(1, L);
    lambda_hat = cell(1, L);
    [A_hat{1}, lambda_hat{1}] = normalize_factors(Factors1);
    [A_hat{2}, lambda_hat{2}] = normalize_factors(Factors2);

    infos.opts = opts;


    %% C-SECSI-JEVD
elseif strcmp(method, 'C-SECSI-JEVD')
    if opts.preserve_path
        oldPath = path;
    end
    addpath(genpath(fullfile(rootDir, '_toolboxes', 'Tensor_Toolbox_CRL')))
    addpath(genpath(fullfile(rootDir, 'Tensor_Decomposition', 'Coupled_CP', 'Coupled_SECSI')))
    addpath(genpath(fullfile(rootDir, 'Simultaneous_Matrix_Diagonalization')))

    % parse coupling
    output_format = 2;
    coupling = parse_coupling(I, R, coupling, output_format);

    % check input
    if gt(L, 2)
        error('coupled_CPD_algorithms:C_SECSI_JEVD', 'this method currently supports only two tensors')
    end
    if gt(length(find(coupling.modes)), 1)
        error('coupled_CPD_algorithms:C_SECSI_JEVD', 'works for one coupled mode only')
    end
    for l = 1:L
        if isa(Xt{l}, 'double')
            % just move on...
        elseif isa(Xt{l}, 'tensor') || isa(Xt{l}, 'sptensor')
            Xt{l} = double(Xt{l});
        else
            error('datatype ''%s'' is not supported by method ''C-SECSI-JEVD''', class(Xt{l}));
        end
    end
    if VERBOSE, disp(opts); end

    % default options for SECSI (TODO: check if these apply for
    % C_SECSI_JEVD)
    opts.heuristic = setparam(opts, 'heuristic', 'REC PS'); % BM | REC PS | CON PS | RES
    opts.usehooi = setparam(opts, 'usehooi', false); % true | false
    opts.overridecf2 = setparam(opts, 'overridecf2', false); % true | false
    opts.whichsmds = setparam(opts, 'whichsmds', []); % all | bestcond
    opts.selsmds = setparam(opts, 'selsmds', []); % all | bestres
    opts.selfinal = setparam(opts, 'selfinal', []); % bm | ps
    opts.solveperm = setparam(opts, 'solveperm', []); % cp amp | ang loadvec
    opts.bestn_cond = setparam(opts, 'bestn_cond', []);
    opts.bestn_res = setparam(opts, 'bestn_res', []);
    if ~isfield(opts, 'diag')
        opts.diag = [];
    end
    opts.diag.DisplayWarnings = setparam(opts.diag, 'DisplayWarnings', 1);

    % decompose using C_SECSI_JEVD
    [Factors1, Factors2] =  C_SECSI_JEVD(Xt{1}, Xt{2}, R, coupling.modes, opts.heuristic, ...
        'usehooi', opts.usehooi, ...
        'overridecf2', opts.overridecf2, ...
        'whichsmds', opts.whichsmds, ...
        'selsmds', opts.selsmds, ...
        'selfinal', opts.selfinal, ...
        'solveperm', opts.solveperm, ...
        'bestn_cond', opts.bestn_cond, ...
        'bestn_res', opts.bestn_res);
    %'bestn_res', opts.bestn_res), ...
    %'DisplayWarnings', opts.diag.DisplayWarnings);

    A_hat = cell(1, L);
    lambda_hat = cell(1, L);
    [A_hat{1}, lambda_hat{1}] = normalize_factors(Factors1);
    [A_hat{2}, lambda_hat{2}] = normalize_factors(Factors2);

    infos.opts = opts;
    infos.coupling = coupling;


    %% C-SECSI-edit
elseif strcmp(method, 'C-SECSI-edit')
    if opts.preserve_path
        oldPath = path;
    end
    addpath(genpath(fullfile(rootDir, '_toolboxes', 'Tensor_Toolbox_CRL')))
    addpath(genpath(fullfile(rootDir, 'Tensor_Decomposition', 'Coupled_CP', 'Coupled_SECSI')))
    addpath(genpath(fullfile(rootDir, 'Simultaneous_Matrix_Diagonalization')))

    % parse coupling
    output_format = 2;
    coupling = parse_coupling(I, R, coupling, output_format);

    % check input
    if gt(L, 2)
        error('coupled_CPD_algorithms:C_SECSI', 'this method currently supports only two tensors')
    end
    if gt(length(find(coupling.modes)), 1)
        error('coupled_CPD_algorithms:C_SECSI', 'works for one coupled mode only')
    end
    for l = 1:L
        if isa(Xt{l}, 'double')
            % just move on...
        elseif isa(Xt{l}, 'tensor') || isa(Xt{l}, 'sptensor')
            Xt{l} = double(Xt{l});
        else
            error('datatype ''%s'' is not supported by method ''C_SECSI_5_edit''', class(Xt{l}));
        end
    end
    if VERBOSE, disp(opts); end

    % default options for SECSI (TODO: check if these apply for C_SECSI_5)
    opts.heuristic = setparam(opts, 'heuristic', 'REC PS'); % BM | REC PS | CON PS | RES
    opts.usehooi = setparam(opts, 'usehooi', false); % true | false
    opts.overridecf2 = setparam(opts, 'overridecf2', false); % true | false
    opts.whichsmds = setparam(opts, 'whichsmds', []); % all | bestcond
    opts.selsmds = setparam(opts, 'selsmds', []); % all | bestres
    opts.selfinal = setparam(opts, 'selfinal', []); % bm | ps
    opts.solveperm = setparam(opts, 'solveperm', []); % cp amp | ang loadvec
    opts.bestn_cond = setparam(opts, 'bestn_cond', []);
    opts.bestn_res = setparam(opts, 'bestn_res', []);

    % decompose using C-SECSI-edit
    [Factors1, Factors2] = C_SECSI_5_edit(Xt{1}, Xt{2}, R, coupling.modes, opts.heuristic, ...
        'usehooi', opts.usehooi, ...
        'overridecf2', opts.overridecf2, ...
        'whichsmds', opts.whichsmds, ...
        'selsmds', opts.selsmds, ...
        'selfinal', opts.selfinal, ...
        'solveperm', opts.solveperm, ...
        'bestn_cond', opts.bestn_cond, ...
        'bestn_res', opts.bestn_res);
    %'bestn_res', opts.bestn_res), ...
    %'DisplayWarnings', opts.diag.DisplayWarnings);

    A_hat = cell(1, L);
    lambda_hat = cell(1, L);
    [A_hat{1}, lambda_hat{1}] = normalize_factors(Factors1);
    [A_hat{2}, lambda_hat{2}] = normalize_factors(Factors2);

    infos.opts = opts;
    infos.coupling = coupling;


    %% C-SECSI-Lt (Original C-SECSI Version from Alla's Master Thesis)
elseif strcmp(method, 'C-SECSI-Lt')
    if opts.preserve_path
        oldPath = path;
    end
    addpath(genpath(fullfile(rootDir, '_toolboxes', 'Tensor_Toolbox_CRL_beta')))
    addpath(genpath(fullfile(rootDir, 'Tensor_Decomposition', 'Coupled_CP', 'Coupled_SECSI')))
    addpath(genpath(fullfile(rootDir, 'Simultaneous_Matrix_Diagonalization')))

    % parse coupling
    output_format = 1;
    coupling = parse_coupling(I, R, coupling, output_format);

    if gt(length(find(coupling)), 1)
        error('coupled_CPD_algorithms:C_SECSI_Lt', 'works for one coupled mode only')
    end
    for l = 1:L
        if isa(Xt{l}, 'double')
            % just move on...
        elseif isa(Xt{l}, 'tensor') || isa(Xt{l}, 'sptensor')
            Xt{l} = double(Xt{l});
        else
            error('datatype ''%s'' is not supported by method ''C_SECSI-Lt''', class(Xt{l}));
        end
    end
    if VERBOSE, disp(opts); end

    % default options
    opts.heuristic   = setparam(opts, 'heuristic', 'REC PS'); % BM | REC PS | CON PS | RES
    opts.usehooi     = setparam(opts, 'usehooi', false); % true | false
    opts.overridecf2 = setparam(opts, 'overridecf2', false); % true | false
    opts.whichsmds   = setparam(opts, 'whichsmds', []); % all | bestcond
    opts.selsmds     = setparam(opts, 'selsmds', []); % all | bestres
    opts.selfinal    = setparam(opts, 'selfinal', []); % bm | ps
    opts.solveperm   = setparam(opts, 'solveperm', []); % cp amp | ang loadvec
    opts.bestn_cond  = setparam(opts, 'bestn_cond', []);
    opts.bestn_res   = setparam(opts, 'bestn_res', []);
    if ~isfield(opts, 'DisplayWarnings')
        opts.DisplayWarnings = setparam(opts, 'DisplayWarnings', 1);
    end

    % decompose tensors
    [Factors_est] = C_SECSI_Lt(Xt, R, coupling, opts.heuristic, ...
        'usehooi', opts.usehooi, ...
        'overridecf2', opts.overridecf2, ...
        'whichsmds', opts.whichsmds, ...
        'selsmds', opts.selsmds, ...
        'selfinal', opts.selfinal, ...
        'solveperm', opts.solveperm, ...
        'bestn_cond', opts.bestn_cond, ...
        'bestn_res', opts.bestn_res, ...
        'DisplayWarnings', opts.DisplayWarnings);

    % normalize factors
    A_hat = cell(1, L);
    lambda_hat = cell(1, L);
    for l = 1:L
        [A_hat{l}, lambda_hat{l}] = normalize_factors(Factors_est{l});
    end

    infos.opts = opts;
    infos.coupling = coupling;


    %% C-SECSI-Beta (Development Version based on C-SECSI-Lt)
elseif strcmp(method, 'C-SECSI-Beta')
    if opts.preserve_path
        oldPath = path;
    end
    addpath(genpath(fullfile(rootDir, '_toolboxes', 'Tensor_Toolbox_CRL_beta')))
    addpath(genpath(fullfile(rootDir, 'Tensor_Decomposition', 'Coupled_CP', 'Coupled_SECSI')))
    addpath(genpath(fullfile(rootDir, 'Simultaneous_Matrix_Diagonalization')))

    % parse coupling
    output_format = 1;
    coupling = parse_coupling(I, R, coupling, output_format);

    for l = 1:L
        if isa(Xt{l}, 'double')
            % just move on...
        elseif isa(Xt{l}, 'tensor') || isa(Xt{l}, 'sptensor')
            Xt{l} = double(Xt{l});
        else
            error('datatype ''%s'' is not supported by method ''C-SECSI-Beta''', class(Xt{l}));
        end
    end
    if VERBOSE, disp(opts); end

    % default options
    opts.heuristic   = setparam(opts, 'heuristic', 'REC PS'); % BM | REC PS | CON PS | RES
    opts.usehooi     = setparam(opts, 'usehooi', false); % true | false
    opts.overridecf2 = setparam(opts, 'overridecf2', false); % true | false
    opts.whichsmds   = setparam(opts, 'whichsmds', []); % all | bestcond
    opts.selsmds     = setparam(opts, 'selsmds', []); % all | bestres
    opts.selfinal    = setparam(opts, 'selfinal', []); % bm | ps
    opts.solveperm   = setparam(opts, 'solveperm', []); % cp amp | ang loadvec
    opts.bestn_cond  = setparam(opts, 'bestn_cond', []);
    opts.bestn_res   = setparam(opts, 'bestn_res', []);
    if ~isfield(opts, 'DisplayWarnings')
        opts.DisplayWarnings = setparam(opts, 'DisplayWarnings', 1);
    end

    % decompose tensors
    [F_est, infos] = C_SECSI_beta(Xt, R, coupling, opts.heuristic, opts);

    % normalize factors
    A_hat = cell(1, L);
    lambda_hat = cell(1, L);
    for l = 1:L
        [A_hat{l}, lambda_hat{l}] = normalize_factors(F_est{l});
    end

    infos.opts = opts;
    infos.coupling = coupling;

    infos.bins = 1:8;           % for histogram only
    infos.bin_labels = 1:8;     % for histogram only

    % relabel SMDs according to numbering in SECSI-Minimal
    idx_SMD = [5, 6, 3, 4, 1, 2, 7, 8];
    for l = 1:L
        infos.individual(l).idx_min = idx_SMD(infos.individual(l).idx_min);
    end


    %% C-SECSI (Minimal)
elseif strcmp(method, 'C-SECSI-Minimal')
    if opts.preserve_path
        oldPath = path;
    end
    addpath(genpath(fullfile(rootDir, '_toolboxes', 'Tensor_Toolbox_CRL_beta')))
    addpath(genpath(fullfile(rootDir, 'Tensor_Decomposition', 'Coupled_CP', 'Coupled_SECSI')))
    addpath(genpath(fullfile(rootDir, 'Simultaneous_Matrix_Diagonalization')))

    % parse coupling
    output_format = 1;
    coupling = parse_coupling(I, R, coupling, output_format);

    for l = 1:L
        if isa(Xt{l}, 'double')
            % just move on...
        elseif isa(Xt{l}, 'tensor') || isa(Xt{l}, 'sptensor')
            Xt{l} = double(Xt{l});
        else
            error('datatype ''%s'' is not supported by method ''C-SECSI-Minimal''', class(Xt{l}));
        end
    end
    if VERBOSE, disp(opts); end

    % default options
    opts.foo   = setparam(opts, 'foo', true);
    % TODO: handle warnings in C_SECSI_minimal_3way function
    %     if ~isfield(opts, 'DisplayWarnings')
    %         opts.DisplayWarnings = setparam(opts, 'DisplayWarnings', 1);
    %     end

    % decompose tensors using C-SECSI-Minimal
    [A_hat, infos] = C_SECSI_minimal_3way_v2(Xt, R, coupling, opts);

    lambda_hat = cell(1, L);
    for l = 1:L
        [A_hat{l}, lambda_hat{l}] = normalize_factors(A_hat{l});
    end

    infos.opts = opts;
    infos.coupling = coupling;

    infos.bins = 1:6;           % for histogram only
    infos.bin_labels = 1:6;     % for histogram only


    %% CCP_NLS
elseif strcmp(method, 'CCP_NLS')
    if opts.preserve_path
        oldPath = path;
    end
    addpath(genpath(fullfile(rootDir, '_toolboxes', 'tensorlab_2016-03-28')))

    % parse coupling
    output_format = 1;
    coupling = parse_coupling(I, R, coupling, output_format);

    for l = 1:L
        if isa(Xt{l}, 'double')
            % just move on...
        elseif isa(Xt{l}, 'tensor')
            Xt{l} = double(Xt{l});
        elseif isa(Xt{l}, 'sptensor')
            % TODO: convert to Tensorlab sparse format!
        else
            error('datatype ''%s'' is not supported by method ''CCP_NLS''', class(Xt{l}));
        end
    end
    if VERBOSE, disp(opts); end

    % initialize factors
    I_all = I{1};
    for l = 2:L
        I_all = [I_all, I{l}(~logical(coupling))];  %#ok<AGROW>
    end
    A_all_0 = cpd_rnd(I_all, R);

    % assign to modes
    factors_idx = cell(1, L);
    factors_idx{1} = 1:length(I{1});
    num_fact = factors_idx{1}(end);
    for l = 2:L
        factors_idx{l} = 1:length(I{l});
        numFactThisMode = sum(~logical(coupling));
        factors_idx{l}(~logical(coupling)) = num_fact + 1:num_fact + numFactThisMode;
        num_fact = num_fact + numFactThisMode;
    end
    assert(eq(length(I_all), num_fact), 'oh-oh!')

    A_all_hat = ccpd_nls(Xt, A_all_0, factors_idx);

    A_hat = cell(1, L);
    lambda_hat = cell(1, L);
    for l = 1:L
        [A_hat{l}, lambda_hat{l}] = normalize_factors(A_all_hat(factors_idx{l}));
    end

    infos.opts = opts;
    infos.coupling = coupling;


    %% CCP_MINF
elseif strcmp(method, 'CCP_MINF')
    if opts.preserve_path
        oldPath = path;
    end
    addpath(genpath(fullfile(rootDir, '_toolboxes', 'tensorlab_2016-03-28')))

    % parse coupling
    output_format = 1;
    coupling = parse_coupling(I, R, coupling, output_format);

    for l = 1:L
        if isa(Xt{l}, 'double')
            % just move on...
        elseif isa(Xt{l}, 'tensor')
            Xt{l} = double(Xt{l});
        elseif isa(Xt{l}, 'sptensor')
            % TODO: convert to Tensorlab sparse format!
        else
            error('datatype ''%s'' is not supported by method ''CCP_MINF''', class(Xt{l}));
        end
    end
    if VERBOSE, disp(opts); end


    % initialize all factors
    I_all = I{1};
    for l = 2:L
        I_all = [I_all, I{l}(~logical(coupling))];  %#ok<AGROW>
    end
    A_all_0 = cpd_rnd(I_all, R);

    % assign to modes
    factors_idx = cell(1, L);
    factors_idx{1} = 1:length(I{1});
    num_fact = factors_idx{1}(end);
    for l = 2:L
        factors_idx{l} = 1:length(I{l});
        numFactThisMode = sum(~logical(coupling));
        factors_idx{l}(~logical(coupling)) = num_fact + 1:num_fact + numFactThisMode;
        num_fact = num_fact + numFactThisMode;
    end
    assert(eq(length(I_all), num_fact), 'oh-oh!')

    A_all_hat = ccpd_minf(Xt, A_all_0, factors_idx);

    A_hat = cell(1, L);
    lambda_hat = cell(1, L);
    for l = 1:L
        [A_hat{l}, lambda_hat{l}] = normalize_factors(A_all_hat(factors_idx{l}));
    end
    infos.opts = opts;
    infos.coupling = coupling;


else
    fprintf('method ''%s'' not supported', method)
end

% ensure column vectors
% for l = 1:L
%     A_hat{l} = A_hat{l}(:);
%     lambda_hat{l} = lambda_hat{l}(:);
% end

if opts.preserve_path
    path(oldPath) % leave path as it was before we entered the function
end

%% EoF
end
