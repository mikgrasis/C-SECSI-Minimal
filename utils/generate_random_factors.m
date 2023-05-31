function A = generate_random_factors(I, R, factors_type, opts_fac)
% Generate set of random factors for tensor experiments.
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
% |     Date authored:  August 2018
% |     Modifications:
% |     13.06.2021 - method -> factors_type, opts -> opts_fac (MG)
% |     11.06.2021 - change output in case of N = 1 (one factor only) back cell array (MG)
% |     28.04.2021 - examples for correlation (MG)
% |     21.04.2021 - code review (MG)
% |     13.05.2020 - method 'gen_matrix' added (MG)
% |     09.06.2019 - new cases 'orthonormal' and 'unitary' (MG)
% |     05.06.2019 - new case 'complexFixedCondition' added (MG)
% |     02.05.2019 - new case 'fixedCondition' (MG)
% |     02.05.2019 - case 'otherwise' added (MG)
% |     01.05.2019 - fixed bug in methods 'sparseGaussian', 'sparseUniform', etc. if N = 1 (MG)
% |     01.05.2019 - new case 'badCondition' (MG)
% |     30.04.2019 - changed output in case of N = 1 (one factor only) to double (no cell array) (MG)
% |     30.04.2019 - new cases 'binaryOrthogonal' and 'binaryCorrelated' (MG)
% |     26.04.2019 - case 'ones' added (MG)
% |     23.04.2019 - code review (notation changed from rank-F to rank-R) (MG)
% |----------------------------------------------------------------
%
% Usage:
% I = [4, 8, 7]; R = 3;
% factors_type = 'gaussian';
% opts_fac.rho = [0.99, 0, 0]
% A = generate_random_factors(I, R, factors_type, opts_fac)
%
% generates a cell array with random matrices of size I_n x R, n = 1,...,N.
%
% Examples:
% 1) correlation method 'fantasy'
%   factors_type = 'gaussian'; % gaussian | complexGaussian
%   opts_fac.rho = [0.99, 0, 0];
%   opts_fac.sType = 'fantasy';
%   A = generate_random_factors(I, R, factors_type, opts_fac)
%
% 2) correlation method 'onesMain' (Cholesky)
%   factors_type = 'gaussian'; % gaussian | complexGaussian
%   opts_fac.sType = 'onesMain'; onesMain | toeplitz
%   opts_fac.rho = [0.99, 0, 0];
%   A = generate_random_factors(I, R, factors_type, opts_fac)
%
% 3) correlation method 'fixedCondition'
%   factors_type = 'fixedCondition'; % fixedCondition | complexFixedCondition
%   opts_fac.cond = [57, 0, 0];
%   A = generate_random_factors(I, R, factors_type, opts_fac)
%
% 4) correlation method 'gen_matrix'
%   factors_type = 'gen_matrix';
%   opts_fac.rho = [0.99, 0, 0];
%   A = generate_random_factors(I, R, factors_type, opts_fac)
%
% Dense Matrices
% 'gaussian', 'complexGaussian', 'gen_matrix', 'fixedCondition',
% 'complexFixedCondition', 'orthonormal', 'unitary', 'badCondition',
% 'uniform', 'integer', 'uniformWithOutliers', 'ones'
% Binary Matrices
%  'binary', 'binary_toss', 'binaryFullRank', 'binaryOrthogonal',
%  'binaryCorrelated',
% Sparse Matrices
%  'sparseGaussian', 'sparseUniform', 'sparseInteger',
%  'sparseGaussianFullRank', 'sparseUniformFullRank', 'sparseIntegerFullRank'
% Sketching Matrices (also sparse)
%  'countSketch', 'countSketchUnsigned', 'sketchWithReplacement'
%
% Input:
%   I - vector of dimensions I(1),...,I(N), where N is order of tensor Xt
%   R - rank of tensor, optionally: ranks of tensor R(1),...,R(N)
%   factors_type - type of matrix to generate
% Dense Matrices:
%       'gaussian'               - standard Gaussian entries
%       'complexGaussian'        - standard complex Gaussian entries
%       'gen_matrix'             - matrices with specified pairwise correlation between columns [6]
%       'fixedCondition'         - fixed condition number
%                                  default: opts_fac.cond = 100
%       'complexFixedCondition'  - complex matrix with fixed condition number
%                                  default: opts_fac.cond = 100
%       'orthonormal'            - real-valued matrix with orthonormal columns
%       'unitary'                - complex-valued matrix with unitary columns
%       'badCondition'           - matrices similar to the ones in [2], based on 'binaryOrthogonal'
%                                  default: opts_fac.val = 0.8
%       'uniform'                - uniform distribution
%       'integer'                - integer values
%                                  default: opts_fac.range = [1, 10]
%       'uniformWithOutliers'    - uniformy distributed entries with outliers [1]
%       'ones'                   - matrix of ones
% Binary Matrices:
%       'binary'                 - binary matrix with approximately density*m*n nonzero entries
%                                  default: opts_fac.density = 0.2
%       'binary_toss             - binary matrix, keep only < opts_fac.prob
%                                  default: opts_fac.prob = 0.2
%       'binaryFullRank'         - full rank binary matrix with approximately density*m*n nonzero entries
%                                  default: opts_fac.density = 0.2
%       'binaryOrthogonal'       - matrix with binary orthogonal columns
%                                  (works only for I(n)==powers of two!!)
%       'binaryCorrelated'        - binary matrix with strongly correlated columns similar to the one used in [2]
% Sparse Matrices:
%       'sparseGaussian'         - applies sparsity pattern 'binary' to 'gaussian'
%       'sparseUniform'          - applies sparsity pattern 'binary' to 'uniform'
%       'sparseInteger'          - applies sparsity pattern 'binary' to 'integer'
%       'sparseGaussianFullRank' - applies sparsity pattern 'binaryFullRank' to 'gaussian'
%       'sparseUniformFullRank'  - applies sparsity pattern 'binaryFullRank' to 'uniform'
%       'sparseIntegerFullRank'  - applies sparsity pattern 'binaryFullRank' to 'integer'
% Sketching Matrices (also sparse):
%       'countSketch'            - sketching matrix for CountSketch algorithm as described in [3]
%                                  Note: output is of size R(n) x I(n)
%       'countSketchUnsigned'    - sketching matrix for CountSketch algorithm (without the sign)
%                                  Note: output is of size R(n) x I(n)
%       'sketchWithReplacement'  - (binary) sketching matrix from [4]
%                                  Note: output is of size R(n) x R(n)
%   opts_fac - parameters to be passed on
%       opts_fac.range   - range of integer values for randi, e.g., [iMin, iMax]
%       opts_fac.density - density of binary/sparse matrices, e.g., 0.2
%       opts_fac.val     - sets value inserted in case 'badCondition'
%       opts_fac.cond    - condition numbers for case 'fixedCondition' (1 x N) vector
%                      	->can be specifed as one-for-all, e.g.,
%                           opts_fac.cond = 10
%                      	->or specify for each mode individually, e.g.,
%                           opts_fac.cond = [100, 100, 17]
%                       ->set to zero if no condition number is specified, e.g.,
%                           opts_fac.cond = [100, 0, 0]
%       opts_fac.theta   - sets parameter theta for method 'sketchWithReplacement'
%       opts_fac.rho     - sets correlation for factors in methods where it
%                       makes sense... (see end of function...)
%
% Output:
%   A - set of factor matrices, cell(N, 1), typically of size I(n) x R(n)
%
% References:
%   [1] E. C. Chi and T. G. Kolda, "On Tensors, Sparsity, and Nonnegative
%       Factorizations, " SIAM J. Matrix Anal. Appl., vol. 33, no. 4, pp.
%       1272-1299, 2012.
%   [2] F. Roemer and M. Haardt, "A semi-algebraic framework for
%       approximate CP decompositions via simultaneous matrix
%       diagonalizations (SECSI), " Signal Processing, vol. 93, no. 9, pp.
%       2722-2738, 2013.
%   [3] O. A. Malik and S. Becker, "Low-Rank Tucker Decomposition of Large
%       Tensors Using TensorSketch, " Adv. Neural Inf. Process. Syst. 31,
%       no. NeurIPS, pp. 10117-10127, 2018.
%   [4] A. Kolbeinsson, J. Kossaifi, Y. Panagakis, A. Anandkumar, I.
%       Tzoulaki, and P. Matthews, "Stochastically Rank-Regularized Tensor
%       Regression Networks, " Feb. 2019.
%   [5] Y. Shi and A. Anandkumar, "Multi-dimensional Tensor Sketch, " Jan. 2019.
%   [6] A.-H. Phan, P. Tichavsky, and A. Cichocki, "CANDECOMP/PARAFAC
%       Decomposition of High-Order Tensors Through Tensor Reshaping,"
%       IEEE Trans. Signal Process., vol. 61, no. 19, pp. 4847-4860, Oct. 2013.
if nargin < 4
    opts_fac = struct;
end
opts_fac.range = setparam(opts_fac, 'range', [1, 10]);  % range of integers for method 'integer'
opts_fac.density = setparam(opts_fac, 'density', 0.2);  % approximately density*m*n nonzero entries for methods 'binary' and 'binaryFullRank'
opts_fac.val = setparam(opts_fac, 'val', 0.8);          % value to fill in for entries different than 1 in method 'badCondition'
opts_fac.theta = setparam(opts_fac, 'theta', 0.4);      % sets parameter theta for method 'sketchWithReplacement'
opts_fac.cond = setparam(opts_fac, 'cond', 10);         % condition number for method 'fixedCondition'
opts_fac.prob = setparam(opts_fac, 'prob', 0.2);        % probability for success in method 'binary_toss'
opts_fac.sType = setparam(opts_fac, 'sType', 'fantasy');% method for how to apply correlation if opts_fac.rho is set
if nargin < 3
    factors_type = 'gaussian';
end
% if we input a vector for second argument it must be same size as I
if length(R) > 1
    assert(eq(length(I), length(R)), 'input vectors I and R must be same length');
else
    R = ones(length(I), 1) * R;
end

% tensor order
N = length(I);

% initialize
A = cell(1, N);


% Dense Matrices:
% 'gaussian', 'complexGaussian', 'gen_matrix', 'fixedCondition',
% 'complexFixedCondition', 'orthonormal', 'unitary', 'badCondition',
% 'uniform', 'integer', 'uniformWithOutliers', 'ones'
% Binary Matrices:
%  'binary', 'binary_toss', 'binaryFullRank', 'binaryOrthogonal',
%  'binaryCorrelated',
% Sparse Matrices:
%  'sparseGaussian', 'sparseUniform', 'sparseInteger',
%  'sparseGaussianFullRank', 'sparseUniformFullRank', 'sparseIntegerFullRank'
% Sketching Matrices (also sparse):
%  'countSketch', 'countSketchUnsigned', 'sketchWithReplacement'
switch factors_type
    case 'gaussian'
        for n = 1:N
            A{n} = randn(I(n), R(n));
        end
    case 'complexGaussian'
        for n = 1:N
            A{n} = (randn(I(n), R(n)) + 1i * randn(I(n), R(n))) / sqrt(2);
        end
    case 'gen_matrix' % requires to load TensorBox_2018!
                      % add this to your script:
                      % addpath(genpath(fullfile(rootDir, '_toolboxes', 'TensorBox_2018')))
        if isfield(opts_fac, 'rho')
            for n = 1:N
                A{n} = gen_matrix(I(n), R(n), opts_fac.rho(n));
            end
        else
            for n = 1:N
                A{n} = randn(I(n), R(n));
            end
            %error('randomFactors:genMatrix', 'method ''gen_matrix'' requires field opts_fac.rho to be specified!')
            warning('gen_matrix: field opts_fac.rho not specified! returning standard Gaussian factors!')
        end
    case 'fixedCondition'
        % handle parameter
        if eq(size(opts_fac.cond), [1, 1])
            opts_fac.cond = ones(1, N) * opts_fac.cond;
        end
        % https://math.stackexchange.com/questions/198515/can-we-generate-random-singular-matrices-with-desired-condition-number-using-mat
        for n = 1:N
            if opts_fac.cond(n)
                A{n} = randn(I(n), R(n));
                [U, S, V] = svd(A{n});
                S(S ~= 0) = linspace(opts_fac.cond(n), 1, R(n));
                A{n} = U * S * V';
            else
                % if no condition number is specified...
                A{n} = randn(I(n), R(n));
            end
        end
    case 'complexFixedCondition'
        % handle parameter
        if eq(size(opts_fac.cond), [1, 1])
            opts_fac.cond = ones(1, N) * opts_fac.cond;
        end
        % https://math.stackexchange.com/questions/198515/can-we-generate-random-singular-matrices-with-desired-condition-number-using-mat
        for n = 1:N
            if opts_fac.cond(n)
                A{n} = (randn(I(n), R(n)) + 1i * randn(I(n), R(n))) / sqrt(2);
                [U, S, V] = svd(A{n});
                S(S ~= 0) = linspace(opts_fac.cond(n), 1, R(n));
                A{n} = U * S * V';
            else
                % if no condition number is specified...
                A{n} = (randn(I(n), R(n)) + 1i * randn(I(n), R(n))) / sqrt(2);
            end
        end
    case 'orthonormal'
        for n = 1:N
            [A{n}, ~] = qr(randn(I(n), R(n)), 0);
        end
    case 'unitary'
        for n = 1:N
            [A{n}, ~] = qr(randn(I(n), R(n))+1i*randn(I(n), R(n)), 0);
        end
    case 'uniform'
        for n = 1:N
            A{n} = rand(I(n), R(n));
        end
    case 'badCondition'
        % Generate matrices similar to the ones in Roemer et al. (2013)
        A = generate_random_factors(I, R, 'binaryCorrelated');
        for n = 1:N
            A{n}(A{n} == 0) = opts_fac.val;
        end
    case 'integer' %full-rank integer....
        for n = 1:N
            t = 0;
            while t < 10 %this doesn't need many trials..
                A{n} = randi(opts_fac.range, [I(n), R(n)]);
                if eq(rank(A{n}), R(n))
                    break
                end
                t = t + 1;
            end
        end
    case 'uniformWithOutliers'
        % generate factor matrices with a few large entries in each column
        % from: T4_cpapr_doc.m [1]
        for n = 1:N
            A{n} = rand(I(n), R(n));
            for f = 1:R(n)
                p = randperm(I(n));
                nbig = round((1 / R(n))*I(n));
                A{n}(p(1:nbig), f) = 100 * A{n}(p(1:nbig), f);
            end
        end
    case 'ones'
        for n = 1:N
            A{n} = ones(I(n), R(n));
        end
    case 'binary'
        for n = 1:N
            A{n} = full(double(sprand(I(n), R(n), opts_fac.density) > 0));
        end
    case 'binary_toss'
        for n = 1:N
            A{n} = rand(I(n), R(n));
            A{n} = A{n} < opts_fac.prob;
        end
    case 'binaryFullRank' % full-rank binary...
        for n = 1:N
            t = 0;
            numTrials = 100000;
            while t < numTrials
                % sparse uniformly distributed random matrix with approximately density*m*n nonzero entries
                % WARNING: this does not work well for small matrices!!!
                A{n} = full(double(sprand(I(n), R(n), opts_fac.density) > 0));
                if eq(rank(A{n}), R(n))
                    break
                end
                t = t + 1;
            end
            if t == numTrials
                warning('hi! I slipped in a rank deficient one... just so you know!!!')
            end
        end
    case 'binaryOrthogonal'
        % https://www.mathworks.com/matlabcentral/answers/379929-how-to-generate-a-set-of-n-mutually-orthogonal-n-being-a-power-of-2-n-dimensional-binary-vectors
        for n = 1:N
            if ne(mod(I(n), 2), 0)
                error('method ''binaryOrthogonal'' only works for I(n) powers of two')
            end
            A{n} = -((dec2bin(0:(2^(N - 1) - 1), N) - '0') * 2 - 1);
        end
    case 'binaryCorrelated'
        % inspired by
        % https://www.mathworks.com/matlabcentral/answers/379929-how-to-generate-a-set-of-n-mutually-orthogonal-n-being-a-power-of-2-n-dimensional-binary-vectors
        for n = 1:N
            temp = (dec2bin((2^I(n) - 1):-1:2^(I(n) - 1), I(n)) - '0')';
            A{n} = temp(:, 1:R(n));
        end
    case 'sparseGaussian'
        A = generate_random_factors(I, R, 'gaussian', opts_fac);
        W = generate_random_factors(I, R, 'binary', opts_fac);
        for n = 1:N
            % apply sparsity...
            A{n} = A{n} .* W{n};
        end
    case 'sparseUniform'
        A = generate_random_factors(I, R, 'uniform', opts_fac);
        W = generate_random_factors(I, R, 'binary', opts_fac);
        for n = 1:N
            % apply sparsity...
            A{n} = A{n} .* W{n};
        end
    case 'sparseInteger'
        A = generate_random_factors(I, R, 'integer', opts_fac);
        W = generate_random_factors(I, R, 'binary', opts_fac);
        for n = 1:N
            % apply sparsity...
            A{n} = A{n} .* W{n};
        end
    case 'sparseGaussianFullRank'
        A = generate_random_factors(I, R, 'gaussian', opts_fac);
        W = generate_random_factors(I, R, 'binaryFullRank', opts_fac);
        for n = 1:N
            % apply sparsity...
            A{n} = A{n} .* W{n};
        end
    case 'sparseUniformFullRank'
        A = generate_random_factors(I, R, 'uniform', opts_fac);
        W = generate_random_factors(I, R, 'binaryFullRank', opts_fac);
        for n = 1:N
            % apply sparsity...
            A{n} = A{n} .* W{n};
        end
    case 'sparseIntegerFullRank'
        A = generate_random_factors(I, R, 'integer', opts_fac);
        W = generate_random_factors(I, R, 'binaryFullRank', opts_fac);
        for n = 1:N
            % apply sparsity...
            A{n} = A{n} .* W{n};
        end
    case 'countSketch'
        % Sketching matrices for CountSketch algorithm as described in [3].
        %
        % Note: We redefine the inputs R(n) to be the sketching dimensions J(n).
        % The output for this method therefore results in an
        %                       R(n) x I(n)
        % rather than the standard (I(n) x R(n)) matrix!
        %
        % Usage in [3]: S * A{n}
        for n = 1:N
            % draw sign functions s(n) : I(n) -> {+, -}, n = 1, ..., N
            s = sign(randn(I(n), 1));

            % draw hash functions h(n) : I(n) ->  J(n), n = 1, ..., N
            h = randi([1, R(n)], I(n), 1);

            % in matrix P, choose one element in every row to be non-zero
            P = zeros(R(n), I(n));
            for i = 1:I(n)
                P(h(i), i) = 1;
            end

            % output S = P * D of size R(n) x I(n)
            A{n} = P * diag(s);
        end
    case 'countSketchUnsigned'
        % Same as 'countSketch' but without the sign.
        %
        % Note: We will the inputs R(n) to be the sketching dimensions J(n).
        % Note that the output for this method therefore results in an
        %                       R(n) x I(n)
        % rather than the standard (I(n) x R(n))!°
        %
        % Note: this matrix corresponds to H{n}.' in [5]
        %
        % Usage in [5]: (Xt .* St) \nmode{ H{n} }
        for n = 1:N
            % draw hash functions h(n) : I(n) -> J(n), n = 1, ..., N
            h = randi([1, R(n)], I(n), 1);

            % choose one element in every row to be non-zero
            A{n} = zeros(R(n), I(n));
            for i = 1:I(n)
                A{n}(h(i), i) = 1;
            end
        end
    case 'sketchWithReplacement'
        % uniform sampling matrix M{n} selecting K_n elements (with replacement)
        %   -first R(n) - K(n) rows are zero
        %   -then set one entry to 1 in each row
        %
        % usage in [4]: Utilde{n} = U{n} * M{n}.'
        for n = 1:N
            % compute number of elements to select
            K_n = floor(R(n)*opts_fac.theta);

            % in matrix P, choose one element in every row to be non-zero
            A{n} = zeros(R(n), R(n));
            for r = 1:K_n
                A{n}(r, randi([1, R(n)])) = 1;
            end
        end
    otherwise
        error('generate_random_factors: method not known')
end

% correlate columns if requested
if any(strcmp(factors_type, {'gaussian', 'complexGaussian'})) && isfield(opts_fac, 'rho')
    assert(eq(length(opts_fac.rho), N), 'random_factors:correlation', ...
        'correlation vector requires same length as number of dimensions');
    opts_corr.sType = opts_fac.sType;
    for n = 1:N
        if opts_fac.rho(n)
            opts_corr.rho = opts_fac.rho(n);
            A{n} = correlate_columns(A{n}, opts_corr);
        end
    end
end
end
