function [Xt, A, lambda] = generate_random_tensor(I, R, method, opts)
% Generates a random tensor with specified i.i.d. entries.
%
% |----------------------------------------------------------------
% | (C) 2020 Mikus Grasis
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
% |     Date authored: August 2018
% |     Last modifications:
% |     23.06.2020 - Tensor Tools header added (MG)
% |     19.09.2019 - adopted coding rules formatting (MG)
% |     04.06.2019 - case 'complexGaussianIID' added (MG)
% |     30.04.2019 - header added (MG)
% |----------------------------------------------------------------
%
% Usage:
%   Xt = generate_random_tensor(I, [], 'gaussianIID')
%
% generates a random tensor with entries drawn from a zero-mean standard
% Gaussian distribution.
%
%   [Xt, A, lambda] = generate_random_tensor(I, R, 'gaussianFactors')
%
% generates a random CP-tensor with factors drawn from a zero-mean standard
% Gaussian distribution.
%
% Inputs:
%   I       - vector of dimensions I(1),...,I(N)
%   R       - rank of tensor (only required for methods that generate tensor from factors)
%   method  - generate entries of random tensor:
%           'gaussianIID', 'complexGaussianIID', 'uniformIID', 'integerIID'
%               'sparseGaussianIID', 'sparseUniformIID', 'sparseIntegerIID'
%             or from factor matrices:
%            'gaussianFactors', 'uniformFactors', 'integerFactors', 'binaryFactors'
%               'sparseGaussianFactors', 'sparseUniformFactors', 'sparseIntegerFactors'
%
% Outputs:
%   Xt      - random tensor (double)
%   A       - set of factors (if R is provided)
%   lambda  - scalings (if R is provided)
%
% Notation:
%   order N, rank R, dimensions I(1),...,I(N)

%% Use parameters from input or by using defaults
if nargin < 4
    opts = struct;
end
opts.pow = setparam(opts, 'pow', 1);
opts.range = setparam(opts, 'range', [1, 10]);
opts.density = setparam(opts, 'density', 0.1);

%% Generate random tensor
switch method
    case 'gaussianIID'
        if ~isempty(R)
            fprintf('input argument R = %d ignored\n', R)
        end
        Xt = randn(I) .* sqrt(opts.pow);
    case 'complexGaussianIID'
        if ~isempty(R)
            fprintf('input argument R = %d ignored\n', R)
        end
        Xt = (randn(I) + 1i * randn(I)) / sqrt(2) * sqrt(opts.pow);
    case 'uniformIID'
        if ~isempty(R)
            fprintf('input argument R = %d ignored\n', R)
        end
        Xt = rand(I);
    case 'integerIID'
        if ~isempty(R)
            fprintf('input argument R = %d ignored\n', R)
        end
        Xt = randi(opts.range, I);
    case 'binaryIID'
        if ~isempty(R)
            fprintf('input argument R = %d ignored\n', R)
        end
        Xt = double(rand(I) < opts.density);
    case 'sparseGaussianIID'
        if ~isempty(R)
            fprintf('input argument R = %d ignored\n', R)
        end
        W = generate_random_tensor(I, [], 'binaryIID', opts);
        Xt = generate_random_tensor(I, [], 'gaussianIID', opts) .* W;
    case 'sparseUniformIID'
        if ~isempty(R)
            fprintf('input argument R = %d ignored\n', R)
        end
        W = generate_random_tensor(I, [], 'binaryIID', opts);
        Xt = generate_random_tensor(I, [], 'uniformIID', opts) .* W;
    case 'sparseIntegerIID'
        if ~isempty(R)
            fprintf('input argument R = %d ignored\n', R)
        end
        W = generate_random_tensor(I, [], 'binaryIID', opts);
        Xt = generate_random_tensor(I, [], 'integerIID', opts) .* W;
    case 'gaussianFactors'
        A = generate_random_factors(I, R, 'gaussian', opts);
        Xt = cp_construct(A);
        [A, lambda] = normalize_factors(A);
    case 'uniformFactors'
        A = generate_random_factors(I, R, 'uniform', opts);
        Xt = cp_construct(A);
        [A, lambda] = normalize_factors(A);
    case 'integerFactors'
        A = generate_random_factors(I, R, 'integer', opts);
        Xt = cp_construct(A);
        [A, lambda] = normalize_factors(A);
    case 'binaryFactors'
        A = generate_random_factors(I, R, 'binary', opts);
        Xt = cp_construct(A);
        [A, lambda] = normalize_factors(A);
    case 'sparseGaussianFactors'
        A = generate_random_factors(I, R, 'sparseGaussian', opts);
        Xt = cp_construct(A);
        [A, lambda] = normalize_factors(A);
    case 'sparseUniformFactors'
        A = generate_random_factors(I, R, 'sparseUniform', opts);
        Xt = cp_construct(A);
        [A, lambda] = normalize_factors(A);
    case 'sparseIntegerFactors'
        A = generate_random_factors(I, R, 'sparseInteger', opts);
        lambda = randi(opts.range, [R, 1]);
        Xt = cp_construct(A, lambda);
    otherwise
        error('generate_random_tensor: method %s not supported', method)
end
end
