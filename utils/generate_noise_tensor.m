function [Nt, sigma] = generate_noise_tensor(PS, I, sn, noise_type, method)
% Generates a noise tensor with a certain SNR with respect to a given data
% tensor.
%
% |----------------------------------------------------------------
% | (C) 2019 Mikus Grasis
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
% |     18.10.2021 - code review (MG)
% |     16.06.2019 - signature changed from (Xt, SNR, nType, method) to (PS, I, SNR, nType, method) (MG)
% |     04.06.2019 - case 'complexGaussianIID' added (MG)
% |     24.04.2019 - header added (MG)
% |----------------------------------------------------------------
%
% Example:
%   I = [3, 4, 5];
%   Xt = randn(I);
%   PS = tensor_power(Xt);
%   sn = 6;
%   noise_type = 'gaussianIID';
%   Nt = generate_noise_tensor(PS, I, sn, noise_type, 'sigma');
%
%   generates a noise tensor with entries drawn from a zero mean Gaussian
%   distribution scaled to obtain the SNR given in sn.
%
% Inputs:
%   PS     - signal power ( = variance of tensor entries, compute, e.g., via PS = tensor_power(Xt))
%   I      - size of tensor I(1), ..., I(N)
%   SNR    - target SNR for noise tensor
%   noise_type  - how to generate entries of random tensor, options include:
%               'complexGaussianIID', 'gaussianIID', 'uniformIID', 'integerIID', 'sparseGaussianIID', 'sparseUniformIID', 'sparseIntegerIID'
%                   ->see function generate_random_tensor.m for a full list of options!
%   method - defines scaling of noise tensor
%            'exact' - Noise tensor is drawn randomly and the scaled to obtain the *exact* SNR.
%            'sigma' - Noise tensor is drawn randomly with variance according to the SNR.
%                      In this case, the empirical SNR is a random variable, which might be more more realistic.
% Outputs:
%   Nt     - noise tensor (double)
%   sigma  - standard deviation of noise
if nargin < 5
    method = 'sigma';
end
switch method
    case 'sigma'
        % calculate noise standard deviation
        sigma = sqrt(PS*10^(-sn/10));

        % draw noise tensor with according variance
        opts.pow = sigma^2;
        Nt = generate_random_tensor(I, [], noise_type, opts);

    case 'exact'
        % draw noise tensor
        Nt = generate_random_tensor(I, [], noise_type);

        % check noise power
        powN = tensor_power(Nt);

        % adjust power of noise tensor accordingly
        powNtarget = PS / (10^(sn/10)+eps);
        Nt = Nt * sqrt(powNtarget/powN);

        sigma = sqrt(powNtarget/powN);
        % In TDALAB/setNoise.m we find the following:
        % r = sqrt(powX/powN) * (10^((-SNR/20)));
        % Nt = Nt * r;
        %assertElementsAlmostEqual(sqrt(powNtarget/powN), r) %OK

    otherwise
        error('generate_noise_tensor: method %s not supported', method)
end
end
