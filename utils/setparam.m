function x = setparam(opts, name, default)
% Set options according to specified value or to default value.
%
% |----------------------------------------------------------------
% | Adopted from cp_nmu.m, Tensor Toolbox 2.6
% |-----------------------------------------------------------------
%
% opts.tol = setparam(opts, 'tol', 1e-4);
%
% sets the parameter opts.tol to 1e-4, if it has not been set previously.
%
% Inputs: opts    - struct for parameters
%         name    - parameter name
%         default - default value to use, if parameter has not been set
%
% Output: x       - parameter value
if isfield(opts, name);
    x = opts.(name);
else
    x = default;
end
end
