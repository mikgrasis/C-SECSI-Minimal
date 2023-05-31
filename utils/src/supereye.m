function It = supereye(N, R)

% SUPEREYE   Computes the identity tensor of order N and size R.
%
% |----------------------------------------------------------------
% | (C) 2007 TU Ilmenau, Communications Research Laboratory
% |
% |     Florian Roemer
% |
% |     Advisors:
% |        Dipl.-Ing. Giovanni Del Galdo
% |        Univ.-Prof. Dr.-Ing. Martin Haardt
% |
% |     Last modifications: 01/10/2020
% |----------------------------------------------------------------
%
% Syntax:
%   It = SUEPREYE(N, R)
%
% Input:
%   N - order of the identity tensor (N-dimensional).
%   R - size of the identity tensor (SIZE(I)=[R,R,...,R]).
%
% Output:
%   It - identity tensor of size R * ones(1, N) having ones on its
%   hyperdiagonal and zeros elsewhere.
%
% Notation:
%   Order N, Rank R, Dimensions I   
It = zeros(R*ones(1, N));

%%% Slow but safe
% for nd = 1:d
%     ed = [zeros(1,nd-1), 1, zeros(1,d-nd)].';
%     rankoneterm = ed;
%     for r = 1+1:R
%         rankoneterm = outer_product({rankoneterm,ed});
%     end
%     I = I + rankoneterm;
% end

%%% Fast but weird
for nd = 1:R
    It = subsasgn(It, struct('type', '()', 'subs', {mat2cell(nd*ones(1, N), 1, ones(1, N))}), 1);
end