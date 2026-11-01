function alpha = mpp_absorption(theta, freq, fluid)
%MPP_ABSORPTION  Four-cell origami micro-perforated absorber, forward incidence.
%   ALPHA = MPP_ABSORPTION(THETA, FREQ, FLUID)
%   THETA is N-by-20. Columns 1:12 are hole diameters (m), cell-major:
%   d11 d12 d13 d21 ... d43. Columns 13:20 are cavity heights (m):
%   H11 H12 H21 ... H42. FREQ is a column of frequencies in Hz.
%   FLUID is the struct returned by stinson_fluid. ALPHA is N-by-numel(FREQ).

if nargin < 3 || isempty(fluid)
    fluid = stinson_fluid(freq);
end

theta = double(theta);
if isvector(theta)
    theta = theta(:).';
end
freq = double(freq(:));
nBatch = size(theta, 1);
nFreq = numel(freq);
const = acoustic_constants();

d = zeros(nBatch, 4, 3);
H = zeros(nBatch, 4, 2);
for s = 1:4
    d(:, s, :) = reshape(theta(:, (s - 1) * 3 + (1:3)), nBatch, 1, 3);
    H(:, s, :) = reshape(theta(:, 12 + (s - 1) * 2 + (1:2)), nBatch, 1, 2);
end
Heq = equivalent_height(H);

omega = 2 * pi * freq(:).';
k = fluid.k(:).';
Zc = fluid.Z(:).';
Y = zeros(2, 2, nBatch, nFreq);

for s = 1:4
    Tp1 = panel_matrix(mpp_impedance(d(:, s, 1), omega, const));
    Tp2 = panel_matrix(mpp_impedance(d(:, s, 2), omega, const));
    Tp3 = panel_matrix(mpp_impedance(d(:, s, 3), omega, const));
    Tc1 = cavity_matrix(k, Zc, Heq(:, s, 1));
    Tc2 = cavity_matrix(k, Zc, Heq(:, s, 2));
    Tm = pagemtimes(Tp1, pagemtimes(Tc1, pagemtimes(Tp2, pagemtimes(Tc2, Tp3))));

    A = reshape(Tm(1, 1, :, :), nBatch, nFreq);
    B = reshape(Tm(1, 2, :, :), nBatch, nFreq);
    C = reshape(Tm(2, 1, :, :), nBatch, nFreq);
    D = reshape(Tm(2, 2, :, :), nBatch, nFreq);

    Ys = zeros(2, 2, nBatch, nFreq);
    Ys(1, 1, :, :) = D ./ B;
    Ys(1, 2, :, :) = C - A .* D ./ B;
    Ys(2, 1, :, :) = 1 ./ B;
    Ys(2, 2, :, :) = -A ./ B;
    Y = Y + Ys / 4;
end

Y11 = reshape(Y(1, 1, :, :), nBatch, nFreq);
Y12 = reshape(Y(1, 2, :, :), nBatch, nFreq);
Y21 = reshape(Y(2, 1, :, :), nBatch, nFreq);
Y22 = reshape(Y(2, 2, :, :), nBatch, nFreq);

AA = -Y22 ./ Y21;
BB = 1 ./ Y21;
CC = Y12 - Y11 .* Y22 ./ Y21;
DD = Y11 ./ Y21;
Z0 = const.Z0;
den = AA + BB / Z0 + CC * Z0 + DD;
rf = (AA + BB / Z0 - CC * Z0 - DD) ./ den;
tau = 2 ./ den;
alpha = real(1 - abs(rf).^2 - abs(tau).^2);
end

function Heq = equivalent_height(H)
% H in metres, any shape. Origami parameters match the original script.
Hmm = H * 1e3;
h = Hmm / 4;
q = sqrt(max(0, 20^2 - h.^2));
a = 30;
hc = 20;
n0 = 4;
beta = 45;
V11 = a .* h .* q / 16;
V12 = h .* (a - hc) .* q / (16 * tand(beta));
V13 = h .* q / 24 .* (hc / tand(beta) - q / 2);
Vmm3 = 8 * n0 .* (a^2 .* h / 8 - V11 - V12 - V13);
Heq = Vmm3 * 1e-9 / (30e-3)^2;
end

function Z = mpp_impedance(d, omega, const)
d = d(:);
omega = omega(:).';
y = (d / 2) ./ sqrt(const.eta ./ (const.rho0 .* omega));
q = y .* sqrt(-1i);
F = 1 - 2 * besselj(1, q) ./ (q .* besselj(0, q));
phi = pi * d.^2 / (4 * const.S);
t = 2e-3;
Z = 1i .* omega .* const.rho0 .* t ./ (phi .* F) ...
    + sqrt(2) * const.eta .* y ./ (phi .* d) ...
    + 1i * 0.85 .* omega .* const.rho0 .* d ./ phi;
end

function T = panel_matrix(Z)
[nBatch, nFreq] = size(Z);
T = zeros(2, 2, nBatch, nFreq);
T(1, 1, :, :) = 1;
T(2, 2, :, :) = 1;
T(1, 2, :, :) = reshape(Z, 1, 1, nBatch, nFreq);
end

function T = cavity_matrix(k, Zc, height)
kh = height(:) .* k(:).';
c = cos(kh);
s = sin(kh);
[nBatch, nFreq] = size(kh);
T = zeros(2, 2, nBatch, nFreq);
T(1, 1, :, :) = reshape(c, 1, 1, nBatch, nFreq);
T(2, 2, :, :) = reshape(c, 1, 1, nBatch, nFreq);
T(1, 2, :, :) = reshape(1i .* Zc(:).' .* s, 1, 1, nBatch, nFreq);
T(2, 1, :, :) = reshape(1i .* s ./ Zc(:).', 1, 1, nBatch, nFreq);
end
