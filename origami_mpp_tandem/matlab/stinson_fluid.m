function fluid = stinson_fluid(freq)
%STINSON_FLUID  Effective wavenumber and impedance of the rectangular cavity.
const = acoustic_constants();
omega = 2 * pi * double(freq(:));
N = 200;
m = (0:N).';
n = 0:N;
am2 = ((2 * m + 1) * pi / const.W).^2;
bn2 = ((2 * n + 1) * pi / const.L).^2;
lambda2 = am2 + bn2;
coef = lambda2 ./ (am2 .* bn2);

nFreq = numel(omega);
Sv = zeros(1, nFreq);
St = zeros(1, nFreq);
step = 40;
for i0 = 1:step:(N + 1)
    i1 = min(N + 1, i0 + step - 1);
    lam = lambda2(i0:i1, :);
    c = coef(i0:i1, :);
    w = reshape(omega, 1, 1, []);
    Sv = Sv + reshape(sum(sum(c ./ (lam + 1i * w / const.nu), 1), 2), 1, []);
    St = St + reshape(sum(sum(c ./ (lam + 1i * w / const.mu), 1), 2), 1, []);
end

Fv = 64 * Sv / (const.W^2 * const.L^2);
Ft = 64 * St / (const.W^2 * const.L^2);
rho = const.rho0 ./ (1 - Fv);
K = const.gamma * const.P0 ./ (1 + (const.gamma - 1) * Ft);
fluid.k = omega(:).' .* sqrt(rho ./ K);
fluid.Z = sqrt(rho .* K);
fluid.const = const;
end
