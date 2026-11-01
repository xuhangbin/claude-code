clear;
clc;
close all;

% 本文件自带全部子函数。无论当前文件夹在哪里，只要运行本脚本即可。
% 逆网络输入是声学状态 [中心频率Hz, 带宽目标Hz, 最小吸声系数, 带宽类型]
% 带宽类型 -1：带宽低于目标；+1：带宽高于目标。
% 状态1：400 Hz，带宽目标 40 Hz（小于 50 Hz），最小吸声系数 0.5
% 状态2：400 Hz，带宽目标 230 Hz（大于 200 Hz），最小吸声系数 0.5
% 需要 MATLAB R2020b 及以上（pagemtimes）。

freq = (100:2:700).';
fluid = stinson_fluid(freq);

state = [
    400, 40,  0.5, -1
    400, 230, 0.5,  1
];
stateName = {'状态1 窄带', '状态2 宽带'};

model = train_state_tandem(freq, fluid, 1200);
thisDir = fileparts(mfilename('fullpath'));
save(fullfile(thisDir, 'state_tandem.mat'), 'model', 'state');

alpha = zeros(2, numel(freq));
found = zeros(2, 20);
for i = 1:2
    fprintf('%s\n', stateName{i});
    guess = propose_state(model, state(i, :));
    found(i, :) = refine_state(state(i, :), guess, freq, fluid);
    alpha(i, :) = mpp_absorption(found(i, :), freq, fluid);
    [fPeak, bandwidth, aPeak] = measure_state(freq, alpha(i, :), state(i, 3));
    fprintf('  峰值 %.1f Hz，带宽 %.1f Hz，峰值吸声 %.3f\n', fPeak, bandwidth, aPeak);
end

figure('Color', 'w', 'Position', [120 80 860 640]);
for i = 1:2
    subplot(2, 1, i);
    plot(freq, alpha(i, :), 'b-', 'LineWidth', 1.8);
    hold on;
    yline(state(i, 3), 'k--', 'LineWidth', 1.0);
    xline(state(i, 1), 'r--', 'LineWidth', 1.0);
    [fPeak, bandwidth, aPeak] = measure_state(freq, alpha(i, :), state(i, 3));
    title(sprintf('%s   峰值 %.0f Hz   带宽 %.0f Hz   \\alpha_{peak}=%.2f', ...
        stateName{i}, fPeak, bandwidth, aPeak), 'FontWeight', 'normal');
    xlabel('Frequency/Hz');
    ylabel('Absorption');
    xlim([freq(1) freq(end)]);
    ylim([0 1]);
    grid on;
    box on;
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14, 'LineWidth', 1.1, 'TickDir', 'in');
end
saveas(gcf, fullfile(thisDir, 'narrow_to_wide.png'));
save(fullfile(thisDir, 'narrow_to_wide_design.mat'), 'found', 'alpha', 'freq', 'state');


%% ===== acoustic_constants.m =====
function const = acoustic_constants()
const.rho0 = 1.29;
const.c0 = 343;
const.eta = 1.89e-5;
const.kappa = 0.026;
const.Cv = 712;
const.gamma = 1.4;
const.P0 = const.rho0 * const.c0^2 / const.gamma;
const.Z0 = const.rho0 * const.c0;
const.nu = const.eta / const.rho0;
const.mu = const.kappa / (const.rho0 * const.Cv);
const.W = 30e-3;
const.L = 30e-3;
const.S = const.W * const.L;
const.dMin = 1e-3;
const.dMax = 8e-3;
const.hMin = 5e-3;
const.hMax = 80e-3;
end

%% ===== stinson_fluid.m =====
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

%% ===== parameter_bounds.m =====
function [low, high] = parameter_bounds()
const = acoustic_constants();
low = [repmat(const.dMin, 1, 12), repmat(const.hMin, 1, 8)];
high = [repmat(const.dMax, 1, 12), repmat(const.hMax, 1, 8)];
end

%% ===== normalize_theta.m =====
function unit = normalize_theta(theta)
[low, high] = parameter_bounds();
unit = (theta - low) ./ (high - low);
end

%% ===== denormalize_theta.m =====
function theta = denormalize_theta(unit)
[low, high] = parameter_bounds();
theta = low + unit .* (high - low);
end

%% ===== normalize_state.m =====
function x = normalize_state(state)
% [f_Hz, bandwidth_Hz, alpha_min, band_type] -> inverse-network input.
x = [(state(:, 1) - 100) / 600, state(:, 2) / 600, state(:, 3), state(:, 4)];
end

%% ===== measure_state.m =====
function [fPeak, bandwidth, aPeak] = measure_state(freq, alpha, alphaMin)
%MEASURE_STATE  Peak frequency and alpha>=alphaMin bandwidth.
%   Bandwidth is the contiguous width around the maximum where absorption
%   stays at or above ALPHAMIN. A peak below ALPHAMIN has zero bandwidth.
%   FREQ is a frequency column. ALPHA is N-by-F or a single spectrum.

freq = double(freq(:));
alpha = double(alpha);
if isvector(alpha)
    alpha = alpha(:).';
end
df = freq(2) - freq(1);
nBatch = size(alpha, 1);
fPeak = zeros(nBatch, 1);
bandwidth = zeros(nBatch, 1);
aPeak = zeros(nBatch, 1);

for i = 1:nBatch
    row = alpha(i, :);
    [aPeak(i), idx] = max(row);
    fPeak(i) = freq(idx);
    if aPeak(i) < alphaMin
        bandwidth(i) = 0;
        continue
    end
    left = idx;
    while left > 1 && row(left) >= alphaMin
        left = left - 1;
    end
    if row(left) < alphaMin
        left = left + 1;
    end
    right = idx;
    while right < numel(row) && row(right) >= alphaMin
        right = right + 1;
    end
    if row(right) < alphaMin
        right = right - 1;
    end
    bandwidth(i) = freq(right) - freq(left) + df;
end
end

%% ===== state_loss.m =====
function loss = state_loss(freq, alpha, state)
%STATE_LOSS  How far a spectrum is from a requested acoustic state.
%   STATE is 1-by-4 or N-by-4:
%     [f_center_Hz, bandwidth_Hz, alpha_min, band_type]
%   band_type = -1 asks for bandwidth < bandwidth_Hz (narrow).
%   band_type = +1 asks for bandwidth > bandwidth_Hz (wide).
%   The peak absorption must also reach alpha_min, and the peak frequency
%   must sit on f_center_Hz. LOSS is one scalar per spectrum.

[fPeak, bandwidth, aPeak] = measure_state(freq, alpha, state_alpha_min(state));
targetF = state(:, 1);
limitBw = state(:, 2);
alphaMin = state(:, 3);
bandType = state(:, 4);
if size(state, 1) == 1 && numel(fPeak) > 1
    targetF = repmat(targetF, numel(fPeak), 1);
    limitBw = repmat(limitBw, numel(fPeak), 1);
    alphaMin = repmat(alphaMin, numel(fPeak), 1);
    bandType = repmat(bandType, numel(fPeak), 1);
end

freqLoss = ((fPeak - targetF) / 15).^2;
alphaLoss = (max(0, alphaMin - aPeak) / 0.05).^2;
narrow = max(0, bandwidth - limitBw) / 25;
wide = max(0, limitBw - bandwidth) / 25;
bwLoss = (bandType < 0) .* narrow.^2 + (bandType > 0) .* wide.^2;
loss = freqLoss + bwLoss + alphaLoss;
end

function alphaMin = state_alpha_min(state)
alphaMin = state(1, 3);
end

%% ===== soft_state_loss.m =====
function [loss, gradAlpha] = soft_state_loss(freq, alpha, state)
%SOFT_STATE_LOSS  Differentiable state loss and d(loss)/d(alpha).
%   Same state encoding as STATE_LOSS. GRADALPHA is N-by-F when N spectra
%   are compared with N states, or with one shared state.

freq = double(freq(:));
alpha = double(alpha);
nBatch = size(alpha, 1);
nFreq = numel(freq);
if size(state, 1) == 1 && nBatch > 1
    state = repmat(state, nBatch, 1);
end

tau = 0.05;
temperature = 0.02;
df = freq(2) - freq(1);
sig = 1 ./ (1 + exp(-(alpha - state(:, 3)) / tau));
bw = sum(sig, 2) * df;

logits = alpha / temperature;
logits = logits - max(logits, [], 2);
weight = exp(logits);
weight = weight ./ sum(weight, 2);
fPeak = weight * freq;
aPeak = sum(weight .* alpha, 2);

targetF = state(:, 1);
limitBw = state(:, 2);
alphaMin = state(:, 3);
bandType = state(:, 4);

dFreq = 2 * (fPeak - targetF) / (15^2);
dAlpha = zeros(nBatch, 1);
below = aPeak < alphaMin;
dAlpha(below) = -2 * (alphaMin(below) - aPeak(below)) / (0.05^2);
dBw = zeros(nBatch, 1);
narrow = bandType < 0 & bw > limitBw;
wide = bandType > 0 & bw < limitBw;
dBw(narrow) = 2 * (bw(narrow) - limitBw(narrow)) / (25^2);
dBw(wide) = -2 * (limitBw(wide) - bw(wide)) / (25^2);

freqLoss = ((fPeak - targetF) / 15).^2;
alphaLoss = (max(0, alphaMin - aPeak) / 0.05).^2;
bwTerm = zeros(nBatch, 1);
bwTerm(bandType < 0) = (max(0, bw(bandType < 0) - limitBw(bandType < 0)) / 25).^2;
bwTerm(bandType > 0) = (max(0, limitBw(bandType > 0) - bw(bandType > 0)) / 25).^2;
loss = freqLoss + bwTerm + alphaLoss;

dsig = sig .* (1 - sig) / tau * df;
% softmax: dfp/da_j = w_j (f_j - fp) / T
% dap/da_j = w_j * (1 + (a_j - ap) / T)
centerF = fPeak .* ones(1, nFreq);
centerA = aPeak .* ones(1, nFreq);
gradAlpha = dFreq .* (weight .* (freq(:).' - centerF) / temperature) ...
    + dBw .* dsig ...
    + dAlpha .* (weight .* (1 + (alpha - centerA) / temperature));
end

%% ===== mpp_absorption.m =====
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

%% ===== train_state_tandem.m =====
function model = train_state_tandem(freq, fluid, nSamples)
%TRAIN_STATE_TANDEM  Train the tandem pair whose inverse input is a state.
%   State columns are [f_center_Hz, bandwidth_Hz, alpha_min, band_type].
%   band_type = -1 narrow (bandwidth below the given value), +1 wide
%   (bandwidth above it), 0 match that bandwidth. The curve itself is never
%   the network input.

if nargin < 3
    nSamples = 1200;
end
rng(0);
freq = freq(:);
[low, high] = parameter_bounds();
theta = low + rand(nSamples, 20) .* (high - low);
alpha = zeros(nSamples, numel(freq));
fprintf('Generating %d transfer-matrix spectra...\n', nSamples);
for i0 = 1:64:nSamples
    i1 = min(nSamples, i0 + 63);
    alpha(i0:i1, :) = mpp_absorption(theta(i0:i1, :), freq, fluid);
end
unit = normalize_theta(theta);
[fPeak, bandwidth, aPeak] = measure_state(freq, alpha, 0.5);
keep = aPeak >= 0.55;
bandType = zeros(nSamples, 1);
bandType(bandwidth < 80) = -1;
bandType(bandwidth >= 80) = 1;
states = [fPeak, max(bandwidth, 2), 0.5 * ones(nSamples, 1), bandType];
states = states(keep, :);

model.forward = init_net(20, [96 96], numel(freq));
model.inverse = init_net(4, [96 96], 20);
model.freq = freq;

fprintf('Training the forward network...\n');
model.forward = train_regression(model.forward, unit, alpha, 40, 1e-3);
fprintf('Training the inverse network...\n');
model.inverse = train_inverse(model.inverse, model.forward, states, freq, 40, 1e-3);
end

function net = init_net(nIn, hidden, nOut)
sizes = [nIn, hidden, nOut];
net.W = cell(1, numel(sizes) - 1);
net.b = cell(1, numel(sizes) - 1);
net.mW = cell(1, numel(sizes) - 1);
net.mB = cell(1, numel(sizes) - 1);
net.vW = cell(1, numel(sizes) - 1);
net.vB = cell(1, numel(sizes) - 1);
for layer = 1:numel(sizes) - 1
    net.W{layer} = randn(sizes(layer), sizes(layer + 1)) * sqrt(2 / sizes(layer));
    net.b{layer} = zeros(1, sizes(layer + 1));
    net.mW{layer} = zeros(size(net.W{layer}));
    net.mB{layer} = zeros(size(net.b{layer}));
    net.vW{layer} = net.mW{layer};
    net.vB{layer} = net.mB{layer};
end
net.step = 0;
end

function net = train_regression(net, x, y, epochs, lr)
n = size(x, 1);
for epoch = 1:epochs
    order = randperm(n);
    total = 0;
    batches = 0;
    for i0 = 1:64:n
        idx = order(i0:min(n, i0 + 63));
        [pred, cache] = mlp_forward(net, x(idx, :));
        err = pred - y(idx, :);
        total = total + mean(err(:).^2);
        batches = batches + 1;
        grad = 2 * err / size(err, 2);
        net = mlp_update(net, cache, grad, lr);
    end
    if epoch == 1 || mod(epoch, 10) == 0
        fprintf('  forward epoch %d  mse %.5f\n', epoch, total / batches);
    end
end
end

function net = train_inverse(inverse, forward, states, freq, epochs, lr)
n = size(states, 1);
for epoch = 1:epochs
    order = randperm(n);
    total = 0;
    seen = 0;
    for i0 = 1:64:n
        idx = order(i0:min(n, i0 + 63));
        batchState = states(idx, :);
        [unit, cacheInv] = mlp_forward(inverse, normalize_state(batchState));
        [spectrum, cacheFwd] = mlp_forward(forward, unit);
        [loss, gradSpectrum] = soft_state_loss(freq, spectrum, batchState);
        gradUnit = mlp_grad_input(forward, cacheFwd, gradSpectrum);
        inverse = mlp_update(inverse, cacheInv, gradUnit, lr);
        total = total + sum(loss);
        seen = seen + numel(loss);
    end
    if epoch == 1 || mod(epoch, 10) == 0
        fprintf('  inverse epoch %d  state loss %.4f\n', epoch, total / max(seen, 1));
    end
end
net = inverse;
end

function [y, cache] = mlp_forward(net, x)
nLayer = numel(net.W);
cache.x = cell(1, nLayer);
cache.z = cell(1, nLayer);
h = x;
for layer = 1:nLayer - 1
    cache.x{layer} = h;
    z = h * net.W{layer} + net.b{layer};
    cache.z{layer} = z;
    sig = 1 ./ (1 + exp(-z));
    h = z .* sig;
end
cache.x{nLayer} = h;
z = h * net.W{nLayer} + net.b{nLayer};
cache.z{nLayer} = z;
y = 1 ./ (1 + exp(-z));
cache.y = y;
end

function gradX = mlp_grad_input(net, cache, gradY)
[gradX, ~] = backprop(net, cache, gradY);
end

function net = mlp_update(net, cache, gradY, lr)
[~, grads] = backprop(net, cache, gradY);
batch = size(gradY, 1);
net.step = net.step + 1;
b1 = 0.9;
b2 = 0.999;
for layer = 1:numel(net.W)
    gW = grads.W{layer} / batch;
    gB = grads.b{layer} / batch;
    net.mW{layer} = b1 * net.mW{layer} + (1 - b1) * gW;
    net.mB{layer} = b1 * net.mB{layer} + (1 - b1) * gB;
    net.vW{layer} = b2 * net.vW{layer} + (1 - b2) * gW.^2;
    net.vB{layer} = b2 * net.vB{layer} + (1 - b2) * gB.^2;
    mW = net.mW{layer} / (1 - b1^net.step);
    mB = net.mB{layer} / (1 - b1^net.step);
    vW = net.vW{layer} / (1 - b2^net.step);
    vB = net.vB{layer} / (1 - b2^net.step);
    net.W{layer} = net.W{layer} - lr * mW ./ (sqrt(vW) + 1e-8);
    net.b{layer} = net.b{layer} - lr * mB ./ (sqrt(vB) + 1e-8);
end
end

function [gradX, grads] = backprop(net, cache, gradY)
nLayer = numel(net.W);
grads.W = cell(1, nLayer);
grads.b = cell(1, nLayer);
delta = gradY .* cache.y .* (1 - cache.y);
for layer = nLayer:-1:1
    grads.W{layer} = cache.x{layer}.' * delta;
    grads.b{layer} = sum(delta, 1);
    gradX = delta * net.W{layer}.';
    if layer > 1
        z = cache.z{layer - 1};
        sig = 1 ./ (1 + exp(-z));
        delta = gradX .* (sig .* (1 + z .* (1 - sig)));
    end
end
end

%% ===== propose_state.m =====
function theta = propose_state(model, state)
%PROPOSE_STATE  Inverse network: acoustic state -> geometry (metres).
unit = mlp_apply(model.inverse, normalize_state(state));
theta = denormalize_theta(unit);
end

function y = mlp_apply(net, x)
h = x;
for layer = 1:numel(net.W) - 1
    z = h * net.W{layer} + net.b{layer};
    sig = 1 ./ (1 + exp(-z));
    h = z .* sig;
end
z = h * net.W{end} + net.b{end};
y = 1 ./ (1 + exp(-z));
end

%% ===== refine_state.m =====
function theta = refine_state(state, theta0, freq, fluid)
%REFINE_STATE  Differential evolution on the transfer-matrix model.
%   THETA0 from the inverse network is inserted into the initial population.
%   The cost is STATE_LOSS, so the search enforces the acoustic state.

[low, high] = parameter_bounds();
nParam = 20;
popSize = 36;
generations = 22;
population = low + rand(popSize, nParam) .* (high - low);
if ~isempty(theta0)
    population(1, :) = min(max(theta0, low), high);
end

cost = evaluate(population);
[bestCost, bestIdx] = min(cost);
best = population(bestIdx, :);
fprintf('  initial state loss %.4f\n', bestCost);

for generation = 1:generations
    trial = population;
    for i = 1:popSize
        ids = randperm(popSize, 3);
        mutant = population(ids(1), :) + 0.6 * (population(ids(2), :) - population(ids(3), :));
        cross = rand(1, nParam) < 0.7;
        cross(randi(nParam)) = true;
        candidate = population(i, :);
        candidate(cross) = mutant(cross);
        candidate = min(max(candidate, low), high);
        trial(i, :) = candidate;
    end
    trialCost = evaluate(trial);
    better = trialCost < cost;
    population(better, :) = trial(better, :);
    cost(better) = trialCost(better);
    [genBest, genIdx] = min(cost);
    if genBest < bestCost
        bestCost = genBest;
        best = population(genIdx, :);
    end
end
fprintf('  refined state loss %.4f\n', bestCost);
theta = best;

    function value = evaluate(batch)
        spectrum = mpp_absorption(batch, freq, fluid);
        value = state_loss(freq, spectrum, state);
    end
end
