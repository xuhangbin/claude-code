% 折纸腔三段微穿孔板的双 MLP 代理与两状态搜索
%
% 物理模型是 acoustic_nn/origami_cavity_mpp.m 里的单管道：
%   三块微穿孔板（厚度 2 mm，每块 1 个孔）夹两个折纸腔。
%   孔径 d1、d2、d3 两态共用，状态 2 只改两个腔的实际高度。
%   折纸几何固定：a=30 mm，hc=20 mm，n0=3，beta=45°。
%   实际高度 0.5–54 mm。等效高度进入腔体传递矩阵，透射相位用实际高度。
%
% 正向网络  5 -> 64 -> 128 -> 128 -> 2*Nf
% 逆向网络  2*Nf -> 256 -> 256 -> 128 -> 5
% 隐藏层 ReLU，输出线性，Glorot 均匀初始化，Adam，学习率 1e-3，损失 MSE。
% 训练频率 linspace(100,600,101)。最后用传递矩阵在 100:1:600 Hz 上核对并画图。
%
% 直接运行本文件。不需要 Deep Learning Toolbox。
% 短流程：setenv('RUN_MODE','quick'); origami_dual_mlp

function origami_dual_mlp

if exist('OCTAVE_VERSION', 'builtin')
    page_output_immediately(1);
end

rng(1);

quick = strcmp(getenv('RUN_MODE'), 'quick');

const = makeConst();
fTrain = linspace(100, 600, 101);
fPlot = 100:1:600;

if quick
    nBase = 500;
    nPert = 2;
    maxHot = 60;
    maxEpoch = 2;
    invEpoch = 1;
    nRestart = 2;
    nPop = 16;
    nGen = 2;
    nRandom = 240;
    maxRounds = 1;
    batchSize = 128;
else
    nBase = 20000;
    nPert = 5;
    maxHot = 2500;
    maxEpoch = 48;
    invEpoch = 16;
    nRestart = 8;
    nPop = 64;
    nGen = 28;
    nRandom = 36000;
    maxRounds = 3;
    batchSize = 256;
end
learnRate = 1e-3;

%% 样本点与体积自检

verifyPhysics(const, fPlot);

if strcmp(getenv('RUN_MODE'), 'verify')
    fprintf('只做物理自检，到此结束。\n');
    return;
end

checkNetworkGradient();

%% 数据集：均匀样本，再在高吸声样本附近加密

fprintf('生成训练集（均匀 %d 组，并在高峰附近扰动）...\n', nBase);
[X, Y] = buildDataset(nBase, nPert, maxHot, fTrain, const);
fprintf('可用样本 %d 组。\n', size(X, 1));

nVal = max(200, round(0.1 * size(X, 1)));
nVal = min(nVal, size(X, 1) - 2);
perm = randperm(size(X, 1));
X = X(perm, :);
Y = Y(perm, :);
XVal = X(1:nVal, :);
YVal = Y(1:nVal, :);
XTrain = X(nVal + 1:end, :);
YTrain = Y(nVal + 1:end, :);

xMean = mean(XTrain, 1);
xStd = std(XTrain, 0, 1);
xStd(xStd < 1e-12) = 1;
yMean = mean(YTrain, 1);
yStd = std(YTrain, 0, 1);
yStd(yStd < 1e-12) = 1;

XTrainS = (XTrain - xMean) ./ xStd;
XValS = (XVal - xMean) ./ xStd;

%% 正向网络

nSpec = 2 * numel(fTrain);
fprintf('\n===== 训练正向网络：5 -> 64 -> 128 -> 128 -> %d =====\n', nSpec);
forwardNet = initMLP([5, 64, 128, 128, nSpec], {'relu', 'relu', 'relu', 'linear'});
minEpoch = 2;
if ~quick
    minEpoch = 24;
end
forwardNet = trainRegressor(forwardNet, XTrainS, YTrain, XValS, YVal, ...
    maxEpoch, minEpoch, batchSize, learnRate, '正向网络');

YHat = mlpPredict(forwardNet, XValS);
nf = numel(fTrain);
maeF = mean(mean(abs(YHat(:, 1:nf) - YVal(:, 1:nf))));
maeB = mean(mean(abs(YHat(:, nf + 1:end) - YVal(:, nf + 1:end))));
fprintf('验证集 MAE  正向 %.4f  反向 %.4f\n', maeF, maeB);
flushOut();

%% 逆向网络（冻住正向网络）

fprintf('\n===== 训练逆向网络：%d -> 256 -> 256 -> 128 -> 5 =====\n', nSpec);
YTrainS = (YTrain - yMean) ./ yStd;
YValS = (YVal - yMean) ./ yStd;
inverseNet = initMLP([nSpec, 256, 256, 128, 5], {'relu', 'relu', 'relu', 'linear'});
inverseNet = trainAutoencoder(inverseNet, forwardNet, YTrainS, YTrain, ...
    YValS, YVal, invEpoch, batchSize, learnRate);

%% 用网络搜索，再用传递矩阵打分。未达标就在高峰附近补样本继续训。

lo = [const.dMin, const.dMin, const.dMin, const.Hmin, const.Hmin, const.Hmin, const.Hmin];
hi = [const.dMax, const.dMax, const.dMax, const.Hmax, const.Hmax, const.Hmax, const.Hmax];

best = [];
for roundId = 1:maxRounds
    fprintf('\n===== 第 %d 轮：网络搜索，传递矩阵复算 =====\n', roundId);
    seeds = inverseSeeds(inverseNet, fTrain, yMean, yStd, xMean, xStd, lo, hi);
    nominal = [2.75, 1.73, 0.42, 54, 42, 20, 30];
    seeds = [nominal; seeds]; %#ok<AGROW>
    seeds = min(max(seeds, lo), hi);

    pool = proposePool(forwardNet, xMean, xStd, fTrain, seeds, ...
        lo, hi, nRestart, nPop, nGen, nRandom);
    fprintf('传递矩阵复算 %d 个候选...\n', size(pool, 1));
    scored = rankByPhysics(pool, fPlot, const);
    printReport(scored.report, '本轮最优');
    if isempty(best) || betterDesign(scored, best)
        best = scored;
    end
    if scored.report.allOk
        fprintf('传递矩阵判定两态都达标，停止搜索。\n');
        break;
    end
    if roundId == maxRounds
        break;
    end
    centers = scored.centers;
    fprintf('未达标。在 %d 个较优几何附近补样本并继续训练正向网络。\n', size(centers, 1));
    [Xn, Yn] = resonantCloud(centers, fTrain, const, quick);
    XTrain = [XTrain; Xn]; %#ok<AGROW>
    YTrain = [YTrain; Yn]; %#ok<AGROW>
    XTrainS = (XTrain - xMean) ./ xStd;
    extraEpoch = 8;
    if ~quick
        extraEpoch = 16;
    end
    forwardNet = trainRegressor(forwardNet, XTrainS, YTrain, XValS, YVal, ...
        extraEpoch, extraEpoch, batchSize, learnRate, '正向网络续训');
    inverseNet = trainAutoencoder(inverseNet, forwardNet, ...
        (YTrain - yMean) ./ yStd, YTrain, YValS, YVal, ...
        max(1, round(invEpoch / 2)), batchSize, learnRate);
    flushOut();
end

report = best.report;
dMm = best.dMm;
H1mm = best.H1mm;
H2mm = best.H2mm;
af1 = best.af1;
ab1 = best.ab1;
af2 = best.af2;
ab2 = best.ab2;

[~, ~, report.energy1] = acousticBatch(dMm, H1mm, fPlot, const);
[~, ~, report.energy2] = acousticBatch(dMm, H2mm, fPlot, const);

fprintf('\n========== 传递矩阵最终结果 ==========\n');
printReport(report, '最终');
fprintf('孔径 d1 d2 d3 = %.4f  %.4f  %.4f mm\n', dMm(1), dMm(2), dMm(3));
fprintf('状态1 实际高度 H1 H2 = %.4f  %.4f mm\n', H1mm(1), H1mm(2));
fprintf('状态2 实际高度 H1 H2 = %.4f  %.4f mm\n', H2mm(1), H2mm(2));
fprintf('能量误差 状态1 %.3e  状态2 %.3e\n', report.energy1, report.energy2);

outDir = fileparts(mfilename('fullpath'));
if isempty(outDir)
    outDir = pwd;
end
resultPath = fullfile(outDir, 'origami_two_state.mat');
figPath = fullfile(outDir, 'origami_two_state.png');
save(resultPath, 'fPlot', 'fTrain', 'dMm', 'H1mm', 'H2mm', ...
    'af1', 'ab1', 'af2', 'ab2', 'report', ...
    'forwardNet', 'inverseNet', 'xMean', 'xStd', 'yMean', 'yStd');
fprintf('参数已保存到 %s\n', resultPath);

plotTwoStates(fPlot, af1, ab1, af2, ab2, report, figPath);
fprintf('图已保存到 %s\n', figPath);
end

%% 常量与物理

function const = makeConst()
const.rho = 1.29;
const.c0 = 343;
const.eta = 1.89e-5;
const.Z0 = const.rho * const.c0;
const.t = 2e-3;
const.W = 30e-3;
const.L = 30e-3;
const.a = 30;
const.hc = 20;
const.n0 = 3;
const.beta = 45;
const.Hmin = 0.5;
const.Hmax = 54;
const.dMin = 0.1;
const.dMax = 8;
end

function verifyPhysics(const, fPlot)
[Heq1, V1] = cavityVolume(54, const);
[Heq2, V2] = cavityVolume(42, const);
fprintf('自检 H=54 mm -> V=%.4f mm^3，Heq=%.4f mm\n', V1, Heq1);
fprintf('自检 H=42 mm -> V=%.4f mm^3，Heq=%.4f mm\n', V2, Heq2);
if abs(V1 - 36730.3710) > 1e-2 || abs(Heq1 - 40.8115) > 1e-3
    error('第一腔体积或等效高度与给定值不符。');
end
if abs(V2 - 23231.2003) > 1e-2 || abs(Heq2 - 25.8124) > 1e-3
    error('第二腔体积或等效高度与给定值不符。');
end

[af, ab, energyErr] = acousticBatch([2.75, 1.73, 0.42], [54, 42], 100, const);
fprintf('自检 100 Hz 正向 %.8f 反向 %.8f 能量误差 %.3e\n', af, ab, energyErr);
if abs(af - 0.07867460) > 1e-7 || abs(ab - 0.03191483) > 1e-7
    error('100 Hz 吸声系数与 origami_cavity_mpp.csv 不符。');
end
if energyErr > 1e-8
    error('样本点能量误差过大。');
end

[afS, abS] = acousticBatch([2.75, 1.73, 0.42], [54, 42], fPlot, const);
info = analyzeCurves(afS, abS, abS, afS, fPlot);
if info.meanB1 < 0.45
    fprintf('样本几何的正向宽带平均 %.3f，低于 0.45，只作自检。\n', info.meanB1);
else
    fprintf('样本几何在 %.0f–%.0f Hz 正向平均 %.3f，反向平均 %.3f。\n', ...
        info.fLo1, info.fHi1, info.meanB1, info.meanO1);
end
end

function [Heq_mm, V] = cavityVolume(H_mm, const)
h = H_mm ./ const.n0;
q = sqrt(const.hc.^2 - h.^2);
V1 = const.a .* h .* q / 16;
V2 = h .* (const.a - const.hc) .* q / (16 * tand(const.beta));
V3 = h .* q / 24 .* (const.hc / tand(const.beta) - q / 2);
V = 8 * const.n0 .* (const.a.^2 .* h / 8 - V1 - V2 - V3);
S = const.W * const.L * 1e6;
Heq_mm = V ./ S;
end

function [af, ab, energyErr] = acousticBatch(d_mm, H_mm, f, const)
d_mm = reshapeBatch(d_mm, 3);
H_mm = reshapeBatch(H_mm, 2);
f = f(:).';
d = d_mm * 1e-3;
H = H_mm * 1e-3;
omega = 2 * pi * f;
k = omega / const.c0;
B = size(d, 1);
Nf = numel(f);
Heq = [cavityVolume(H_mm(:, 1), const), cavityVolume(H_mm(:, 2), const)];
Heq = Heq * 1e-3;
phi = pi * (d.^2) / (4 * const.W * const.L);
Z1 = plateImpedance(d(:, 1), phi(:, 1), omega, const);
Z2 = plateImpedance(d(:, 2), phi(:, 2), omega, const);
Z3 = plateImpedance(d(:, 3), phi(:, 3), omega, const);
T11 = ones(B, Nf);
T22 = ones(B, Nf);
T12 = Z1;
T21 = zeros(B, Nf);
[T11, T12, T21, T22] = mulCavity(T11, T12, T21, T22, Heq(:, 1), k, const.Z0);
[T11, T12, T21, T22] = mulPlate(T11, T12, T21, T22, Z2);
[T11, T12, T21, T22] = mulCavity(T11, T12, T21, T22, Heq(:, 2), k, const.Z0);
[T11, T12, T21, T22] = mulPlate(T11, T12, T21, T22, Z3);
D = T11 + T12 / const.Z0 + T21 * const.Z0 + T22;
Ltot = 3 * const.t + H(:, 1) + H(:, 2);
tau = 2 * exp(1i * (Ltot * k)) ./ D;
rf = (T11 + T12 / const.Z0 - T21 * const.Z0 - T22) ./ D;
rb = (-T11 + T12 / const.Z0 - T21 * const.Z0 + T22) ./ D;
Rf = abs(rf).^2;
Rb = abs(rb).^2;
Tr = abs(tau).^2;
af = 1 - Rf - Tr;
ab = 1 - Rb - Tr;
good = isfinite(af) & isfinite(ab) & isfinite(Rf) & isfinite(Rb) & isfinite(Tr);
if any(good(:))
    e1 = abs(af + Rf + Tr - 1);
    e2 = abs(ab + Rb + Tr - 1);
    energyErr = max(max(e1(good)), max(e2(good)));
else
    energyErr = inf;
end
af(~isfinite(af)) = 0;
ab(~isfinite(ab)) = 0;
end

function M = reshapeBatch(M, nCol)
if isvector(M) && numel(M) == nCol
    M = reshape(M, 1, nCol);
end
end

function Z = plateImpedance(d, phi, omega, const)
y = (d / 2) ./ sqrt(const.eta ./ (const.rho * omega));
q = y * sqrt(-1i);
invF = (1 - 2 * besselj(1, q) ./ (q .* besselj(0, q))).^(-1);
Z = 1i .* omega .* const.rho .* const.t ./ phi .* invF ...
    + sqrt(2) * const.eta .* y ./ (phi .* d) ...
    + 1i * 0.85 .* omega .* const.rho .* d ./ phi;
Z = reshape(Z, numel(d), numel(omega));
end

function [C11, C12, C21, C22] = mulPlate(A11, A12, A21, A22, Z)
C11 = A11;
C21 = A21;
C12 = A11 .* Z + A12;
C22 = A21 .* Z + A22;
end

function [C11, C12, C21, C22] = mulCavity(A11, A12, A21, A22, Heq, k, Z0)
kh = Heq .* k;
c = cos(kh);
s = sin(kh);
B11 = c;
B22 = c;
B12 = 1i * Z0 .* s;
B21 = 1i * s ./ Z0;
C11 = A11 .* B11 + A12 .* B21;
C12 = A11 .* B12 + A12 .* B22;
C21 = A21 .* B11 + A22 .* B21;
C22 = A21 .* B12 + A22 .* B22;
end

%% 数据集

function [X, Y] = buildDataset(nBase, nPert, maxHot, f, const)
[X, Y] = sampleUniform(nBase, f, const);
peak = max(Y, [], 2);
hot = find(peak >= 0.72);
if numel(hot) < 30
    hot = find(peak >= 0.45);
end
if isempty(hot)
    return;
end
if numel(hot) > maxHot
    hot = hot(randperm(numel(hot), maxHot));
end
sig = [0.15, 0.10, 0.15, 2.5, 2.5];
nH = numel(hot);
Xp = zeros(nH * nPert, 5);
for i = 1:nPert
    rows = (i - 1) * nH + (1:nH);
    Xp(rows, :) = X(hot, :) + randn(nH, 5) .* sig;
end
lo = [const.dMin, const.dMin, const.dMin, const.Hmin, const.Hmin];
hi = [const.dMax, const.dMax, const.dMax, const.Hmax, const.Hmax];
Xp = min(max(Xp, lo), hi);
[af, ab] = acousticChunk(Xp(:, 1:3), Xp(:, 4:5), f, const, 4000);
Yp = [af, ab];
good = all(isfinite(Yp), 2);
X = [X; Xp(good, :)];
Y = [Y; Yp(good, :)];
fprintf('  高峰附近追加 %d 组。\n', sum(good));
end

function [X, Y] = sampleUniform(n, f, const)
d = const.dMin + (const.dMax - const.dMin) * rand(n, 3);
H = const.Hmin + (const.Hmax - const.Hmin) * rand(n, 2);
[af, ab] = acousticChunk(d, H, f, const, 4000);
X = [d, H];
Y = [af, ab];
good = all(isfinite(Y), 2);
X = X(good, :);
Y = Y(good, :);
end

function [af, ab] = acousticChunk(d, H, f, const, chunk)
n = size(d, 1);
nf = numel(f);
af = zeros(n, nf);
ab = zeros(n, nf);
for s = 1:chunk:n
    e = min(s + chunk - 1, n);
    [af(s:e, :), ab(s:e, :)] = acousticBatch(d(s:e, :), H(s:e, :), f, const);
end
end

function [Xn, Yn] = resonantCloud(centers7, f, const, quick)
lo5 = [const.dMin, const.dMin, const.dMin, const.Hmin, const.Hmin];
hi5 = [const.dMax, const.dMax, const.dMax, const.Hmax, const.Hmax];
base = [centers7(:, 1:5); [centers7(:, 1:3), centers7(:, 6:7)]];
base = dedupe(base, 0.15);
nPer = 12;
nUni = 2500;
if quick
    nPer = 2;
    nUni = 80;
end
sigW = [0.28, 0.16, 0.28, 4.5, 4.5];
sigT = [0.08, 0.05, 0.08, 1.4, 1.4];
nB = size(base, 1);
Xw = repmat(base, nPer, 1) + randn(nB * nPer, 5) .* sigW;
Xt = repmat(base, nPer, 1) + randn(nB * nPer, 5) .* sigT;
Xu = lo5 + (hi5 - lo5) .* rand(nUni, 5);
Xn = min(max([Xw; Xt; Xu], lo5), hi5);
[af, ab] = acousticChunk(Xn(:, 1:3), Xn(:, 4:5), f, const, 4000);
Yn = [af, ab];
good = all(isfinite(Yn), 2);
Xn = Xn(good, :);
Yn = Yn(good, :);
fprintf('  续训样本 %d 组。\n', size(Xn, 1));
end

%% 网络

function net = initMLP(sizes, acts)
net.W = cell(1, numel(sizes) - 1);
net.b = cell(1, numel(sizes) - 1);
net.act = acts;
for k = 1:numel(sizes) - 1
    limit = sqrt(6 / (sizes(k) + sizes(k + 1)));
    net.W{k} = (2 * rand(sizes(k + 1), sizes(k)) - 1) * limit;
    net.b{k} = zeros(sizes(k + 1), 1);
end
end

function Y = mlpPredict(net, X)
cache = mlpForward(net, X.');
Y = cache.y.';
end

function cache = mlpForward(net, X)
h = X;
for k = 1:numel(net.W)
    z = net.W{k} * h + net.b{k};
    cache.z{k} = z;
    cache.h{k} = h;
    if strcmp(net.act{k}, 'relu')
        h = max(z, 0);
    else
        h = z;
    end
end
cache.y = h;
end

function grads = mlpBackward(net, cache, dY)
g = dY;
nLayer = numel(net.W);
grads.W = cell(1, nLayer);
grads.b = cell(1, nLayer);
for k = nLayer:-1:1
    if strcmp(net.act{k}, 'relu')
        g = g .* (cache.z{k} > 0);
    end
    grads.W{k} = g * cache.h{k}.';
    grads.b{k} = sum(g, 2);
    g = net.W{k}.' * g;
end
end

function adam = initAdam(net)
adam.t = 0;
for k = 1:numel(net.W)
    adam.mW{k} = zeros(size(net.W{k}));
    adam.vW{k} = zeros(size(net.W{k}));
    adam.mb{k} = zeros(size(net.b{k}));
    adam.vb{k} = zeros(size(net.b{k}));
end
end

function [net, adam] = adamApply(net, adam, grads, learnRate)
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;
adam.t = adam.t + 1;
for k = 1:numel(net.W)
    adam.mW{k} = beta1 * adam.mW{k} + (1 - beta1) * grads.W{k};
    adam.vW{k} = beta2 * adam.vW{k} + (1 - beta2) * (grads.W{k}.^2);
    adam.mb{k} = beta1 * adam.mb{k} + (1 - beta1) * grads.b{k};
    adam.vb{k} = beta2 * adam.vb{k} + (1 - beta2) * (grads.b{k}.^2);
    mW = adam.mW{k} / (1 - beta1^adam.t);
    vW = adam.vW{k} / (1 - beta2^adam.t);
    mb = adam.mb{k} / (1 - beta1^adam.t);
    vb = adam.vb{k} / (1 - beta2^adam.t);
    net.W{k} = net.W{k} - learnRate * mW ./ (sqrt(vW) + epsilon);
    net.b{k} = net.b{k} - learnRate * mb ./ (sqrt(vb) + epsilon);
end
end

function net = trainRegressor(net, X, Y, XVal, YVal, epochs, minEpoch, batchSize, learnRate, name)
adam = initAdam(net);
n = size(X, 1);
useBatch = min(batchSize, n);
bestMae = inf;
bestNet = net;
stall = 0;
for epoch = 1:epochs
    order = randperm(n);
    total = 0;
    seen = 0;
    for s = 1:useBatch:n
        batch = order(s:min(s + useBatch - 1, n));
        cache = mlpForward(net, X(batch, :).');
        residual = cache.y - Y(batch, :).';
        total = total + sum(residual(:).^2);
        seen = seen + numel(residual);
        grads = mlpBackward(net, cache, 2 * residual / numel(residual));
        [net, adam] = adamApply(net, adam, grads, learnRate);
    end
    pred = mlpPredict(net, XVal);
    mae = mean(abs(pred(:) - YVal(:)));
    fprintf('%s epoch %02d  train_mse=%.6e  val_mae=%.6e\n', ...
        name, epoch, total / max(seen, 1), mae);
    flushOut();
    if mae + 1e-5 < bestMae
        bestMae = mae;
        bestNet = copyNet(net);
        stall = 0;
    else
        stall = stall + 1;
    end
    if epoch >= minEpoch && stall >= 8
        fprintf('%s 验证误差停住，采用第 %d 轮之前的最优权重。\n', name, epoch);
        break;
    end
end
net = bestNet;
end

function inverseNet = trainAutoencoder(inverseNet, forwardNet, SStd, SRaw, ...
    SValStd, SValRaw, epochs, batchSize, learnRate)
adam = initAdam(inverseNet);
n = size(SStd, 1);
useBatch = min(batchSize, n);
for epoch = 1:epochs
    order = randperm(n);
    total = 0;
    seen = 0;
    for s = 1:useBatch:n
        batch = order(s:min(s + useBatch - 1, n));
        invCache = mlpForward(inverseNet, SStd(batch, :).');
        fwdCache = mlpForward(forwardNet, invCache.y);
        residual = fwdCache.y - SRaw(batch, :).';
        total = total + sum(residual(:).^2);
        seen = seen + numel(residual);
        dParam = backwardToInput(forwardNet, fwdCache, 2 * residual / numel(residual));
        grads = mlpBackward(inverseNet, invCache, dParam);
        [inverseNet, adam] = adamApply(inverseNet, adam, grads, learnRate);
    end
    pred = mlpPredict(inverseNet, SValStd);
    recon = mlpPredict(forwardNet, pred);
    valMse = mean((recon(:) - SValRaw(:)).^2);
    fprintf('逆向网络 epoch %02d  train_mse=%.6e  val_mse=%.6e\n', ...
        epoch, total / max(seen, 1), valMse);
    flushOut();
end
end

function dX = backwardToInput(net, cache, dY)
g = dY;
for k = numel(net.W):-1:1
    if strcmp(net.act{k}, 'relu')
        g = g .* (cache.z{k} > 0);
    end
    g = net.W{k}.' * g;
end
dX = g;
end

function checkNetworkGradient()
net = initMLP([3, 5, 2], {'relu', 'linear'});
X = randn(3, 4);
cache = mlpForward(net, X);
dY = randn(size(cache.y));
grads = mlpBackward(net, cache, dY);
delta = 1e-6;
numeric = zeros(size(net.W{1}));
for row = 1:size(net.W{1}, 1)
    for col = 1:size(net.W{1}, 2)
        net.W{1}(row, col) = net.W{1}(row, col) + delta;
        yp = mlpForward(net, X);
        net.W{1}(row, col) = net.W{1}(row, col) - 2 * delta;
        ym = mlpForward(net, X);
        net.W{1}(row, col) = net.W{1}(row, col) + delta;
        numeric(row, col) = sum(dY(:) .* (yp.y(:) - ym.y(:))) / (2 * delta);
    end
end
err = max(abs(numeric(:) - grads.W{1}(:)));
dX = backwardToInput(net, cache, dY);
Xp = X;
Xp(1, 1) = Xp(1, 1) + delta;
yp = mlpForward(net, Xp);
Xm = X;
Xm(1, 1) = Xm(1, 1) - delta;
ym = mlpForward(net, Xm);
numIn = sum(dY(:) .* (yp.y(:) - ym.y(:))) / (2 * delta);
err = max(err, abs(numIn - dX(1, 1)));
if err > 1e-5
    error('网络反向传播自检失败，误差 %.3e', err);
end
fprintf('网络反向传播自检通过，最大误差 %.3e\n', err);
end

function out = copyNet(net)
out.act = net.act;
out.W = cell(size(net.W));
out.b = cell(size(net.b));
for k = 1:numel(net.W)
    out.W{k} = net.W{k};
    out.b{k} = net.b{k};
end
end

%% 搜索

function seeds = inverseSeeds(inverseNet, f, yMean, yStd, xMean, xStd, lo, hi)
broadLos = [140, 220, 300, 380];
peakHzs = [180, 260, 360, 470, 540];
seeds = zeros(0, 7);
for ib = 1:numel(broadLos)
    for ip = 1:numel(peakHzs)
        if abs(peakHzs(ip) - (broadLos(ib) + 100)) < 70
            continue;
        end
        tForward = shapeTarget(f, true, broadLos(ib), peakHzs(ip));
        tBackward = shapeTarget(f, false, broadLos(ib), max(140, peakHzs(ip) - 160));
        p1 = decodeTarget(inverseNet, tForward, yMean, yStd, xMean, xStd);
        p2 = decodeTarget(inverseNet, tBackward, yMean, yStd, xMean, xStd);
        d = 0.5 * (p1(1:3) + p2(1:3));
        row = [d, p1(4:5), p2(4:5)];
        row = min(max(row, lo), hi);
        seeds = [seeds; row]; %#ok<AGROW>
    end
end
if size(seeds, 1) > 12
    seeds = seeds(1:12, :);
end
end

function s = shapeTarget(f, forwardIsBroad, broadLo, peakHz)
af = 0.12 * ones(size(f));
ab = 0.12 * ones(size(f));
band = f >= broadLo & f <= broadLo + 200;
peak = 0.92 * exp(-0.5 * ((f - peakHz) / 16).^2);
if forwardIsBroad
    af(band) = 0.70;
    ab = max(ab, peak);
else
    ab(band) = 0.70;
    af = max(af, peak);
end
s = [af, ab];
end

function x = decodeTarget(inverseNet, target, yMean, yStd, xMean, xStd)
z = mlpPredict(inverseNet, (target - yMean) ./ yStd);
x = z .* xStd + xMean;
end

function pool = proposePool(net, xMean, xStd, f, seeds, lo, hi, nRestart, nPop, nGen, nRandom)
pool = seeds;
chunk = 4000;
keepEach = 60;
if nRandom < chunk
    chunk = nRandom;
    keepEach = max(20, round(nRandom / 4));
end
bag = zeros(0, 7);
bagL = zeros(0, 1);
fprintf('网络随机筛选 %d 个几何...\n', nRandom);
for s = 1:chunk:nRandom
    n = min(chunk, nRandom - s + 1);
    X = lo + (hi - lo) .* rand(n, 7);
    L = populationLoss(net, xMean, xStd, X, f);
    [Ls, order] = sort(L);
    k = min(keepEach, n);
    bag = [bag; X(order(1:k), :)]; %#ok<AGROW>
    bagL = [bagL; Ls(1:k)]; %#ok<AGROW>
end
[~, order] = sort(bagL);
nKeep = min(400, numel(order));
pool = [pool; bag(order(1:nKeep), :)]; %#ok<AGROW>
fprintf('  随机筛选最优网络损失 %.4f\n', bagL(order(1)));

bestX = [];
bestL = inf;
for r = 1:nRestart
    pop = lo + (hi - lo) .* rand(nPop, 7);
    nSeed = min(size(seeds, 1), max(1, floor(nPop / 5)));
    pop(1:nSeed, :) = seeds(randperm(size(seeds, 1), nSeed), :);
    if ~isempty(bestX)
        pop(2, :) = bestX;
    end
    for g = 1:nGen
        L = populationLoss(net, xMean, xStd, pop, f);
        [Ls, order] = sort(L);
        pop = pop(order, :);
        kTop = min(6, nPop);
        pool = [pool; pop(1:kTop, :)]; %#ok<AGROW>
        if Ls(1) < bestL
            bestL = Ls(1);
            bestX = pop(1, :);
        end
        if g == nGen || mod(g, 7) == 0
            fprintf('  重启 %d  第 %02d 代  网络损失 %.4f\n', r, g, Ls(1));
            flushOut();
        end
        pop = reproduce(pop, nPop, lo, hi);
    end
end
pool = dedupe(min(max(pool, lo), hi), 0.04);
fprintf('去重后候选 %d 个。网络最优损失 %.4f\n', size(pool, 1), bestL);
end

function L = populationLoss(net, xMean, xStd, pop, f)
y1 = mlpPredict(net, ([pop(:, 1:3), pop(:, 4:5)] - xMean) ./ xStd);
y2 = mlpPredict(net, ([pop(:, 1:3), pop(:, 6:7)] - xMean) ./ xStd);
nf = numel(f);
af1 = min(max(y1(:, 1:nf), 0), 1);
ab1 = min(max(y1(:, nf + 1:end), 0), 1);
af2 = min(max(y2(:, 1:nf), 0), 1);
ab2 = min(max(y2(:, nf + 1:end), 0), 1);
L = batchShapeLoss(af1, ab1, af2, ab2, f);
end

function pop = reproduce(popSorted, nPop, lo, hi)
nElite = min(4, size(popSorted, 1));
elites = popSorted(1:nElite, :);
nVar = numel(lo);
nKids = nPop - nElite;
parentPool = max(nElite, ceil(size(popSorted, 1) / 3));
i1 = randi(parentPool, nKids, 1);
i2 = randi(parentPool, nKids, 1);
mix = rand(nKids, nVar);
kids = mix .* popSorted(i1, :) + (1 - mix) .* popSorted(i2, :);
span = hi - lo;
mut = rand(nKids, nVar) < 0.28;
kids = kids + mut .* (0.12 .* span .* randn(nKids, nVar));
jump = rand(nKids, 1) < 0.15;
if any(jump)
    kids(jump, 4:7) = kids(jump, 4:7) + randn(sum(jump), 4) .* (0.18 .* span(4:7));
end
kids = min(max(kids, lo), hi);
pop = [elites; kids];
end

function L = batchShapeLoss(af1, ab1, af2, ab2, f)
[mb1, mo1] = pickWindow(af1, ab1, f);
[mb2, mo2] = pickWindow(ab2, af2, f);
[h1, f1, w1] = batchGlobalPeak(ab1, f);
[h2, f2, w2] = batchGlobalPeak(af2, f);
shift = abs(f1 - f2);
L = 20 * (max(0, 0.45 - mb1).^2 + max(0, 0.20 - (mb1 - mo1)).^2 ...
    + max(0, 0.45 - mb2).^2 + max(0, 0.20 - (mb2 - mo2)).^2 ...
    + max(0, 0.80 - h1).^2 + max(0, (w1 - 80) / 80).^2 ...
    + max(0, 0.80 - h2).^2 + max(0, (w2 - 80) / 80).^2 ...
    + (max(0, 100 - shift) / 100).^2) ...
    - 0.04 * (mb1 + mb2) - 0.015 * ((mb1 - mo1) + (mb2 - mo2)) - 0.01 * (h1 + h2);
end

function [mb, mo] = pickWindow(broad, other, f)
df = mean(diff(f));
nWin = max(2, round(200 / df));
nWin = min(nWin, size(broad, 2));
cB = [zeros(size(broad, 1), 1), cumsum(broad, 2)];
cO = [zeros(size(other, 1), 1), cumsum(other, 2)];
nOut = size(broad, 2) - nWin + 1;
meanB = (cB(:, nWin + 1:end) - cB(:, 1:nOut)) / nWin;
meanO = (cO(:, nWin + 1:end) - cO(:, 1:nOut)) / nWin;
margin = min(meanB - 0.45, (meanB - meanO) - 0.20);
[~, idx] = max(margin, [], 2);
rows = (1:size(broad, 1)).';
mb = meanB(sub2ind(size(meanB), rows, idx));
mo = meanO(sub2ind(size(meanO), rows, idx));
end

function [height, freq, width] = batchGlobalPeak(a, f)
[height, idx] = max(a, [], 2);
df = mean(diff(f));
N = size(a, 2);
B = size(a, 1);
width = zeros(B, 1);
above = a >= 0.5;
for b = 1:B
    i = idx(b);
    if height(b) < 0.5
        width(b) = 0;
    else
        lo = i;
        hi = i;
        while lo > 1 && above(b, lo - 1)
            lo = lo - 1;
        end
        while hi < N && above(b, hi + 1)
            hi = hi + 1;
        end
        width(b) = (hi - lo + 1) * df;
    end
end
freq = f(idx);
freq = freq(:);
height = height(:);
end

function scored = rankByPhysics(pool, f, const)
n = size(pool, 1);
af = zeros(n, numel(f));
ab = zeros(n, numel(f));
af2 = zeros(n, numel(f));
ab2 = zeros(n, numel(f));
chunk = 300;
for s = 1:chunk:n
    e = min(s + chunk - 1, n);
    d = pool(s:e, 1:3);
    [af(s:e, :), ab(s:e, :)] = acousticBatch(d, pool(s:e, 4:5), f, const);
    [af2(s:e, :), ab2(s:e, :)] = acousticBatch(d, pool(s:e, 6:7), f, const);
end
loss = inf(n, 1);
ok = false(n, 1);
infos = cell(n, 1);
for i = 1:n
    infos{i} = analyzeCurves(af(i, :), ab(i, :), af2(i, :), ab2(i, :), f);
    loss(i) = infos{i}.loss;
    ok(i) = infos{i}.allOk;
end
if any(ok)
    idx = find(ok);
    [~, j] = min(loss(idx));
    best = idx(j);
else
    [~, best] = min(loss);
end
[~, order] = sort(loss);
nCenter = min(12, n);
centers = dedupe(pool(order(1:nCenter), :), 0.2);
if size(centers, 1) < 4
    centers = pool(order(1:min(8, n)), :);
end
scored.report = infos{best};
scored.dMm = pool(best, 1:3);
scored.H1mm = pool(best, 4:5);
scored.H2mm = pool(best, 6:7);
scored.af1 = af(best, :);
scored.ab1 = ab(best, :);
scored.af2 = af2(best, :);
scored.ab2 = ab2(best, :);
scored.centers = centers;
scored.loss = loss(best);
scored.nFeasible = sum(ok);
fprintf('达标候选 %d / %d，选中损失 %.4f\n', sum(ok), n, loss(best));
end

function yes = betterDesign(a, b)
if a.report.allOk && ~b.report.allOk
    yes = true;
elseif ~a.report.allOk && b.report.allOk
    yes = false;
else
    yes = a.loss < b.loss;
end
end

function info = analyzeCurves(af1, ab1, af2, ab2, f)
af1 = af1(:).';
ab1 = ab1(:).';
af2 = af2(:).';
ab2 = ab2(:).';
[info.fLo1, info.fHi1, info.meanB1, info.meanO1] = bestWindow(f, af1, ab1);
[info.fLo2, info.fHi2, info.meanB2, info.meanO2] = bestWindow(f, ab2, af2);
info.gap1 = info.meanB1 - info.meanO1;
info.gap2 = info.meanB2 - info.meanO2;
p1 = localPeaks(ab1, f);
p2 = localPeaks(af2, f);
pk = choosePeakPair(p1, p2);
info.h1 = pk(1);
info.fp1 = pk(2);
info.w1 = pk(3);
info.h2 = pk(4);
info.fp2 = pk(5);
info.w2 = pk(6);
info.shift = abs(info.fp1 - info.fp2);
info.okBroad1 = info.meanB1 >= 0.45 && info.gap1 >= 0.20;
info.okNarrow1 = info.h1 >= 0.80 && info.w1 <= 80;
info.ok1 = info.okBroad1 && info.okNarrow1;
info.okBroad2 = info.meanB2 >= 0.45 && info.gap2 >= 0.20;
info.okNarrow2 = info.h2 >= 0.80 && info.w2 <= 80;
info.ok2 = info.okBroad2 && info.okNarrow2;
info.okShift = info.shift >= 100;
info.allOk = info.ok1 && info.ok2 && info.okShift;
info.loss = 20 * (max(0, 0.45 - info.meanB1)^2 + max(0, 0.20 - info.gap1)^2 ...
    + max(0, 0.45 - info.meanB2)^2 + max(0, 0.20 - info.gap2)^2 ...
    + max(0, 0.80 - info.h1)^2 + max(0, (info.w1 - 80) / 80)^2 ...
    + max(0, 0.80 - info.h2)^2 + max(0, (info.w2 - 80) / 80)^2 ...
    + (max(0, 100 - info.shift) / 100)^2) ...
    - 0.04 * (info.meanB1 + info.meanB2) ...
    - 0.015 * (info.gap1 + info.gap2) ...
    - 0.01 * (info.h1 + info.h2) ...
    + 0.0004 * (info.w1 + info.w2);
info.energy1 = nan;
info.energy2 = nan;
end

function [fLo, fHi, meanB, meanO] = bestWindow(f, broad, other)
df = mean(diff(f));
nWin = max(2, round(200 / df));
nWin = min(nWin, numel(f));
cB = [0, cumsum(broad(:).')];
cO = [0, cumsum(other(:).')];
nOut = numel(f) - nWin + 1;
mb = (cB(1 + nWin:end) - cB(1:nOut)) / nWin;
mo = (cO(1 + nWin:end) - cO(1:nOut)) / nWin;
margin = min(mb - 0.45, (mb - mo) - 0.20);
[~, i] = max(margin);
fLo = f(i);
fHi = f(i + nWin - 1);
meanB = mb(i);
meanO = mo(i);
end

function peaks = localPeaks(a, f)
a = a(:).';
f = f(:).';
N = numel(a);
df = mean(diff(f));
peaks = zeros(0, 3);
for i = 1:N
    if i > 1 && a(i) < a(i - 1)
        continue;
    end
    if i < N && a(i) < a(i + 1)
        continue;
    end
    if i < N && a(i) == a(i + 1)
        continue;
    end
    j = i;
    while j > 1 && a(j - 1) == a(i)
        j = j - 1;
    end
    iPeak = round((j + i) / 2);
    if a(iPeak) >= 0.5
        lo = iPeak;
        hi = iPeak;
        while lo > 1 && a(lo - 1) >= 0.5
            lo = lo - 1;
        end
        while hi < N && a(hi + 1) >= 0.5
            hi = hi + 1;
        end
        width = (hi - lo + 1) * df;
    else
        width = 0;
    end
    peaks(end + 1, :) = [a(iPeak), f(iPeak), width]; %#ok<AGROW>
end
if isempty(peaks)
    [h, ix] = max(a);
    peaks = [h, f(ix), 0];
end
end

function best = choosePeakPair(p1, p2)
bestL = inf;
best = [p1(1, :), p2(1, :)];
for i = 1:size(p1, 1)
    for j = 1:size(p2, 1)
        h1 = p1(i, 1);
        f1 = p1(i, 2);
        w1 = p1(i, 3);
        h2 = p2(j, 1);
        f2 = p2(j, 2);
        w2 = p2(j, 3);
        L = max(0, 0.80 - h1)^2 + max(0, (w1 - 80) / 80)^2 ...
            + max(0, 0.80 - h2)^2 + max(0, (w2 - 80) / 80)^2 ...
            + (max(0, 100 - abs(f1 - f2)) / 100)^2 ...
            - 1e-3 * (h1 + h2);
        if L < bestL
            bestL = L;
            best = [h1, f1, w1, h2, f2, w2];
        end
    end
end
end

function X = dedupe(X, tol)
if isempty(X)
    return;
end
key = round(X / tol);
[~, ia] = unique(key, 'rows', 'stable');
X = X(ia, :);
end

function printReport(info, name)
fprintf(['%s 状态1：正向 %.0f–%.0f Hz 平均 %.4f，反向平均 %.4f，高出 %.4f；' ...
    '反向峰 %.4f @ %.0f Hz，连续超过 0.5 的宽度 %.0f Hz。'], ...
    name, info.fLo1, info.fHi1, info.meanB1, info.meanO1, info.gap1, ...
    info.h1, info.fp1, info.w1);
fprintf(' 宽带%s，窄带%s。\n', passFail(info.okBroad1), passFail(info.okNarrow1));
fprintf(['%s 状态2：反向 %.0f–%.0f Hz 平均 %.4f，正向平均 %.4f，高出 %.4f；' ...
    '正向峰 %.4f @ %.0f Hz，连续超过 0.5 的宽度 %.0f Hz。'], ...
    name, info.fLo2, info.fHi2, info.meanB2, info.meanO2, info.gap2, ...
    info.h2, info.fp2, info.w2);
fprintf(' 宽带%s，窄带%s。\n', passFail(info.okBroad2), passFail(info.okNarrow2));
fprintf('%s 窄带峰从 %.0f Hz 移到 %.0f Hz，移动 %.0f Hz，%s。\n', ...
    name, info.fp1, info.fp2, info.shift, passFail(info.okShift));
if info.allOk
    fprintf('%s：两态的宽带、窄带和峰位移动都达到标准。\n', name);
else
    fprintf('%s：还没有同时达到两态标准和 100 Hz 峰位移动。\n', name);
end
end

function s = passFail(flag)
if flag
    s = '达标';
else
    s = '未达标';
end
end

function plotTwoStates(f, af1, ab1, af2, ab2, info, figPath)
fontName = 'WenQuanYi Micro Hei';
if exist('OCTAVE_VERSION', 'builtin')
    set(0, 'defaultaxesfontname', fontName);
    set(0, 'defaulttextfontname', fontName);
end
fig = figure('Color', 'w', 'Position', [80, 40, 980, 760], 'visible', 'off');
subplot(2, 1, 1);
hold on;
shadeBand(info.fLo1, info.fHi1, [0.75, 0.85, 1]);
hF1 = plot(f, af1, 'b-', 'LineWidth', 1.8);
hB1 = plot(f, ab1, 'r--', 'LineWidth', 1.8);
plot(f([1, end]), [0.5, 0.5], 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
labelPeak(f, af1, '正向吸声', 'b', []);
labelPeak(f, ab1, '反向吸声', 'r', info.fp1);
styleAxis(f, '状态1：蓝实线=正向吸声（宽带），红虚线=反向吸声（窄带）');
legend([hF1, hB1], {'正向吸声', '反向吸声'}, 'Location', 'northeast');

subplot(2, 1, 2);
hold on;
shadeBand(info.fLo2, info.fHi2, [1, 0.86, 0.75]);
hF2 = plot(f, af2, 'b-', 'LineWidth', 1.8);
hB2 = plot(f, ab2, 'r--', 'LineWidth', 1.8);
plot(f([1, end]), [0.5, 0.5], 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
labelPeak(f, af2, '正向吸声', 'b', info.fp2);
labelPeak(f, ab2, '反向吸声', 'r', []);
styleAxis(f, '状态2：蓝实线=正向吸声（窄带），红虚线=反向吸声（宽带）');
legend([hF2, hB2], {'正向吸声', '反向吸声'}, 'Location', 'northeast');

print(fig, figPath, '-dpng', '-r140');
close(fig);
end

function shadeBand(fLo, fHi, color)
patch([fLo, fHi, fHi, fLo], [0, 0, 1.18, 1.18], color, ...
    'FaceAlpha', 0.28, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

function styleAxis(f, name)
grid on;
box on;
xlim([f(1), f(end)]);
ylim([0, 1.18]);
xlabel('Frequency (Hz)');
ylabel('Absorption');
title(name);
end

function labelPeak(f, a, label, color, forcedF)
if isempty(forcedF)
    [pk, ix] = max(a);
    x = f(ix);
else
    [~, ix] = min(abs(f - forcedF));
    x = f(ix);
    pk = a(ix);
end
if x > f(1) + 0.78 * (f(end) - f(1))
    align = 'right';
elseif x < f(1) + 0.22 * (f(end) - f(1))
    align = 'left';
else
    align = 'center';
end
text(x, min(pk + 0.035, 1.08), label, 'Color', color, ...
    'HorizontalAlignment', align, 'VerticalAlignment', 'bottom', ...
    'FontSize', 11, 'Clipping', 'on');
end

function flushOut()
if exist('OCTAVE_VERSION', 'builtin')
    fflush(stdout);
end
end
