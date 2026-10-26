function origami_dual_mlp
% 四单胞折纸腔的双 MLP
%
% 物理和 origami_four_cell.m 相同：每个单胞三块微穿孔板、两个折纸腔。
% 折纸腔先按体积公式得到等效高度，传递矩阵用等效高度。
% 每个腔的实际高度搜索范围是 0.5–54 mm。孔径 0.1–8 mm。
%
% 两种状态共用 12 个孔径，只改 8 个实际腔高。
% 状态 1：正向在某段约 200 Hz 上平均吸声更高，反向是窄峰。
% 状态 2：反过来。窄峰要移动 100 Hz 以上。
%
% 运行：origami_dual_mlp
% 短流程：setenv('RUN_MODE','quick'); origami_dual_mlp

clc;
close all;
rng(7);

quick = strcmp(getenv('RUN_MODE'), 'quick');
if quick
    nSamples = 2048;
    maxEpochs = 4;
    nRestart = 2;
    popSize = 24;
    nGen = 8;
else
    nSamples = 60000;
    maxEpochs = 16;
    nRestart = 6;
    popSize = 48;
    nGen = 25;
end

fTrain = linspace(100, 600, 101);
fPlot = 100:1:600;
air = airProps();
geom = geomProps();
model = geom;
model.air = air;
[model.kEff, model.zEff] = ductStinson(fTrain, air, geom.W, geom.L, geom.stinsonN);

dNom = 1e-3 * repmat([2.75, 1.73, 0.42], 1, 4);
hNom = 1e-3 * 47 * ones(1, 8);
[af0, ab0] = origamiResponse(dNom, hNom, fTrain, model);
fprintf('名义点 100–600 Hz 平均正向 %.4f，平均反向 %.4f\n', mean(af0), mean(ab0));
Heq = equivalentHeight(1e-3 * 47 * ones(4, 2));
fprintf('H = 47 mm 时等效高度 %.4f mm\n', Heq(1, 1) * 1e3);

dLower = 0.1e-3 * ones(1, 12);
dUpper = 8e-3 * ones(1, 12);
hLower = 0.5e-3 * ones(1, 8);
hUpper = 54e-3 * ones(1, 8);

fprintf('生成 %d 组样本...\n', nSamples);
[X, Y] = generateDataset(nSamples, fTrain, model, dLower, dUpper, hLower, hUpper, 256);
nTrain = floor(0.8 * nSamples);
nVal = floor(0.1 * nSamples);
nTest = nSamples - nTrain - nVal;
XTrain = X(1:nTrain, :);
YTrain = Y(1:nTrain, :);
XVal = X(nTrain + (1:nVal), :);
YVal = Y(nTrain + (1:nVal), :);
XTest = X(nTrain + nVal + (1:nTest), :);
YTest = Y(nTrain + nVal + (1:nTest), :);

xMean = mean(XTrain, 1);
xStd = std(XTrain, 0, 1);
xStd(xStd < 1e-12) = 1;
XTrainS = (XTrain - xMean) ./ xStd;
XValS = (XVal - xMean) ./ xStd;
XTestS = (XTest - xMean) ./ xStd;

nSpec = 2 * numel(fTrain);
fprintf('训练正向网络...\n');
forwardNet = initMLP([20, 64, 128, 128, nSpec], {'relu', 'relu', 'relu', 'linear'});
forwardNet = trainRegressor(forwardNet, XTrainS, YTrain, XValS, YVal, maxEpochs, 256, 1e-3);
YHat = mlpPredict(forwardNet, XTestS);
nf = numel(fTrain);
maeF = mean(abs(YHat(:, 1:nf) - YTest(:, 1:nf)), 'all');
maeB = mean(abs(YHat(:, nf + 1:end) - YTest(:, nf + 1:end)), 'all');
fprintf('测试集 MAE  正向 %.4f  反向 %.4f\n', maeF, maeB);

[kPlot, zPlot] = ductStinson(fPlot, air, geom.W, geom.L, geom.stinsonN);
modelPlot = model;
modelPlot.kEff = kPlot;
modelPlot.zEff = zPlot;

fprintf('用正向网络搜索两种腔高，再用传递矩阵挑选...\n');
bestPhys = inf;
xBest = [];
af1 = [];
ab1 = [];
af2 = [];
ab2 = [];
for iRestart = 1:nRestart
    rng(2000 + iRestart);
    xTry = shapeSearch(forwardNet, xMean, xStd, fTrain, ...
        dLower, dUpper, hLower, hUpper, popSize, nGen);
    [afTry1, abTry1] = origamiResponse(xTry(1:12), xTry(13:20), fPlot, modelPlot);
    [afTry2, abTry2] = origamiResponse(xTry(1:12), xTry(21:28), fPlot, modelPlot);
    phys = curveLoss(fPlot, afTry1, abTry1, afTry2, abTry2);
    fprintf('第 %d 组  传递矩阵适应度 %.4f\n', iRestart, phys);
    if phys < bestPhys
        bestPhys = phys;
        xBest = xTry;
        af1 = afTry1;
        ab1 = abTry1;
        af2 = afTry2;
        ab2 = abTry2;
    end
end

dBest = xBest(1:12);
hBest1 = xBest(13:20);
hBest2 = xBest(21:28);
reportShape('传递矩阵 状态1', fPlot, af1, ab1, true);
reportShape('传递矩阵 状态2', fPlot, af2, ab2, false);
[~, ~, fN1] = absorptionWidth(fPlot, ab1);
[~, ~, fN2] = absorptionWidth(fPlot, af2);
fprintf('窄带峰从 %.0f Hz 移到 %.0f Hz，移动 %.0f Hz。\n', fN1, fN2, abs(fN2 - fN1));
fprintf('传递矩阵适应度 %.6f\n', bestPhys);

fprintf('\n共用孔径 (mm)\n');
disp(reshape(dBest, 3, 4).' * 1e3);
fprintf('状态 1 实际腔高 (mm)\n');
disp(reshape(hBest1, 2, 4).' * 1e3);
fprintf('状态 2 实际腔高 (mm)\n');
disp(reshape(hBest2, 2, 4).' * 1e3);
fprintf('状态 1 等效腔高 (mm)\n');
disp(equivalentHeight(reshape(hBest1, 2, 4).') * 1e3);
fprintf('状态 2 等效腔高 (mm)\n');
disp(equivalentHeight(reshape(hBest2, 2, 4).') * 1e3);

[~, ~, fLo1, fHi1] = bestWindow(fPlot, af1, ab1);
[~, ~, fLo2, fHi2] = bestWindow(fPlot, ab2, af2);
fig = figure('Color', 'w', 'Position', [80, 40, 980, 760]);
if exist('OCTAVE_VERSION', 'builtin')
    set(fig, 'defaultaxesfontname', 'WenQuanYi Micro Hei');
    set(fig, 'defaulttextfontname', 'WenQuanYi Micro Hei');
end
subplot(2, 1, 1);
hold on;
patch([fLo1, fHi1, fHi1, fLo1], [0, 0, 1, 1], [0.75, 0.85, 1], ...
    'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
hF = plot(fPlot, af1, 'b-', 'LineWidth', 1.8);
hB = plot(fPlot, ab1, 'r--', 'LineWidth', 1.8);
hL = plot(fPlot([1, end]), [0.5, 0.5], 'k:', 'LineWidth', 1.0);
markPeak(fPlot, af1, '正向', 'b');
markPeak(fPlot, ab1, '反向', 'r');
grid on; box on;
xlim([100, 600]); ylim([0, 1.05]);
xlabel('Frequency (Hz)'); ylabel('Absorption');
title('状态1：蓝实线=正向吸声（宽带），红虚线=反向吸声（窄带）');
legend([hF, hB, hL], {'正向吸声', '反向吸声', '0.5'}, 'Location', 'northeast');

subplot(2, 1, 2);
hold on;
patch([fLo2, fHi2, fHi2, fLo2], [0, 0, 1, 1], [1, 0.85, 0.75], ...
    'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
hF = plot(fPlot, af2, 'b-', 'LineWidth', 1.8);
hB = plot(fPlot, ab2, 'r--', 'LineWidth', 1.8);
hL = plot(fPlot([1, end]), [0.5, 0.5], 'k:', 'LineWidth', 1.0);
markPeak(fPlot, af2, '正向', 'b');
markPeak(fPlot, ab2, '反向', 'r');
grid on; box on;
xlim([100, 600]); ylim([0, 1.05]);
xlabel('Frequency (Hz)'); ylabel('Absorption');
title('状态2：蓝实线=正向吸声（窄带），红虚线=反向吸声（宽带）');
legend([hF, hB, hL], {'正向吸声', '反向吸声', '0.5'}, 'Location', 'northeast');

outDir = fileparts(mfilename('fullpath'));
if isempty(outDir)
    outDir = pwd;
end
figPath = fullfile(outDir, 'origami_two_state.png');
matPath = fullfile(outDir, 'origami_two_state.mat');
print(fig, figPath, '-dpng', '-r140');
save(matPath, 'fPlot', 'fTrain', 'dBest', 'hBest1', 'hBest2', ...
    'af1', 'ab1', 'af2', 'ab2', 'bestPhys', 'forwardNet', 'xMean', 'xStd', 'maeF', 'maeB');
fprintf('图已保存到 %s\n', figPath);
end

function air = airProps()
air.rho0 = 1.29;
air.c0 = 343;
air.eta = 1.89e-5;
air.kappa = 0.026;
air.Cv = 712;
air.gamma = 1.4;
air.P0 = air.rho0 * air.c0 ^ 2 / air.gamma;
air.Z0 = air.rho0 * air.c0;
air.nu = air.eta / air.rho0;
air.muBar = air.kappa / (air.rho0 * air.Cv);
end

function geom = geomProps()
geom.W = 30e-3;
geom.L = 30e-3;
geom.t = 2e-3;
geom.a = 30;
geom.hc = 20;
geom.n0 = 3;
geom.beta = 45;
geom.stinsonN = 200;
end

function Heq = equivalentHeight(H)
a = 30;
hc = 20;
n0 = 3;
beta = 45;
Hmm = H * 1e3;
h = Hmm / n0;
q = sqrt(max(hc ^ 2 - h .^ 2, 0));
tb = tand(beta);
V1 = a .* h .* q / 16;
V2 = h .* (a - hc) .* q / (16 * tb);
V3 = h .* q / 24 .* (hc / tb - q / 2);
V = 8 * n0 .* (a ^ 2 .* h / 8 - V1 - V2 - V3);
Heq = V / (30e-3 * 30e-3 * 1e6) * 1e-3;
end

function [kEff, zEff] = ductStinson(f, air, W, L, N)
omega = 2 * pi * reshape(f, 1, []);
nf = numel(omega);
bn = (2 * (0:N) + 1) * pi / L;
Sv = zeros(1, nf);
St = zeros(1, nf);
for m = 0:N
    am = (2 * m + 1) * pi / W;
    lambda2 = am ^ 2 + bn .^ 2;
    wmn = 1 ./ (am ^ 2 * bn .^ 2);
    num = wmn .* lambda2;
    Sv = Sv + sum(num.' ./ (lambda2.' + 1i * omega / air.nu), 1);
    St = St + sum(num.' ./ (lambda2.' + 1i * omega / air.muBar), 1);
end
factor = 64 / (W ^ 2 * L ^ 2);
Fv = factor * Sv;
Ft = factor * St;
rhoEff = air.rho0 ./ (1 - Fv);
kMod = air.gamma * air.P0 ./ (1 + (air.gamma - 1) * Ft);
kEff = omega .* sqrt(rhoEff ./ kMod);
zEff = sqrt(rhoEff .* kMod);
end

function [alphaF, alphaB] = origamiResponse(dFlat, hFlat, f, model)
if size(dFlat, 1) == 1 && numel(dFlat) == 12
    dFlat = reshape(dFlat, 1, 12);
end
if size(hFlat, 1) == 1 && numel(hFlat) == 8
    hFlat = reshape(hFlat, 1, 8);
end
B = size(dFlat, 1);
omega = 2 * pi * reshape(f, 1, []);
Nf = numel(omega);
D = diametersFromRow(dFlat);
Hv = equivalentHeight(depthsFromRow(hFlat));
phi = pi * (D .^ 2) / (4 * model.W * model.L);
air = model.air;
Zp = zeros(4, 3, B, Nf);
for s = 1:4
    for j = 1:3
        d = reshape(D(s, j, :), B, 1);
        ph = reshape(phi(s, j, :), B, 1);
        y = (d / 2) ./ sqrt(air.eta ./ (air.rho0 * omega));
        qv = y * sqrt(-1i);
        F = 1 - 2 * besselj(1, qv) ./ (qv .* besselj(0, qv));
        Z = 1i * air.rho0 * model.t * omega ./ (ph .* F) ...
            + sqrt(2) * air.eta * y ./ (ph .* d) ...
            + 1i * 0.85 * air.rho0 * d .* omega ./ ph;
        Zp(s, j, :, :) = reshape(Z, 1, 1, B, Nf);
    end
end

Y11 = zeros(B, Nf);
Y12 = zeros(B, Nf);
Y21 = zeros(B, Nf);
Y22 = zeros(B, Nf);
kRow = reshape(model.kEff, 1, []);
zRow = reshape(model.zEff, 1, []);
for s = 1:4
    Z1 = reshape(Zp(s, 1, :, :), B, Nf);
    Z2 = reshape(Zp(s, 2, :, :), B, Nf);
    Z3 = reshape(Zp(s, 3, :, :), B, Nf);
    H1 = reshape(Hv(s, 1, :), B, 1);
    H2 = reshape(Hv(s, 2, :), B, 1);
    [A, Bm, C, Dm] = cellTransfer(Z1, Z2, Z3, H1, H2, kRow, zRow);
    Y11 = Y11 + 0.25 * (Dm ./ Bm);
    Y12 = Y12 + 0.25 * (C - A .* Dm ./ Bm);
    Y21 = Y21 + 0.25 * (1 ./ Bm);
    Y22 = Y22 + 0.25 * (-A ./ Bm);
end
A = -Y22 ./ Y21;
Bm = 1 ./ Y21;
C = Y12 - Y11 .* Y22 ./ Y21;
Dm = Y11 ./ Y21;
Z0 = air.Z0;
den = A + Bm / Z0 + C * Z0 + Dm;
tau = 2 ./ den;
rf = (A + Bm / Z0 - C * Z0 - Dm) ./ den;
rb = (-A + Bm / Z0 - C * Z0 + Dm) ./ den;
alphaF = 1 - abs(rf) .^ 2 - abs(tau) .^ 2;
alphaB = 1 - abs(rb) .^ 2 - abs(tau) .^ 2;
end

function [A, Bm, C, Dm] = cellTransfer(Z1, Z2, Z3, H1, H2, kRow, zRow)
M = plate(Z1);
M = mulCell(M, cavity(H1, kRow, zRow));
M = mulCell(M, plate(Z2));
M = mulCell(M, cavity(H2, kRow, zRow));
M = mulCell(M, plate(Z3));
A = M{1};
Bm = M{2};
C = M{3};
Dm = M{4};
end

function M = plate(Z)
M = {ones(size(Z)), Z, zeros(size(Z)), ones(size(Z))};
end

function M = cavity(H, kRow, zRow)
kh = H .* kRow;
ck = cos(kh);
sk = sin(kh);
M = {ck, 1i * zRow .* sk, 1i * sk ./ zRow, ck};
end

function M = mulCell(A, B)
A11 = A{1}; A12 = A{2}; A21 = A{3}; A22 = A{4};
B11 = B{1}; B12 = B{2}; B21 = B{3}; B22 = B{4};
M = { ...
    A11 .* B11 + A12 .* B21, ...
    A11 .* B12 + A12 .* B22, ...
    A21 .* B11 + A22 .* B21, ...
    A21 .* B12 + A22 .* B22};
end

function D = diametersFromRow(v)
B = size(v, 1);
tmp = reshape(v.', 3, 4, B);
D = permute(tmp, [2, 1, 3]);
end

function H = depthsFromRow(v)
B = size(v, 1);
tmp = reshape(v.', 2, 4, B);
H = permute(tmp, [2, 1, 3]);
end

function [X, Y] = generateDataset(nSamples, f, model, dLower, dUpper, hLower, hUpper, chunk)
nf = numel(f);
X = zeros(nSamples, 20);
Y = zeros(nSamples, 2 * nf);
filled = 0;
while filled < nSamples
    draw = min(chunk, nSamples - filled);
    dTry = dLower + (dUpper - dLower) .* rand(draw, 12);
    hTry = hLower + (hUpper - hLower) .* rand(draw, 8);
    [af, ab] = origamiResponse(dTry, hTry, f, model);
    ok = all(isfinite(af), 2) & all(isfinite(ab), 2) ...
        & all(af > -0.05 & af < 1.05, 2) & all(ab > -0.05 & ab < 1.05, 2);
    keep = find(ok);
    take = min(numel(keep), nSamples - filled);
    if take == 0
        error('这一批样本的吸声系数都不是有限值。');
    end
    rows = filled + (1:take);
    X(rows, 1:12) = dTry(keep(1:take), :);
    X(rows, 13:20) = hTry(keep(1:take), :);
    Y(rows, 1:nf) = af(keep(1:take), :);
    Y(rows, nf + (1:nf)) = ab(keep(1:take), :);
    filled = filled + take;
    fprintf('  已接受 %d / %d\n', filled, nSamples);
end
end

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
Z = mlpForward(net, X.');
Y = Z.y.';
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
    adam.vW{k} = beta2 * adam.vW{k} + (1 - beta2) * (grads.W{k} .^ 2);
    adam.mb{k} = beta1 * adam.mb{k} + (1 - beta1) * grads.b{k};
    adam.vb{k} = beta2 * adam.vb{k} + (1 - beta2) * (grads.b{k} .^ 2);
    mW = adam.mW{k} / (1 - beta1 ^ adam.t);
    vW = adam.vW{k} / (1 - beta2 ^ adam.t);
    mb = adam.mb{k} / (1 - beta1 ^ adam.t);
    vb = adam.vb{k} / (1 - beta2 ^ adam.t);
    net.W{k} = net.W{k} - learnRate * mW ./ (sqrt(vW) + epsilon);
    net.b{k} = net.b{k} - learnRate * mb ./ (sqrt(vb) + epsilon);
end
end

function net = trainRegressor(net, X, Y, XVal, YVal, epochs, batchSize, learnRate)
adam = initAdam(net);
n = size(X, 1);
useBatch = min(batchSize, n);
for epoch = 1:epochs
    order = randperm(n);
    total = 0;
    seen = 0;
    for start = 1:useBatch:n
        batch = order(start:min(start + useBatch - 1, n));
        cache = mlpForward(net, X(batch, :).');
        residual = cache.y - Y(batch, :).';
        total = total + sum(residual(:) .^ 2);
        seen = seen + numel(residual);
        grads = mlpBackward(net, cache, 2 * residual / numel(residual));
        [net, adam] = adamApply(net, adam, grads, learnRate);
    end
    valPred = mlpPredict(net, XVal);
    valMse = mean((valPred(:) - YVal(:)) .^ 2);
    fprintf('正向网络 epoch %02d  train_mse=%.6e  val_mse=%.6e\n', epoch, total / seen, valMse);
end
end

function xBest = shapeSearch(net, xMean, xStd, f, dLower, dUpper, hLower, hUpper, population, generations)
lower = [dLower, hLower, hLower];
upper = [dUpper, hUpper, hUpper];
nVar = numel(lower);
pop = lower + (upper - lower) .* rand(population, nVar);
bestX = pop(1, :);
bestLoss = inf;
for gen = 1:generations
    loss = zeros(population, 1);
    for i = 1:population
        loss(i) = netLoss(net, pop(i, :), xMean, xStd, f);
    end
    [loss, order] = sort(loss);
    pop = pop(order, :);
    if loss(1) < bestLoss
        bestLoss = loss(1);
        bestX = pop(1, :);
    end
    nextPop = pop(1:min(2, population), :);
    while size(nextPop, 1) < population
        pool = max(2, ceil(population / 3));
        p1 = pop(randi(pool), :);
        p2 = pop(randi(pool), :);
        child = rand * p1 + (1 - rand) * p2;
        span = upper - lower;
        mutate = rand(1, nVar) < 0.25;
        child(mutate) = child(mutate) + 0.10 * span(mutate) .* randn(1, sum(mutate));
        child = min(max(child, lower), upper);
        nextPop = [nextPop; child]; %#ok<AGROW>
    end
    pop = nextPop(1:population, :);
end
xBest = bestX;
end

function loss = netLoss(net, x, xMean, xStd, f)
y1 = predictState(net, x(1:12), x(13:20), xMean, xStd);
y2 = predictState(net, x(1:12), x(21:28), xMean, xStd);
nf = numel(f);
loss = curveLoss(f, y1(1:nf), y1(nf + 1:end), y2(1:nf), y2(nf + 1:end));
end

function y = predictState(net, d, h, xMean, xStd)
y = mlpPredict(net, ([d, h] - xMean) ./ xStd);
end

function loss = curveLoss(f, af1, ab1, af2, ab2)
[loss1, f1] = shapeGap(f, af1, ab1, true);
[loss2, f2] = shapeGap(f, af2, ab2, false);
loss = loss1 + loss2 + (max(0, 100 - abs(f1 - f2)) / 100) ^ 2;
end

function [loss, fNarrow] = shapeGap(f, af, ab, forwardIsBroad)
if forwardIsBroad
    aBroad = af;
    aNarrow = ab;
else
    aBroad = ab;
    aNarrow = af;
end
[meanBroad, meanOther] = bestWindow(f, aBroad, aNarrow);
[widthNarrow, peakNarrow, fNarrow] = absorptionWidth(f, aNarrow);
loss = max(0, 0.20 - (meanBroad - meanOther)) ^ 2 ...
    + max(0, 0.45 - meanBroad) ^ 2 ...
    + (max(0, widthNarrow - 80) / 80) ^ 2 ...
    + max(0, 0.80 - peakNarrow) ^ 2;
end

function [meanBroad, meanOther, fLo, fHi] = bestWindow(f, aBroad, aOther)
df = mean(diff(f));
nWin = min(numel(f), max(2, round(200 / df)));
best = -inf;
meanBroad = 0;
meanOther = 0;
iBest = 1;
for i = 1:(numel(f) - nWin + 1)
    mb = mean(aBroad(i:i + nWin - 1));
    mo = mean(aOther(i:i + nWin - 1));
    if mb - mo > best
        best = mb - mo;
        meanBroad = mb;
        meanOther = mo;
        iBest = i;
    end
end
fLo = f(iBest);
fHi = f(iBest + nWin - 1);
end

function [widthHz, peak, fPeak] = absorptionWidth(f, a)
df = mean(diff(f));
above = a >= 0.5;
best = 0;
run = 0;
for i = 1:numel(above)
    if above(i)
        run = run + 1;
        best = max(best, run);
    else
        run = 0;
    end
end
widthHz = best * df;
[peak, iPeak] = max(a);
fPeak = f(iPeak);
end

function reportShape(name, f, af, ab, forwardIsBroad)
if forwardIsBroad
    aBroad = af; aNarrow = ab; broadName = '正向'; narrowName = '反向';
else
    aBroad = ab; aNarrow = af; broadName = '反向'; narrowName = '正向';
end
[meanBroad, meanOther, fLo, fHi] = bestWindow(f, aBroad, aNarrow);
[widthNarrow, peakNarrow, fNarrow] = absorptionWidth(f, aNarrow);
fprintf(['%s：%s 在 %.0f–%.0f Hz 平均 %.3f，比另一方向高 %.3f；' ...
    '%s 峰 %.2f @ %.0f Hz，连续超过 0.5 的宽度 %.0f Hz\n'], ...
    name, broadName, fLo, fHi, meanBroad, meanBroad - meanOther, ...
    narrowName, peakNarrow, fNarrow, widthNarrow);
end

function markPeak(f, a, label, color)
[pk, ix] = max(a);
text(f(ix), min(pk + 0.04, 1.02), label, 'Color', color, ...
    'HorizontalAlignment', 'center', 'FontSize', 11);
end
