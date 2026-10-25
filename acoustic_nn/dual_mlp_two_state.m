% 四单胞微穿孔板的双 MLP 逆向设计
%
% 物理模型沿用你给出的传递矩阵脚本：4 个单胞，每个单胞 3 块微穿孔板
% 和 2 个腔。变量是 12 个孔径和 8 个腔深，板厚固定为 2 mm，孔数为 1。
%
% 网络按 Li, Xin, Materials & Design 265 (2026) 116031 第 4.1 节：
%   正向网络（解码器）隐藏层 64, 128, 128
%   逆向网络（编码器）隐藏层 256, 256, 128
%   隐藏层 ReLU，输出层线性
%   损失 MSE，优化器 Adam，批大小 256，最多 20 轮，每轮打乱
%   训练样本 300000，划分 240000 / 30000 / 30000
%   输入按训练集做标准化
%   先单独训练正向网络并冻住，再用它作解码器训练逆向网络
%   遗传算法在逆向网络给出的初值附近搜索
%
% 和论文的差别只来自你的目标，论文正文没有这条任务：
%   论文是 12 个参数对应一条 100 点吸声谱。
%   这里是 20 个参数对应正向、反向各一条谱，频率从 100 Hz 到 700 Hz。
%   论文没有给出学习率和遗传算法的种群、代数、适应度，脚本里写明了取值。
%
% 设计结果是同一组孔径下的两套腔深。比较的是 300–500 Hz 的平均吸声系数：
%   状态 1：平均正向比平均反向高出 0.4 以上
%   状态 2：平均正向比平均反向低出 0.4 以上
% 最后的图用传递矩阵画 100–700 Hz，黄色带仍是 300–500 Hz。
% 遗传算法用冻住的正向网络计算适应度，最后用传递矩阵重算并画图。
%
% 直接在 MATLAB 里运行本文件。不需要 Deep Learning Toolbox。
% 正式计算是 30 万样本、20 轮，内存大约 1 GB，时间主要花在生成样本。
% 若只想先走通流程，在系统环境变量里设 RUN_MODE=quick 再运行。

function dual_mlp_two_state
% 运行：在 MATLAB 命令行输入 dual_mlp_two_state
% 短流程：setenv('RUN_MODE','quick'); dual_mlp_two_state

clear;
clc;
close all;

rng(1);

%% 运行规模

quick = strcmp(getenv('RUN_MODE'), 'quick');
physicsOnly = strcmp(getenv('RUN_MODE'), 'physics');

if quick
    nSamples = 512;
    maxEpochs = 2;
    gaPopulation = 8;
    gaGenerations = 3;
else
    nSamples = 300000;
    maxEpochs = 20;
    gaPopulation = 40;
    gaGenerations = 25;
end
batchSize = 256;
learnRate = 1e-3;
physicsChunk = 64;

%% 频率、空气和名义几何

fTrain = linspace(100, 700, 121);
fPlot = 100:1:700;

air = struct();
air.rho0 = 1.29;
air.c0 = 343;
air.eta = 1.89e-5;
air.kappa = 0.026;
air.Cv = 712;
air.gamma = 1.4;
air.P0 = air.rho0 * air.c0^2 / air.gamma;
air.Z0 = air.rho0 * air.c0;
air.nu = air.eta / air.rho0;
air.muBar = air.kappa / (air.rho0 * air.Cv);

geom = struct();
geom.W = 30e-3;
geom.L = 30e-3;
geom.t = 1e-3 * 2 * ones(4, 3);
geom.stinsonN = 200;

% 12 个孔径，按单胞、再按三块板排列，单位 m
% [d11 d12 d13 d21 d22 d23 d31 d32 d33 d41 d42 d43]
dNominal = 1e-3 * [1, 0.8, 4, 2, 0.8, 3, 3, 0.8, 2, 4, 0.8, 1];
hNominal = 1e-3 * 47 * ones(1, 8);

dLower = 0.1e-3 * ones(1, 12);
dUpper = 8e-3 * ones(1, 12);
hLower = 10e-3 * ones(1, 8);
hUpper = 80e-3 * ones(1, 8);

fprintf('预计算训练集频率上的 Stinson 等效参数...\n');
[kTrain, zTrain] = ductWave(fTrain, air, geom.W, geom.L, geom.stinsonN);
modelTrain = geom;
modelTrain.kEff = kTrain;
modelTrain.zEff = zTrain;
modelTrain.air = air;

[afNom, abNom, extraNom] = acousticResponse(dNominal, hNominal, fTrain, modelTrain);
fprintf('名义几何自检：平均正向吸声 %.4f，平均反向吸声 %.4f，传递矩阵 det 误差 %.3e\n', ...
    mean(afNom), mean(abNom), extraNom.detErr);
fprintf('名义几何抽样 af(1)=%.8f af(51)=%.8f ab(end)=%.8f\n', ...
    afNom(1), afNom(51), abNom(end));

if physicsOnly
    fprintf('只做物理自检，到此结束。\n');
    return;
end

if strcmp(getenv('RUN_MODE'), 'bandwidth')
    outDir = fileparts(mfilename('fullpath'));
    if isempty(outDir)
        outDir = pwd;
    end
    runBandwidthSwap(outDir, air, geom, fPlot, dLower, dUpper, hLower, hUpper);
    return;
end

checkNetworkGradient();

%% 用传递矩阵生成数据集

fprintf('生成 %d 组样本...\n', nSamples);
[X, Y] = generateDataset(nSamples, fTrain, modelTrain, ...
    dLower, dUpper, hLower, hUpper, physicsChunk);

if nSamples == 300000
    nTrain = 240000;
    nVal = 30000;
    nTest = 30000;
else
    nTrain = floor(0.8 * nSamples);
    nVal = floor(0.1 * nSamples);
    nTest = nSamples - nTrain - nVal;
end

XTrain = X(1:nTrain, :);
YTrain = Y(1:nTrain, :);
XVal = X(nTrain + (1:nVal), :);
YVal = Y(nTrain + (1:nVal), :);
XTest = X(nTrain + nVal + (1:nTest), :);
YTest = Y(nTrain + nVal + (1:nTest), :);

xMean = mean(XTrain, 1);
xStd = std(XTrain, 0, 1);
xStd(xStd < 1e-12) = 1;
yMean = mean(YTrain, 1);
yStd = std(YTrain, 0, 1);
yStd(yStd < 1e-12) = 1;

XTrainS = (XTrain - xMean) ./ xStd;
XValS = (XVal - xMean) ./ xStd;
XTestS = (XTest - xMean) ./ xStd;
YTrainS = (YTrain - yMean) ./ yStd;
YValS = (YVal - yMean) ./ yStd;
YTestS = (YTest - yMean) ./ yStd;

%% 第一阶段：训练正向网络并冻住

fprintf('\n===== 训练正向网络：20 个参数 -> 正向/反向吸声谱 =====\n');
nSpec = 2 * numel(fTrain);
forwardNet = initMLP([20, 64, 128, 128, nSpec], {'relu', 'relu', 'relu', 'linear'});
forwardNet = trainRegressor(forwardNet, XTrainS, YTrain, XValS, YVal, ...
    maxEpochs, batchSize, learnRate);

YHatTest = mlpPredict(forwardNet, XTestS);
forwardMae = mean(abs(YHatTest - YTest), 1);
nf = numel(fTrain);
fprintf('测试集 MAE  正向吸声 %.4f  反向吸声 %.4f\n', ...
    mean(forwardMae(1:nf)), mean(forwardMae(nf + 1:end)));

%% 第二阶段：冻住正向网络，训练逆向自编码器

fprintf('\n===== 训练逆向网络：吸声谱 -> 20 个参数 =====\n');
inverseNet = initMLP([nSpec, 256, 256, 128, 20], {'relu', 'relu', 'relu', 'linear'});
inverseNet = trainAutoencoder(inverseNet, forwardNet, YTrainS, YTrain, ...
    YValS, YVal, maxEpochs, batchSize, learnRate);

PTestS = mlpPredict(inverseNet, YTestS);
PTest = PTestS .* xStd + xMean;
paramDenom = max(abs(XTest), 1e-12);
paramRel = mean(abs(PTest - XTest) ./ paramDenom, 2);
fprintf('测试集参数平均相对误差 %.2f%%\n', 100 * mean(paramRel));

%% 两种目标谱，逆向网络给初值

bandMargin = 0.4;
targetForward = targetSpectrum(fTrain, 0.75, 0.30);
targetBackward = targetSpectrum(fTrain, 0.30, 0.75);
paramForward = unstandardizeParameters( ...
    mlpPredict(inverseNet, standardizeRows(targetForward, yMean, yStd)), xMean, xStd);
paramBackward = unstandardizeParameters( ...
    mlpPredict(inverseNet, standardizeRows(targetBackward, yMean, yStd)), xMean, xStd);

dInit = 0.5 * (paramForward(1:12) + paramBackward(1:12));
hInit1 = paramForward(13:20);
hInit2 = paramBackward(13:20);
xInit = [dInit, hInit1, hInit2];

lower = [dLower, hLower, hLower];
upper = [dUpper, hUpper, hUpper];
xInit = min(max(xInit, lower), upper);

%% 遗传算法：孔径共用，两套腔深分开

fprintf('\n===== 遗传算法搜索两种腔深 =====\n');
band = fTrain >= 300 & fTrain <= 500;
[xBest, bestLoss] = geneticSearch(forwardNet, xInit, lower, upper, ...
    xMean, xStd, band, bandMargin, gaPopulation, gaGenerations);
fprintf('遗传算法最优适应度 %.6f\n', bestLoss);

dBest = xBest(1:12);
hBest1 = xBest(13:20);
hBest2 = xBest(21:28);

%% 用传递矩阵在 1 Hz 网格上重算

fprintf('用传递矩阵重算 100:1:700 Hz...\n');
[kPlot, zPlot] = ductWave(fPlot, air, geom.W, geom.L, geom.stinsonN);
modelPlot = geom;
modelPlot.kEff = kPlot;
modelPlot.zEff = zPlot;
modelPlot.air = air;

[af1, ab1, extra1] = acousticResponse(dBest, hBest1, fPlot, modelPlot);
[af2, ab2, extra2] = acousticResponse(dBest, hBest2, fPlot, modelPlot);

bandPlot = fPlot >= 300 & fPlot <= 500;
gap1 = mean(af1(bandPlot)) - mean(ab1(bandPlot));
gap2 = mean(ab2(bandPlot)) - mean(af2(bandPlot));
fprintf('\n状态 1（300–500 Hz）平均正向 %.4f，平均反向 %.4f，正向高出 %.4f\n', ...
    mean(af1(bandPlot)), mean(ab1(bandPlot)), gap1);
fprintf('状态 2（300–500 Hz）平均正向 %.4f，平均反向 %.4f，反向高出 %.4f\n', ...
    mean(af2(bandPlot)), mean(ab2(bandPlot)), gap2);
fprintf('传递矩阵 det 误差：状态1 %.3e，状态2 %.3e\n', extra1.detErr, extra2.detErr);

if gap1 > bandMargin
    fprintf('状态 1：300–500 Hz 平均正向比反向高出 %.4f，大于 %.1f。\n', gap1, bandMargin);
else
    fprintf('状态 1：300–500 Hz 平均正向比反向高出 %.4f，尚未大于 %.1f。\n', gap1, bandMargin);
end
if gap2 > bandMargin
    fprintf('状态 2：300–500 Hz 平均反向比正向高出 %.4f，大于 %.1f。\n', gap2, bandMargin);
else
    fprintf('状态 2：300–500 Hz 平均反向比正向高出 %.4f，尚未大于 %.1f。\n', gap2, bandMargin);
end

fprintf('\n共用孔径 (mm)\n');
disp(reshape(dBest, 3, 4).' * 1e3);
fprintf('状态 1 腔深 (mm)\n');
disp(reshape(hBest1, 2, 4).' * 1e3);
fprintf('状态 2 腔深 (mm)\n');
disp(reshape(hBest2, 2, 4).' * 1e3);

outDir = fileparts(mfilename('fullpath'));
if isempty(outDir)
    outDir = pwd;
end
resultPath = fullfile(outDir, 'two_state_design.mat');
save(resultPath, 'fPlot', 'dBest', 'hBest1', 'hBest2', ...
    'af1', 'ab1', 'af2', 'ab2', 'extra1', 'extra2', ...
    'forwardNet', 'inverseNet', 'xMean', 'xStd', 'yMean', 'yStd', ...
    'fTrain', 'bestLoss');
fprintf('结果已保存到 %s\n', resultPath);

plotState(fPlot, af1, ab1, extra1, '(1) 300-500 Hz 正向高出 0.4');
plotState(fPlot, af2, ab2, extra2, '(2) 300-500 Hz 反向高出 0.4');

fig = figure('Color', 'w', 'Position', [120, 40, 980, 760]);
subplot(2, 1, 1);
plot(fPlot, af1, 'b-', 'LineWidth', 1.7);
hold on;
plot(fPlot, ab1, 'r--', 'LineWidth', 1.7);
markBand();
grid on;
box on;
xlabel('Frequency (Hz)');
ylabel('Absorption');
xlim([fPlot(1), fPlot(end)]);
title('(a) State 1, forward mean exceeds backward by 0.4');
legend('Forward', 'Backward', 'Location', 'best');

subplot(2, 1, 2);
plot(fPlot, af2, 'b-', 'LineWidth', 1.7);
hold on;
plot(fPlot, ab2, 'r--', 'LineWidth', 1.7);
markBand();
grid on;
box on;
xlabel('Frequency (Hz)');
ylabel('Absorption');
xlim([fPlot(1), fPlot(end)]);
title('(b) State 2, backward mean exceeds forward by 0.4');
legend('Forward', 'Backward', 'Location', 'best');

figPath = fullfile(outDir, 'two_state_absorption.png');
try
    print(fig, figPath, '-dpng', '-r140');
    fprintf('图已保存到 %s\n', figPath);
catch
    fprintf('当前环境没有把图写成文件，窗口里仍然有曲线。\n');
end
end

%% 局部函数

function [kEff, zEff] = ductWave(f, air, W, L, N)
omega = 2 * pi * reshape(f, 1, []);
nf = numel(omega);
bn = (2 * (0:N) + 1) * pi / L;
Sv = zeros(1, nf);
St = zeros(1, nf);
for m = 0:N
    am = (2 * m + 1) * pi / W;
    lambda2 = am^2 + bn.^2;
    wmn = 1 ./ (am^2 * bn.^2);
    num = wmn .* lambda2;
    Sv = Sv + sum(num.' ./ (lambda2.' + 1i * omega / air.nu), 1);
    St = St + sum(num.' ./ (lambda2.' + 1i * omega / air.muBar), 1);
end
factor = 64 / (W^2 * L^2);
Fv = factor * Sv;
Ft = factor * St;
rhoEff = air.rho0 ./ (1 - Fv);
kMod = air.gamma * air.P0 ./ (1 + (air.gamma - 1) * Ft);
kEff = omega .* sqrt(rhoEff ./ kMod);
zEff = sqrt(rhoEff .* kMod);
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
    [af, ab, ~] = acousticResponse(dTry, hTry, f, model);
    ok = all(isfinite(af), 2) & all(isfinite(ab), 2) ...
        & all(af > -0.05 & af < 1.05, 2) ...
        & all(ab > -0.05 & ab < 1.05, 2);
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

function [alphaF, alphaB, extra] = acousticResponse(dFlat, hFlat, f, model)
if size(dFlat, 1) == 1 && size(dFlat, 2) ~= 12 && numel(dFlat) == 12
    dFlat = reshape(dFlat, 1, 12);
end
if size(hFlat, 1) == 1 && size(hFlat, 2) ~= 8 && numel(hFlat) == 8
    hFlat = reshape(hFlat, 1, 8);
end
B = size(dFlat, 1);
Nf = numel(f);
omega = 2 * pi * reshape(f, 1, []);
D = diametersFromRow(dFlat);
Hv = depthsFromRow(hFlat);
phi = pi * (D .^ 2) / (4 * model.W * model.L);
shear = sqrt(model.air.eta ./ (model.air.rho0 * omega));
y = (D / 2) ./ reshape(shear, 1, 1, 1, []);
q = y * sqrt(-1i);
Fvis = 1 - 2 * besselj(1, q) ./ (q .* besselj(0, q));
phi4 = reshape(phi, 4, 3, B, 1);
D4 = reshape(D, 4, 3, B, 1);
t4 = reshape(model.t, 4, 3, 1, 1);
om4 = reshape(omega, 1, 1, 1, []);
Zp = 1i * om4 * model.air.rho0 .* t4 ./ (phi4 .* Fvis) ...
    + sqrt(2) * model.air.eta .* y ./ (phi4 .* D4) ...
    + 1i * 0.85 * om4 * model.air.rho0 .* D4 ./ phi4;

Y11 = zeros(B, Nf);
Y12 = zeros(B, Nf);
Y21 = zeros(B, Nf);
Y22 = zeros(B, Nf);
for s = 1:4
    Z1 = reshape(Zp(s, 1, :, :), B, Nf);
    Z2 = reshape(Zp(s, 2, :, :), B, Nf);
    Z3 = reshape(Zp(s, 3, :, :), B, Nf);
    h1 = reshape(Hv(s, 1, :), B, 1);
    h2 = reshape(Hv(s, 2, :), B, 1);
    T = chainTransfer(Z1, Z2, Z3, h1, h2, model.kEff, model.zEff);
    A = reshape(T(1, 1, :, :), B, Nf);
    Bm = reshape(T(1, 2, :, :), B, Nf);
    C = reshape(T(2, 1, :, :), B, Nf);
    Dm = reshape(T(2, 2, :, :), B, Nf);
    Y11 = Y11 + 0.25 * (Dm ./ Bm);
    Y12 = Y12 + 0.25 * (C - A .* Dm ./ Bm);
    Y21 = Y21 + 0.25 * (1 ./ Bm);
    Y22 = Y22 + 0.25 * (-A ./ Bm);
end

A = -Y22 ./ Y21;
Bm = 1 ./ Y21;
C = Y12 - Y11 .* Y22 ./ Y21;
Dm = Y11 ./ Y21;
Z0 = model.air.Z0;
den = A + Bm / Z0 + C * Z0 + Dm;
tau = 2 ./ den;
rf = (A + Bm / Z0 - C * Z0 - Dm) ./ den;
rb = (-A + Bm / Z0 - C * Z0 + Dm) ./ den;
Rf = abs(rf) .^ 2;
Rb = abs(rb) .^ 2;
Tr = abs(tau) .^ 2;
alphaF = 1 - Rf - Tr;
alphaB = 1 - Rb - Tr;
extra = struct();
extra.Rf = Rf;
extra.Rb = Rb;
extra.Tr = Tr;
detValue = A .* Dm - Bm .* C;
extra.detErr = max(abs(detValue(:) - 1));
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

function T = chainTransfer(Z1, Z2, Z3, h1, h2, kEff, zEff)
T = mul22(plateMatrix(Z1), cavityMatrix(h1, kEff, zEff));
T = mul22(T, plateMatrix(Z2));
T = mul22(T, cavityMatrix(h2, kEff, zEff));
T = mul22(T, plateMatrix(Z3));
end

function T = plateMatrix(Z)
[B, Nf] = size(Z);
T = zeros(2, 2, B, Nf);
T(1, 1, :, :) = 1;
T(2, 2, :, :) = 1;
T(1, 2, :, :) = reshape(Z, 1, 1, B, Nf);
end

function T = cavityMatrix(depth, kEff, zEff)
kh = depth .* reshape(kEff, 1, []);
ck = cos(kh);
sk = sin(kh);
[B, Nf] = size(kh);
zRow = reshape(zEff, 1, []);
T = zeros(2, 2, B, Nf);
T(1, 1, :, :) = reshape(ck, 1, 1, B, Nf);
T(2, 2, :, :) = reshape(ck, 1, 1, B, Nf);
T(1, 2, :, :) = reshape(1i * zRow .* sk, 1, 1, B, Nf);
T(2, 1, :, :) = reshape(1i * sk ./ zRow, 1, 1, B, Nf);
end

function C = mul22(A, B)
C = zeros(size(A));
C(1, 1, :, :) = A(1, 1, :, :) .* B(1, 1, :, :) + A(1, 2, :, :) .* B(2, 1, :, :);
C(1, 2, :, :) = A(1, 1, :, :) .* B(1, 2, :, :) + A(1, 2, :, :) .* B(2, 2, :, :);
C(2, 1, :, :) = A(2, 1, :, :) .* B(1, 1, :, :) + A(2, 2, :, :) .* B(2, 1, :, :);
C(2, 2, :, :) = A(2, 1, :, :) .* B(1, 2, :, :) + A(2, 2, :, :) .* B(2, 2, :, :);
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
cache.h0 = X;
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

function [grads, dX] = mlpBackward(net, cache, dY)
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
dX = g;
end

function adam = initAdam(net)
adam.mW = cell(size(net.W));
adam.vW = cell(size(net.W));
adam.mb = cell(size(net.b));
adam.vb = cell(size(net.b));
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
    mW = adam.mW{k} / (1 - beta1^adam.t);
    vW = adam.vW{k} / (1 - beta2^adam.t);
    mb = adam.mb{k} / (1 - beta1^adam.t);
    vb = adam.vb{k} / (1 - beta2^adam.t);
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
        xb = X(batch, :).';
        yb = Y(batch, :).';
        cache = mlpForward(net, xb);
        residual = cache.y - yb;
        total = total + sum(residual(:) .^ 2);
        seen = seen + numel(residual);
        dY = 2 * residual / numel(residual);
        grads = mlpBackward(net, cache, dY);
        [net, adam] = adamApply(net, adam, grads, learnRate);
    end
    valPred = mlpPredict(net, XVal);
    valMse = mean((valPred(:) - YVal(:)) .^ 2);
    fprintf('正向网络 epoch %02d  train_mse=%.6e  val_mse=%.6e\n', ...
        epoch, total / seen, valMse);
end
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
    for start = 1:useBatch:n
        batch = order(start:min(start + useBatch - 1, n));
        sb = SStd(batch, :).';
        yb = SRaw(batch, :).';
        invCache = mlpForward(inverseNet, sb);
        fwdCache = mlpForward(forwardNet, invCache.y);
        residual = fwdCache.y - yb;
        total = total + sum(residual(:) .^ 2);
        seen = seen + numel(residual);
        dY = 2 * residual / numel(residual);
        [~, dParam] = mlpBackward(forwardNet, fwdCache, dY);
        grads = mlpBackward(inverseNet, invCache, dParam);
        [inverseNet, adam] = adamApply(inverseNet, adam, grads, learnRate);
    end
    predStd = mlpPredict(inverseNet, SValStd);
    recon = mlpPredict(forwardNet, predStd);
    valMse = mean((recon(:) - SValRaw(:)) .^ 2);
    fprintf('逆向网络 epoch %02d  train_mse=%.6e  val_mse=%.6e\n', ...
        epoch, total / seen, valMse);
end
end

function checkNetworkGradient()
net = initMLP([3, 5, 2], {'relu', 'linear'});
X = randn(3, 4);
cache = mlpForward(net, X);
dY = randn(size(cache.y));
[grads, ~] = mlpBackward(net, cache, dY);
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
if err > 1e-6
    error('网络反向传播自检失败，误差 %.3e', err);
end
fprintf('网络反向传播自检通过，最大误差 %.3e\n', err);
end

function s = targetSpectrum(f, afPlateau, abPlateau)
slope = min(max((f - 100) / (300 - 100), 0), 1);
af = 0.2 + (afPlateau - 0.2) .* slope;
ab = 0.2 + (abPlateau - 0.2) .* slope;
s = [af, ab];
end

function Z = standardizeRows(X, mu, sigma)
Z = (X - mu) ./ sigma;
end

function X = unstandardizeParameters(Z, mu, sigma)
X = Z .* sigma + mu;
end

function [bestX, bestLoss] = geneticSearch(forwardNet, xInit, lower, upper, ...
    xMean, xStd, band, bandMargin, population, generations)
nVar = numel(xInit);
pop = lower + (upper - lower) .* rand(population, nVar);
pop(1, :) = xInit;
bestX = xInit;
bestLoss = inf;
for gen = 1:generations
    loss = zeros(population, 1);
    for i = 1:population
        loss(i) = stateFitness(forwardNet, pop(i, :), xMean, xStd, band, bandMargin);
    end
    [loss, order] = sort(loss);
    pop = pop(order, :);
    if loss(1) < bestLoss
        bestLoss = loss(1);
        bestX = pop(1, :);
    end
    fprintf('遗传算法 第 %02d 代  当前最优 %.6f\n', gen, bestLoss);
    nextPop = pop(1:min(2, population), :);
    while size(nextPop, 1) < population
        parentPool = max(2, ceil(population / 3));
        p1 = pop(randi(parentPool), :);
        p2 = pop(randi(parentPool), :);
        mix = rand;
        child = mix * p1 + (1 - mix) * p2;
        span = upper - lower;
        mutate = rand(1, nVar) < 0.2;
        child(mutate) = child(mutate) + 0.08 * span(mutate) .* randn(1, sum(mutate));
        child = min(max(child, lower), upper);
        nextPop = [nextPop; child]; %#ok<AGROW>
    end
    pop = nextPop(1:population, :);
end
end

function loss = stateFitness(forwardNet, x, xMean, xStd, band, bandMargin)
d = x(1:12);
h1 = x(13:20);
h2 = x(21:28);
y1 = predictState(forwardNet, d, h1, xMean, xStd);
y2 = predictState(forwardNet, d, h2, xMean, xStd);
nf = numel(band);
af1 = y1(1:nf);
ab1 = y1(nf + 1:end);
af2 = y2(1:nf);
ab2 = y2(nf + 1:end);
gap1 = mean(af1(band)) - mean(ab1(band));
gap2 = mean(ab2(band)) - mean(af2(band));
loss = max(0, bandMargin - gap1) ^ 2 + max(0, bandMargin - gap2) ^ 2;
end

function y = predictState(forwardNet, d, h, xMean, xStd)
x = ([d, h] - xMean) ./ xStd;
y = mlpPredict(forwardNet, x);
end

function runBandwidthSwap(outDir, air, geom, fPlot, dLower, dUpper, hLower, hUpper)
trainedPath = fullfile(outDir, 'two_state_design.mat');
if exist(trainedPath, 'file') ~= 2
    error('找不到已训练的正向网络：%s', trainedPath);
end
trained = load(trainedPath);
forwardNet = trained.forwardNet;
xMean = trained.xMean;
xStd = trained.xStd;
fTrain = trained.fTrain(:)';
fprintf('载入已训练正向网络，在 %.0f–%.0f Hz 的网络谱上搜索宽带/窄带对调。\n', ...
    fTrain(1), fTrain(end));

broadTarget = shapeTarget(fTrain, 'broad', 'narrow');
narrowTarget = shapeTarget(fTrain, 'narrow', 'broad');
paramBroad = unstandardizeParameters( ...
    mlpPredict(trained.inverseNet, standardizeRows(broadTarget, trained.yMean, trained.yStd)), ...
    xMean, xStd);
paramNarrow = unstandardizeParameters( ...
    mlpPredict(trained.inverseNet, standardizeRows(narrowTarget, trained.yMean, trained.yStd)), ...
    xMean, xStd);
dInit = 0.5 * (paramBroad(1:12) + paramNarrow(1:12));
xInit = [dInit, paramBroad(13:20), paramNarrow(13:20)];
lower = [dLower, hLower, hLower];
upper = [dUpper, hUpper, hUpper];
xInit = min(max(xInit, lower), upper);
if isfield(trained, 'dBest')
    xPrev = [trained.dBest(:)', trained.hBest1(:)', trained.hBest2(:)'];
    xPrev = min(max(xPrev, lower), upper);
else
    xPrev = xInit;
end

fprintf('\n===== 正向网络搜索：状态1 正向宽带/反向窄带，状态2 对调 =====\n');
[xBest, bestLoss] = shapeSearch(forwardNet, xInit, xPrev, lower, upper, ...
    xMean, xStd, fTrain, 80, 50);
fprintf('宽带/窄带适应度 %.6f\n', bestLoss);

dBest = xBest(1:12);
hBest1 = xBest(13:20);
hBest2 = xBest(21:28);
y1 = predictState(forwardNet, dBest, hBest1, xMean, xStd);
y2 = predictState(forwardNet, dBest, hBest2, xMean, xStd);
nf = numel(fTrain);
reportShape('网络 状态1', fTrain, y1(1:nf), y1(nf + 1:end));
reportShape('网络 状态2', fTrain, y2(1:nf), y2(nf + 1:end));

fprintf('用传递矩阵重算 100:1:700 Hz...\n');
[kPlot, zPlot] = ductWave(fPlot, air, geom.W, geom.L, geom.stinsonN);
modelPlot = geom;
modelPlot.kEff = kPlot;
modelPlot.zEff = zPlot;
modelPlot.air = air;
[af1, ab1, extra1] = acousticResponse(dBest, hBest1, fPlot, modelPlot);
[af2, ab2, extra2] = acousticResponse(dBest, hBest2, fPlot, modelPlot);
reportShape('传递矩阵 状态1', fPlot, af1, ab1);
reportShape('传递矩阵 状态2', fPlot, af2, ab2);
fprintf('传递矩阵 det 误差：状态1 %.3e，状态2 %.3e\n', extra1.detErr, extra2.detErr);

fprintf('\n共用孔径 (mm)\n');
disp(reshape(dBest, 3, 4).' * 1e3);
fprintf('状态 1 腔深 (mm)，正向宽带、反向窄带\n');
disp(reshape(hBest1, 2, 4).' * 1e3);
fprintf('状态 2 腔深 (mm)，正向窄带、反向宽带\n');
disp(reshape(hBest2, 2, 4).' * 1e3);

fig = figure('Color', 'w', 'Position', [80, 40, 980, 760]);
subplot(2, 1, 1);
plot(fPlot, af1, 'b-', 'LineWidth', 1.7);
hold on;
plot(fPlot, ab1, 'r--', 'LineWidth', 1.7);
plot(fPlot([1, end]), [0.5, 0.5], 'k:', 'LineWidth', 1.0);
grid on;
box on;
xlim([fPlot(1), fPlot(end)]);
xlabel('Frequency (Hz)');
ylabel('Absorption');
title('State 1: forward broadband, backward narrowband');
legend('Forward', 'Backward', '0.5 level', 'Location', 'northeast');

subplot(2, 1, 2);
plot(fPlot, af2, 'b-', 'LineWidth', 1.7);
hold on;
plot(fPlot, ab2, 'r--', 'LineWidth', 1.7);
plot(fPlot([1, end]), [0.5, 0.5], 'k:', 'LineWidth', 1.0);
grid on;
box on;
xlim([fPlot(1), fPlot(end)]);
xlabel('Frequency (Hz)');
ylabel('Absorption');
title('State 2: forward narrowband, backward broadband');
legend('Forward', 'Backward', '0.5 level', 'Location', 'northeast');

figPath = fullfile(outDir, 'two_state_bandwidth.png');
resultPath = fullfile(outDir, 'two_state_bandwidth.mat');
save(resultPath, 'fPlot', 'fTrain', 'dBest', 'hBest1', 'hBest2', ...
    'af1', 'ab1', 'af2', 'ab2', 'extra1', 'extra2', 'bestLoss');
print(fig, figPath, '-dpng', '-r140');
fprintf('图已保存到 %s\n', figPath);
fprintf('参数已保存到 %s\n', resultPath);
end

function s = shapeTarget(f, forwardKind, backwardKind)
s = [shapeCurve(f, forwardKind), shapeCurve(f, backwardKind)];
end

function a = shapeCurve(f, kind)
if strcmp(kind, 'broad')
    a = 0.72 * double(f >= 160 & f <= 460);
else
    a = 0.92 * exp(-0.5 * ((f - 340) / 16) .^ 2);
end
end

function [bestX, bestLoss] = shapeSearch(forwardNet, xInit, xPrev, lower, upper, ...
    xMean, xStd, fTrain, population, generations)
nVar = numel(xInit);
pop = lower + (upper - lower) .* rand(population, nVar);
pop(1, :) = xInit;
pop(2, :) = xPrev;
bestX = xInit;
bestLoss = inf;
for gen = 1:generations
    loss = zeros(population, 1);
    for i = 1:population
        loss(i) = shapeFitness(forwardNet, pop(i, :), xMean, xStd, fTrain);
    end
    [loss, order] = sort(loss);
    pop = pop(order, :);
    if loss(1) < bestLoss
        bestLoss = loss(1);
        bestX = pop(1, :);
    end
    fprintf('宽带搜索 第 %02d 代  当前最优 %.6f\n', gen, bestLoss);
    nextPop = pop(1:min(2, population), :);
    while size(nextPop, 1) < population
        parentPool = max(2, ceil(population / 3));
        p1 = pop(randi(parentPool), :);
        p2 = pop(randi(parentPool), :);
        mix = rand;
        child = mix * p1 + (1 - mix) * p2;
        span = upper - lower;
        mutate = rand(1, nVar) < 0.25;
        child(mutate) = child(mutate) + 0.10 * span(mutate) .* randn(1, sum(mutate));
        child = min(max(child, lower), upper);
        nextPop = [nextPop; child]; %#ok<AGROW>
    end
    pop = nextPop(1:population, :);
end
end

function loss = shapeFitness(forwardNet, x, xMean, xStd, f)
d = x(1:12);
y1 = predictState(forwardNet, d, x(13:20), xMean, xStd);
y2 = predictState(forwardNet, d, x(21:28), xMean, xStd);
nf = numel(f);
loss = shapeGap(f, y1(1:nf), y1(nf + 1:end), true) ...
    + shapeGap(f, y2(1:nf), y2(nf + 1:end), false);
end

function loss = shapeGap(f, af, ab, forwardIsBroad)
[wideF, peakF] = absorptionWidth(f, af);
[wideB, peakB] = absorptionWidth(f, ab);
if forwardIsBroad
    wideBroad = wideF;
    wideNarrow = wideB;
    peakNarrow = peakB;
    peakBroad = peakF;
else
    wideBroad = wideB;
    wideNarrow = wideF;
    peakNarrow = peakF;
    peakBroad = peakB;
end
broadGoal = 180;
narrowGoal = 40;
loss = (max(0, broadGoal - wideBroad) / broadGoal) ^ 2 ...
    + (max(0, wideNarrow - narrowGoal) / narrowGoal) ^ 2 ...
    + max(0, 0.75 - peakNarrow) ^ 2 ...
    + max(0, 0.60 - peakBroad) ^ 2 ...
    + (max(0, 3 * wideNarrow - wideBroad) / broadGoal) ^ 2;
end

function [widthHz, peak] = absorptionWidth(f, a)
df = mean(diff(f));
above = a >= 0.5;
best = 0;
run = 0;
for i = 1:numel(above)
    if above(i)
        run = run + 1;
        if run > best
            best = run;
        end
    else
        run = 0;
    end
end
widthHz = best * df;
peak = max(a);
end

function reportShape(name, f, af, ab)
[wideF, peakF] = absorptionWidth(f, af);
[wideB, peakB] = absorptionWidth(f, ab);
[~, iF] = max(af);
[~, iB] = max(ab);
fprintf('%s：正向带宽 %.0f Hz（峰 %.2f @ %.0f Hz），反向带宽 %.0f Hz（峰 %.2f @ %.0f Hz）\n', ...
    name, wideF, peakF, f(iF), wideB, peakB, f(iB));
end

function plotState(f, af, ab, extra, name)
figure('Color', 'w', 'Position', [150, 60, 1050, 900]);
subplot(2, 1, 1);
plot(f, af, 'b-', 'LineWidth', 1.7);
hold on;
plot(f, extra.Rf, 'r--', 'LineWidth', 1.7);
plot(f, extra.Tr, 'k:', 'LineWidth', 1.7);
markBand();
xlim([f(1), f(end)]);
title(['(a) Forward incidence  ' name]);
ylabel('Coefficient');
legend('Absorption', 'Reflection', 'Transmission', 'Location', 'best');
grid on;
box on;

subplot(2, 1, 2);
plot(f, ab, 'b-', 'LineWidth', 1.7);
hold on;
plot(f, extra.Rb, 'r--', 'LineWidth', 1.7);
plot(f, extra.Tr, 'k:', 'LineWidth', 1.7);
markBand();
xlim([f(1), f(end)]);
title(['(b) Backward incidence  ' name]);
xlabel('Frequency (Hz)');
ylabel('Coefficient');
legend('Absorption', 'Reflection', 'Transmission', 'Location', 'best');
grid on;
box on;
end

function markBand()
yl = ylim;
patch([300, 500, 500, 300], [yl(1), yl(1), yl(2), yl(2)], ...
    [1, 0.93, 0.75], 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');
children = get(gca, 'Children');
set(gca, 'Children', [children(2:end); children(1)]);
ylim(yl);
end
