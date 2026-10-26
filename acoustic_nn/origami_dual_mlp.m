% 四单胞折纸腔吸声结构的正向 MLP 与两状态搜索
%
% 物理和 origami_four_cell.m 相同。四个单胞并联，每个单胞是
% 微穿孔板、折纸腔、微穿孔板、折纸腔、微穿孔板。
% 折纸腔先按体积公式换成等效高度，腔体传递矩阵用 Stinson 等效密度和体积模量。
% 折纸几何固定：a = 30 mm，hc = 20 mm，n0 = 4，beta = 45°。
% 实际腔高取 10–80 mm，80 mm 等于 n0*hc。孔径仍取 0.1–8 mm。
%
% 正向网络：20 -> 64 -> 128 -> 128 -> 2*Nf
% 20 个输入是 12 个孔径和 8 个实际腔高，输出是这一几何的正向、反向吸声谱。
% 隐藏层 ReLU，输出线性，Glorot 均匀初始化，Adam，学习率 1e-3，损失是加权 MSE。
%
% 两种状态共用 12 个孔径。状态 2 只改 8 个实际腔高。
% 考核频带固定为 300–500 Hz：
%   状态 1：正向平均吸声 − 反向平均吸声 > 0.4
%   状态 2：反向平均吸声 − 正向平均吸声 > 0.4
% 网络在训练频率上搜索，候选再用 300:1:500 Hz 的传递矩阵打分并局部修正。
% 图和曲线用 100:1:700 Hz。
%
% 直接运行本文件。不需要 Deep Learning Toolbox。
% 短流程：setenv('RUN_MODE','quick'); origami_dual_mlp
% 只做物理核对：setenv('RUN_MODE','verify'); origami_dual_mlp

function origami_dual_mlp

if exist('OCTAVE_VERSION', 'builtin')
    page_output_immediately(1);
    try
        graphics_toolkit('qt');
    catch
        try
            graphics_toolkit('gnuplot');
        catch
        end
    end
end

rng(7);

mode = getenv('RUN_MODE');
const = makeConst();
fTrain = linspace(100, 700, 121);
fBand = 300:2:500;
fScore = 300:1:500;
fPlot = 100:1:700;

modelTrain = prepareModel(fTrain, const);
verifyPhysics(const, modelTrain, fTrain);

if strcmp(mode, 'verify') || strcmp(mode, 'bench')
    if strcmp(mode, 'bench')
        benchPhysics(modelTrain, fTrain);
    end
    fprintf('物理核对结束。\n');
    return;
end

quick = strcmp(mode, 'quick');
if quick
    nSamples = 192;
    nPert = 1;
    maxHot = 24;
    maxEpoch = 2;
    minEpoch = 1;
    nRestart = 1;
    nPop = 8;
    nGen = 2;
    physPop = 6;
    physGen = 1;
    maxRounds = 1;
    batchSize = 64;
    nPairD = 12;
    nPairH = 4;
else
    nSamples = 12000;
    nPert = 4;
    maxHot = 800;
    maxEpoch = 32;
    minEpoch = 14;
    nRestart = 5;
    nPop = 40;
    nGen = 24;
    physPop = 24;
    physGen = 16;
    maxRounds = 2;
    batchSize = 256;
    nPairD = 480;
    nPairH = 24;
end
learnRate = 1e-3;

fprintf('生成训练集（均匀 %d 组）...\n', nSamples);
[X, Y] = buildDataset(nSamples, nPert, maxHot, fTrain, modelTrain, const);
fprintf('可用样本 %d 组。\n', size(X, 1));
fprintf('为同一组孔径配对两种腔高（%d 组孔径，每组 %d 个腔高）...\n', nPairD, nPairH);
[pairSeeds, Xpair, Ypair] = pairedSeeds(modelTrain, const, nPairD, nPairH);
X = [X; Xpair];
Y = [Y; Ypair];
fprintf('加入配对样本后共 %d 组。\n', size(X, 1));

[XTrain, YTrain, XVal, YVal, XTest, YTest, xMean, xStd] = splitStandardize(X, Y);
yWeight = spectrumWeight(fTrain);

nSpec = 2 * numel(fTrain);
fprintf('\n===== 训练正向网络：20 -> 64 -> 128 -> 128 -> %d =====\n', nSpec);
net = initMLP([20, 64, 128, 128, nSpec], {'relu', 'relu', 'relu', 'linear'});
net = trainRegressor(net, XTrain, YTrain, XVal, YVal, ...
    maxEpoch, minEpoch, batchSize, learnRate, yWeight, '正向网络');
[testMaeF, testMaeB] = reportMae(net, XTest, YTest, fTrain, '测试集');

modelBand = prepareModel(fBand, const);
modelScore = prepareModel(fScore, const);
modelPlot = prepareModel(fPlot, const);

best = [];
for roundId = 1:maxRounds
    fprintf('\n===== 第 %d 轮：网络搜索，传递矩阵复核 =====\n', roundId);
    warm = [ ...
        4.13987519, 1.00000000, 3.83105896, 7.00000000, 6.77207171, 1.00000000, ...
        1.00000000, 7.00000000, 6.94331789, 7.00000000, 1.00000000, 7.00000000, ...
        77.29898187, 10.57697070, 51.72463418, 80, 10, 25.72288141, 80, 53.39645462, ...
        32.98254136, 77.31856689, 10.18986936, 10, 75.03983407, 80, 10, 80; ...
        3.159769, 0.36, 4.713910, 3.619750, 8, 6.944, ...
        4.825182, 0.10, 3.069580, 4.422677, 1.197, 0.10, ...
        48.162747, 10, 46.210000, 10, 28.311400, 10, 58.179000, 19.998762, ...
        10.638192, 59, 42.788603, 59, 36.744369, 59, 20.055518, 10];
    seeds = [warm; pairSeeds; structuredSeeds(const)];
    scored = searchOnce(net, xMean, xStd, fTrain, seeds, const, ...
        modelBand, modelScore, nRestart, nPop, nGen, physPop, physGen);
    printDesign(scored, '本轮最优');
    if isempty(best) || betterDesign(scored, best)
        best = scored;
    end
    if scored.ok
        fprintf('传递矩阵判定两态都大于 0.4，停止搜索。\n');
        break;
    end
    if roundId == maxRounds
        break;
    end
    fprintf('尚未达标。在较优几何附近补样本并继续训练。\n');
    [Xn, Yn] = refineCloud(scored.pool, fTrain, modelTrain, const, quick);
    XTrain = [XTrain; standardizeRows(Xn, xMean, xStd)]; %#ok<AGROW>
    YTrain = [YTrain; Yn]; %#ok<AGROW>
    extra = 8;
    if ~quick
        extra = 14;
    end
    net = trainRegressor(net, XTrain, YTrain, XVal, YVal, ...
        extra, extra, batchSize, learnRate, yWeight, '正向网络续训');
    reportMae(net, XTest, YTest, fTrain, '续训后测试集');
end

fprintf('\n用 100:1:700 Hz 的传递矩阵计算最终曲线...\n');
dMm = best.dMm;
H1mm = best.H1mm;
H2mm = best.H2mm;
[af1, ab1, energy1] = acousticMm(dMm, H1mm, fPlot, modelPlot);
[af2, ab2, energy2] = acousticMm(dMm, H2mm, fPlot, modelPlot);
final = bandReport(fPlot, af1, ab1, af2, ab2);
final.energy1 = energy1;
final.energy2 = energy2;
final.dMm = dMm;
final.H1mm = H1mm;
final.H2mm = H2mm;
final.Heq1mm = equivalentHeight(reshapeMm(H1mm, 2)) * 1e3;
final.Heq2mm = equivalentHeight(reshapeMm(H2mm, 2)) * 1e3;

fprintf('\n========== 传递矩阵最终结果（300–500 Hz）==========\n');
printDesign(final, '最终');
fprintf('能量误差  状态1 %.3e  状态2 %.3e\n', energy1, energy2);

outDir = fileparts(mfilename('fullpath'));
if isempty(outDir)
    outDir = pwd;
end
figPath = fullfile(outDir, 'origami_two_state.png');
matPath = fullfile(outDir, 'origami_two_state.mat');
csvPath = fullfile(outDir, 'origami_two_state.csv');
plotTwoStates(fPlot, af1, ab1, af2, ab2, final, figPath);
writeCurves(csvPath, fPlot, af1, ab1, af2, ab2);
maeF = testMaeF;
maeB = testMaeB;
best.maeF = maeF;
best.maeB = maeB;
ok = final.ok;
g1 = final.g1;
g2 = final.g2;
save(matPath, 'fPlot', 'fTrain', 'dMm', 'H1mm', 'H2mm', ...
    'af1', 'ab1', 'af2', 'ab2', 'final', 'ok', 'g1', 'g2', ...
    'net', 'xMean', 'xStd', 'maeF', 'maeB');
fprintf('图已保存到 %s\n', figPath);
fprintf('曲线已保存到 %s\n', csvPath);
fprintf('参数已保存到 %s\n', matPath);
end

function const = makeConst()
const.rho0 = 1.29;
const.c0 = 343;
const.eta = 1.89e-5;
const.kappa = 0.026;
const.Cv = 712;
const.gamma = 1.4;
const.P0 = const.rho0 * const.c0 ^ 2 / const.gamma;
const.Z0 = const.rho0 * const.c0;
const.nu = const.eta / const.rho0;
const.muBar = const.kappa / (const.rho0 * const.Cv);
const.W = 30e-3;
const.L = 30e-3;
const.t = 2e-3;
const.a = 30;
const.hc = 20;
const.n0 = 4;
const.beta = 45;
const.stinsonN = 200;
const.dMin = 0.1;
const.dMax = 8;
const.Hmin = 10;
const.Hmax = 80;
const.gap = 0.4;
const.gapTarget = 0.42;
end

function model = prepareModel(f, const)
model.const = const;
[model.kEff, model.zEff] = ductStinson(f, const);
model.f = f(:).';
end

function [kEff, zEff] = ductStinson(f, const)
omega = 2 * pi * reshape(f, 1, []);
nf = numel(omega);
N = const.stinsonN;
bn = (2 * (0:N) + 1) * pi / const.L;
Sv = zeros(1, nf);
St = zeros(1, nf);
for m = 0:N
    am = (2 * m + 1) * pi / const.W;
    lambda2 = am ^ 2 + bn .^ 2;
    wmn = 1 ./ (am ^ 2 * bn .^ 2);
    num = wmn .* lambda2;
    Sv = Sv + sum(num.' ./ (lambda2.' + 1i * omega / const.nu), 1);
    St = St + sum(num.' ./ (lambda2.' + 1i * omega / const.muBar), 1);
end
factor = 64 / (const.W ^ 2 * const.L ^ 2);
Fv = factor * Sv;
Ft = factor * St;
rhoEff = const.rho0 ./ (1 - Fv);
kMod = const.gamma * const.P0 ./ (1 + (const.gamma - 1) * Ft);
kEff = omega .* sqrt(rhoEff ./ kMod);
zEff = sqrt(rhoEff .* kMod);
end

function Heq = equivalentHeight(H)
% H 的单位是 m，形状任意。Heq 与 H 同形状，单位也是 m。
a = 30;
hc = 20;
n0 = 4;
beta = 45;
Hmm = H * 1e3;
h = Hmm / n0;
q = sqrt(max(hc ^ 2 - h .^ 2, 0));
tb = tand(beta);
V1 = a .* h .* q / 16;
V2 = h .* (a - hc) .* q / (16 * tb);
V3 = h .* q / 24 .* (hc / tb - q / 2);
V = 8 * n0 .* (a ^ 2 .* h / 8 - V1 - V2 - V3);
Heq = V * 1e-9 / (30e-3 * 30e-3);
end

function [alphaF, alphaB, energyErr] = origamiResponse(dFlat, hFlat, f, model)
dFlat = asRows(dFlat, 12);
hFlat = asRows(hFlat, 8);
if size(dFlat, 1) ~= size(hFlat, 1)
    error('孔径和腔高的样本数不一致。');
end
B = size(dFlat, 1);
const = model.const;
omega = 2 * pi * reshape(f, 1, []);
Nf = numel(omega);
D = diametersFromRow(dFlat);
Hv = equivalentHeight(depthsFromRow(hFlat));
phi = pi * (D .^ 2) / (4 * const.W * const.L);
Zp = zeros(4, 3, B, Nf);
for s = 1:4
    for j = 1:3
        d = reshape(D(s, j, :), B, 1);
        ph = reshape(phi(s, j, :), B, 1);
        y = (d / 2) ./ sqrt(const.eta ./ (const.rho0 * omega));
        qv = y * sqrt(-1i);
        F = 1 - 2 * besselj(1, qv) ./ (qv .* besselj(0, qv));
        Z = 1i * const.rho0 * const.t * omega ./ (ph .* F) ...
            + sqrt(2) * const.eta * y ./ (ph .* d) ...
            + 1i * 0.85 * const.rho0 * d .* omega ./ ph;
        Zp(s, j, :, :) = reshape(Z, 1, 1, B, Nf);
    end
end

Y11 = zeros(B, Nf);
Y12 = zeros(B, Nf);
Y21 = zeros(B, Nf);
Y22 = zeros(B, Nf);
kRow = reshape(model.kEff, 1, []);
zRow = reshape(model.zEff, 1, []);
if numel(kRow) ~= Nf || numel(zRow) ~= Nf
    error('频率轴和预先算好的 Stinson 参数长度不一致。');
end
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
Z0 = const.Z0;
den = A + Bm / Z0 + C * Z0 + Dm;
tau = 2 ./ den;
rf = (A + Bm / Z0 - C * Z0 - Dm) ./ den;
rb = (-A + Bm / Z0 - C * Z0 + Dm) ./ den;
Rf = abs(rf) .^ 2;
Rb = abs(rb) .^ 2;
Tr = abs(tau) .^ 2;
alphaF = 1 - Rf - Tr;
alphaB = 1 - Rb - Tr;
if nargout >= 3
    e1 = abs(alphaF + Rf + Tr - 1);
    e2 = abs(alphaB + Rb + Tr - 1);
    good = isfinite(e1) & isfinite(e2);
    if any(good(:))
        energyErr = max(max(e1(good)), max(e2(good)));
    else
        energyErr = inf;
    end
end
end

function [A, Bm, C, Dm] = cellTransfer(Z1, Z2, Z3, H1, H2, kRow, zRow)
M = plateMatrix(Z1);
M = mulCell(M, cavityMatrix(H1, kRow, zRow));
M = mulCell(M, plateMatrix(Z2));
M = mulCell(M, cavityMatrix(H2, kRow, zRow));
M = mulCell(M, plateMatrix(Z3));
A = M{1};
Bm = M{2};
C = M{3};
Dm = M{4};
end

function M = plateMatrix(Z)
M = {ones(size(Z)), Z, zeros(size(Z)), ones(size(Z))};
end

function M = cavityMatrix(H, kRow, zRow)
kh = H .* kRow;
ck = cos(kh);
sk = sin(kh);
M = {ck, 1i * zRow .* sk, 1i * sk ./ zRow, ck};
end

function M = mulCell(A, B)
A11 = A{1};
A12 = A{2};
A21 = A{3};
A22 = A{4};
B11 = B{1};
B12 = B{2};
B21 = B{3};
B22 = B{4};
M = { ...
    A11 .* B11 + A12 .* B21, ...
    A11 .* B12 + A12 .* B22, ...
    A21 .* B11 + A22 .* B21, ...
    A21 .* B12 + A22 .* B22};
end

function M = asRows(M, nCol)
if isvector(M) && numel(M) == nCol
    M = reshape(M, 1, nCol);
end
if size(M, 2) ~= nCol
    error('列数应为 %d。', nCol);
end
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

function M = reshapeMm(vMm, width)
if width == 3
    nCol = 12;
else
    nCol = 8;
end
v = asRows(vMm, nCol) * 1e-3;
if width == 3
    M = diametersFromRow(v);
else
    M = depthsFromRow(v);
end
M = M(:, :, 1);
end

function [af, ab, energyErr] = acousticMm(dMm, hMm, f, model)
[af, ab, energyErr] = origamiResponse(asRows(dMm, 12) * 1e-3, asRows(hMm, 8) * 1e-3, f, model);
end

function verifyPhysics(const, model, f)
[Heq80, V80] = volumeOfHeight(80);
[Heq10, V10] = volumeOfHeight(10);
fprintf('自检 H=80 mm -> V=%.4f mm^3，Heq=%.4f mm\n', V80, Heq80);
fprintf('自检 H=10 mm -> V=%.4f mm^3，Heq=%.4f mm\n', V10, Heq10);
if abs(V80 - 72000) > 1e-2 || abs(Heq80 - 80) > 1e-3
    error('H=80 mm 的四层折纸体积或等效高度不符。');
end
if V10 <= 0 || Heq10 <= 0
    error('H=10 mm 的折纸腔体积不是正数。');
end

d1 = repmat([2.75, 1.73, 0.42], 1, 4);
h1 = 47 * ones(1, 8);
d2 = [0.4, 1.2, 3.6, 2.2, 0.8, 4.4, 1.5, 2.8, 0.6, 3.3, 1.1, 2.4];
h2 = [12, 28, 36, 18, 52, 24, 44, 33];
fCheck = [100, 350, 700];
[kRef, zRef] = stinsonLoop(fCheck, const);
kErr = max(abs(kRef - model.kEff(fCheckIndex(f, fCheck))));
zErr = max(abs(zRef - model.zEff(fCheckIndex(f, fCheck))));
fprintf('Stinson 对照  波数差 %.3e  阻抗差 %.3e\n', kErr, zErr);
if kErr > 1e-8 || zErr > 1e-6
    error('向量化 Stinson 求和与逐项求和不一致。');
end

modelCheck = prepareModel(fCheck, const);
[afA, abA] = acousticMm(d1, h1, fCheck, modelCheck);
[afB, abB] = acousticMm(d2, h2, fCheck, modelCheck);
[afBoth, abBoth] = acousticMm([d1; d2], [h1; h2], fCheck, modelCheck);
afRefA = referenceAlpha(d1, h1, fCheck, const, 'f');
abRefA = referenceAlpha(d1, h1, fCheck, const, 'b');
afRefB = referenceAlpha(d2, h2, fCheck, const, 'f');
abRefB = referenceAlpha(d2, h2, fCheck, const, 'b');
err = max([ ...
    max(abs(afA - afRefA)), max(abs(abA - abRefA)), ...
    max(abs(afB - afRefB)), max(abs(abB - abRefB)), ...
    max(abs(afBoth(1, :) - afA)), max(abs(abBoth(2, :) - abB))]);
fprintf('传递矩阵对照  最大吸声差 %.3e\n', err);
fprintf('名义几何 350 Hz  正向 %.4f  反向 %.4f\n', afA(2), abA(2));
if err > 1e-8
    error('批量传递矩阵和逐单胞参照不一致。');
end
if any(afA < -1e-6) || any(abA < -1e-6) || any(afA > 1 + 1e-6) || any(abA > 1 + 1e-6)
    error('名义几何的吸声系数超出被动范围。');
end
end

function idx = fCheckIndex(fAll, fPick)
idx = zeros(1, numel(fPick));
for n = 1:numel(fPick)
    idx(n) = find(abs(fAll - fPick(n)) < 1e-9, 1);
end
end

function [HeqMm, V] = volumeOfHeight(H_mm)
Heq = equivalentHeight(H_mm * 1e-3);
HeqMm = Heq * 1e3;
V = Heq * (30e-3 * 30e-3) / 1e-9;
end

function [kEff, zEff] = stinsonLoop(f, const)
omega = 2 * pi * reshape(f, 1, []);
nf = numel(omega);
N = const.stinsonN;
Sv = zeros(1, nf);
St = zeros(1, nf);
for m = 0:N
    am = (2 * m + 1) * pi / const.W;
    for n = 0:N
        bn = (2 * n + 1) * pi / const.L;
        lambda2 = am ^ 2 + bn ^ 2;
        wmn = 1 / (am ^ 2 * bn ^ 2);
        Sv = Sv + wmn * lambda2 ./ (lambda2 + 1i * omega / const.nu);
        St = St + wmn * lambda2 ./ (lambda2 + 1i * omega / const.muBar);
    end
end
factor = 64 / (const.W ^ 2 * const.L ^ 2);
Fv = factor * Sv;
Ft = factor * St;
rhoEff = const.rho0 ./ (1 - Fv);
kMod = const.gamma * const.P0 ./ (1 + (const.gamma - 1) * Ft);
kEff = omega .* sqrt(rhoEff ./ kMod);
zEff = sqrt(rhoEff .* kMod);
end

function alpha = referenceAlpha(dMm, hMm, f, const, side)
d = reshapeMm(dMm, 3);
H = reshapeMm(hMm, 2);
S = const.W * const.L;
omega = 2 * pi * reshape(f, 1, []);
Nf = numel(omega);
[kEff, zEff] = stinsonLoop(f, const);
Heq = equivalentHeight(H);
phi = pi * (d .^ 2) / (4 * S);
Zp = complex(zeros(4, 3, Nf));
for s = 1:4
    for j = 1:3
        y = (d(s, j) / 2) ./ sqrt(const.eta ./ (const.rho0 * omega));
        qv = y * sqrt(-1i);
        F = 1 - 2 * besselj(1, qv) ./ (qv .* besselj(0, qv));
        Z = 1i * omega * const.rho0 * const.t ./ (phi(s, j) .* F) ...
            + sqrt(2) * const.eta * y ./ (phi(s, j) * d(s, j)) ...
            + 1i * 0.85 * omega * const.rho0 * d(s, j) / phi(s, j);
        Zp(s, j, :) = reshape(Z, 1, 1, Nf);
    end
end
alpha = zeros(1, Nf);
for ii = 1:Nf
    Yt = complex(zeros(2, 2));
    k = kEff(ii);
    Zc = zEff(ii);
    for s = 1:4
        Tp1 = [1, Zp(s, 1, ii); 0, 1];
        Tp2 = [1, Zp(s, 2, ii); 0, 1];
        Tp3 = [1, Zp(s, 3, ii); 0, 1];
        Tc1 = cavityScalar(Heq(s, 1), k, Zc);
        Tc2 = cavityScalar(Heq(s, 2), k, Zc);
        Tm = Tp1 * Tc1 * Tp2 * Tc2 * Tp3;
        A = Tm(1, 1);
        Bm = Tm(1, 2);
        C = Tm(2, 1);
        Dm = Tm(2, 2);
        Ys = [Dm / Bm, C - A * Dm / Bm; 1 / Bm, -A / Bm];
        Yt = Yt + 0.25 * Ys;
    end
    Y11 = Yt(1, 1);
    Y12 = Yt(1, 2);
    Y21 = Yt(2, 1);
    Y22 = Yt(2, 2);
    A = -Y22 / Y21;
    Bm = 1 / Y21;
    C = Y12 - Y11 * Y22 / Y21;
    Dm = Y11 / Y21;
    den = A + Bm / const.Z0 + C * const.Z0 + Dm;
    tau = 2 / den;
    rf = (A + Bm / const.Z0 - C * const.Z0 - Dm) / den;
    rb = (-A + Bm / const.Z0 - C * const.Z0 + Dm) / den;
    if side == 'f'
        alpha(ii) = 1 - abs(rf) ^ 2 - abs(tau) ^ 2;
    else
        alpha(ii) = 1 - abs(rb) ^ 2 - abs(tau) ^ 2;
    end
end
end

function T = cavityScalar(H, k, Zc)
T = [cos(k * H), 1i * Zc * sin(k * H); 1i * sin(k * H) / Zc, cos(k * H)];
end

function benchPhysics(model, f)
d = constBounds(model.const, 48, 12, model.const.dMin, model.const.dMax);
h = constBounds(model.const, 48, 8, model.const.Hmin, model.const.Hmax);
tic;
[af, ab] = acousticMm(d, h, f, model);
elapsed = toc;
fprintf('计时  %d 组 × %d 个频率  用时 %.2f s，每组 %.3f s\n', ...
    size(d, 1), numel(f), elapsed, elapsed / size(d, 1));
fprintf('这批吸声范围  %.3f – %.3f\n', min([af(:); ab(:)]), max([af(:); ab(:)]));
end

function M = constBounds(~, n, nCol, lo, hi)
M = lo + (hi - lo) * rand(n, nCol);
end

function [X, Y] = buildDataset(nBase, nPert, maxHot, f, model, const)
[X, Y] = sampleUniform(nBase, f, model, const);
band = f >= 300 & f <= 500;
nf = numel(f);
af = Y(:, 1:nf);
ab = Y(:, nf + 1:end);
gap = mean(af(:, band), 2) - mean(ab(:, band), 2);
fprintf('均匀样本在 300–500 Hz 的方向差范围 %.3f 到 %.3f\n', min(gap), max(gap));
[~, order] = sort(abs(gap), 'descend');
nHot = min(maxHot, numel(order));
hot = order(1:nHot);
if nPert <= 0 || nHot == 0
    return;
end
sig = [0.35 * ones(1, 12), 3.5 * ones(1, 8)];
blocks = cell(1, nPert);
for i = 1:nPert
    blocks{i} = X(hot, :) + randn(nHot, 20) .* sig;
end
Xp = cat(1, blocks{:});
Xp = clampDesign(Xp, const);
[af, ab] = acousticChunk(Xp(:, 1:12), Xp(:, 13:20), f, model, 32);
Yp = [af, ab];
good = finiteSpectra(Yp);
X = [X; Xp(good, :)];
Y = [Y; Yp(good, :)];
fprintf('  大方向差附近追加 %d 组。\n', sum(good));
end

function [X, Y] = sampleUniform(n, f, model, const)
nf = numel(f);
X = zeros(n, 20);
Y = zeros(n, 2 * nf);
filled = 0;
while filled < n
    draw = min(64, n - filled);
    d = const.dMin + (const.dMax - const.dMin) * rand(draw, 12);
    h = const.Hmin + (const.Hmax - const.Hmin) * rand(draw, 8);
    [af, ab] = acousticChunk(d, h, f, model, 32);
    Ytry = [af, ab];
    good = finiteSpectra(Ytry);
    take = min(sum(good), n - filled);
    if take == 0
        error('这一批样本的吸声系数都不是有限值。');
    end
    keep = find(good);
    keep = keep(1:take);
    rows = filled + (1:take);
    X(rows, 1:12) = d(keep, :);
    X(rows, 13:20) = h(keep, :);
    Y(rows, :) = Ytry(keep, :);
    filled = filled + take;
    fprintf('  已接受 %d / %d\n', filled, n);
    flushOut();
end
end

function [af, ab] = acousticChunk(d, h, f, model, chunk)
n = size(d, 1);
af = zeros(n, numel(f));
ab = zeros(n, numel(f));
for s = 1:chunk:n
    e = min(s + chunk - 1, n);
    [af(s:e, :), ab(s:e, :)] = acousticMm(d(s:e, :), h(s:e, :), f, model);
end
end

function good = finiteSpectra(Y)
good = all(isfinite(Y), 2) & all(Y > -0.05, 2) & all(Y < 1.05, 2);
end

function X = clampDesign(X, const)
X(:, 1:12) = min(max(X(:, 1:12), const.dMin), const.dMax);
X(:, 13:end) = min(max(X(:, 13:end), const.Hmin), const.Hmax);
end

function [Xn, Yn] = refineCloud(pool, f, model, const, quick)
if isempty(pool)
    Xn = zeros(0, 20);
    Yn = zeros(0, 2 * numel(f));
    return;
end
base = [pool(:, 1:20); [pool(:, 1:12), pool(:, 21:28)]];
nPer = 8;
nUni = 1200;
if quick
    nPer = 1;
    nUni = 16;
end
sig = [0.25 * ones(1, 12), 2.5 * ones(1, 8)];
nB = size(base, 1);
Xw = repmat(base, nPer, 1) + randn(nB * nPer, 20) .* sig;
Xu = [const.dMin + (const.dMax - const.dMin) * rand(nUni, 12), ...
    const.Hmin + (const.Hmax - const.Hmin) * rand(nUni, 8)];
Xn = clampDesign([Xw; Xu], const);
[af, ab] = acousticChunk(Xn(:, 1:12), Xn(:, 13:20), f, model, 32);
Yn = [af, ab];
good = finiteSpectra(Yn);
Xn = Xn(good, :);
Yn = Yn(good, :);
fprintf('  续训样本 %d 组。\n', size(Xn, 1));
end

function [XTrain, YTrain, XVal, YVal, XTest, YTest, xMean, xStd] = splitStandardize(X, Y)
n = size(X, 1);
perm = randperm(n);
X = X(perm, :);
Y = Y(perm, :);
nTest = max(40, round(0.1 * n));
nVal = nTest;
nTest = min(nTest, floor(n / 5));
nVal = min(nVal, floor(n / 5));
XTest = X(1:nTest, :);
YTest = Y(1:nTest, :);
XVal = X(nTest + (1:nVal), :);
YVal = Y(nTest + (1:nVal), :);
XTrainRaw = X(nTest + nVal + 1:end, :);
YTrain = Y(nTest + nVal + 1:end, :);
xMean = mean(XTrainRaw, 1);
xStd = std(XTrainRaw, 0, 1);
xStd(xStd < 1e-12) = 1;
XTrain = standardizeRows(XTrainRaw, xMean, xStd);
XVal = standardizeRows(XVal, xMean, xStd);
XTest = standardizeRows(XTest, xMean, xStd);
end

function Xs = standardizeRows(X, xMean, xStd)
Xs = (X - xMean) ./ xStd;
end

function w = spectrumWeight(f)
nf = numel(f);
band = double(f >= 300 & f <= 500);
wb = 1 + 3 * band;
w = [wb, wb];
w = w / mean(w);
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

function net = trainRegressor(net, X, Y, XVal, YVal, epochs, minEpoch, batchSize, learnRate, yWeight, name)
adam = initAdam(net);
n = size(X, 1);
useBatch = min(batchSize, n);
w = yWeight(:);
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
        diff = cache.y - Y(batch, :).';
        total = total + sum(sum((diff .^ 2) .* w));
        seen = seen + numel(diff);
        grads = mlpBackward(net, cache, (2 / numel(diff)) * (diff .* w));
        [net, adam] = adamApply(net, adam, grads, learnRate);
    end
    pred = mlpPredict(net, XVal);
    mae = mean(abs(pred(:) - YVal(:)));
    fprintf('%s epoch %02d  train_wmse=%.6e  val_mae=%.6e\n', ...
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
        fprintf('%s 验证误差停住，采用此前的最优权重。\n', name);
        break;
    end
end
net = bestNet;
end

function net = copyNet(net)
out = net;
for k = 1:numel(net.W)
    out.W{k} = net.W{k};
    out.b{k} = net.b{k};
end
net = out;
end

function [maeF, maeB, maeBand] = reportMae(net, X, Y, f, name)
pred = mlpPredict(net, X);
nf = numel(f);
band = f >= 300 & f <= 500;
afP = pred(:, 1:nf);
abP = pred(:, nf + 1:end);
afY = Y(:, 1:nf);
abY = Y(:, nf + 1:end);
maeF = mean(abs(afP(:) - afY(:)));
maeB = mean(abs(abP(:) - abY(:)));
bandErr = [afP(:, band) - afY(:, band), abP(:, band) - abY(:, band)];
maeBand = mean(abs(bandErr(:)));
fprintf('%s MAE  正向 %.4f  反向 %.4f  300–500 Hz %.4f\n', name, maeF, maeB, maeBand);
end

function [af, ab] = predictAbs(net, xMean, xStd, Xmm)
Xs = standardizeRows(Xmm, xMean, xStd);
Y = mlpPredict(net, Xs);
nf = size(Y, 2) / 2;
af = Y(:, 1:nf);
ab = Y(:, nf + 1:end);
end

function [seeds, X, Y] = pairedSeeds(model, const, nD, nH)
f = model.f;
band = f >= 300 & f <= 500;
d = const.dMin + (const.dMax - const.dMin) * rand(nD, 12);
dRep = zeros(nD * nH, 12);
hAll = const.Hmin + (const.Hmax - const.Hmin) * rand(nD * nH, 8);
for i = 1:nD
    rows = (i - 1) * nH + (1:nH);
    dRep(rows, :) = repmat(d(i, :), nH, 1);
end
[af, ab] = acousticChunk(dRep, hAll, f, model, 32);
gap = mean(af(:, band), 2) - mean(ab(:, band), 2);
seeds = zeros(nD, 28);
pairGap = zeros(nD, 1);
for i = 1:nD
    rows = (i - 1) * nH + (1:nH);
    g = gap(rows);
    [gMax, iMax] = max(g);
    [gMin, iMin] = min(g);
    if iMax == iMin
        iMin = 1 + mod(iMax, nH);
    end
    seeds(i, :) = [d(i, :), hAll(rows(iMax), :), hAll(rows(iMin), :)];
    pairGap(i) = min(gMax, -gMin);
end
[bestPair, key] = max(pairGap);
gRows = (key - 1) * nH + (1:nH);
gMax = max(gap(gRows));
gMin = min(gap(gRows));
fprintf('  配对后 min(两态差值) 最好 %.4f，超过 0.40 的孔径 %d / %d\n', ...
    bestPair, sum(pairGap > 0.4), nD);
fprintf('  最好的一组：正向高出 %.4f，反向高出 %.4f\n', gMax, -gMin);
X = [dRep, hAll];
Y = [af, ab];
good = finiteSpectra(Y);
X = X(good, :);
Y = Y(good, :);
seeds = clampDesign(seeds, const);
end

function seeds = structuredSeeds(const)
dSets = [ ...
    1.2, 0.4, 4.5, 8.0, 2.4, 0.3, 1.5, 0.6, 6.0, 0.2, 3.2, 8.0; ...
    1.0, 2.5, 5.0, 1.0, 2.5, 5.0, 1.0, 2.5, 5.0, 1.0, 2.5, 5.0; ...
    1.2, 3.0, 6.5, 2.0, 4.0, 7.0, 1.5, 3.5, 5.5, 2.2, 4.5, 6.8; ...
    2.0, 1.0, 4.0, 1.5, 3.0, 6.0, 4.0, 1.2, 2.5, 3.0, 5.0, 1.0];
hSets = [ ...
    60, 20, 70, 25, 45, 15, 75, 30; ...
    25, 65, 18, 55, 22, 70, 35, 80; ...
    40, 15, 55, 28, 68, 18, 50, 12];
nD = size(dSets, 1);
nH = size(hSets, 1);
seeds = zeros(nD * nH * (nH - 1), 28);
row = 0;
for i = 1:nD
    for a = 1:nH
        for b = 1:nH
            if a == b
                continue;
            end
            row = row + 1;
            seeds(row, :) = [dSets(i, :), hSets(a, :), hSets(b, :)];
        end
    end
end
seeds = clampDesign(seeds, const);
extra = [const.dMin + (const.dMax - const.dMin) * rand(8, 12), ...
    const.Hmin + (const.Hmax - const.Hmin) * rand(8, 16)];
seeds = [seeds; extra];
end

function scored = searchOnce(net, xMean, xStd, fTrain, seeds, const, ...
    modelBand, modelScore, nRestart, nPop, nGen, physPop, physGen)
lo = bounds(const, 'lo');
hi = bounds(const, 'hi');
pool = zeros(0, 28);
netLoss = zeros(0, 1);
for restart = 1:nRestart
    fprintf('网络差分进化  第 %d / %d 组\n', restart, nRestart);
    useSeed = seeds;
    if restart > 1
        useSeed = seeds(randperm(size(seeds, 1), min(4, size(seeds, 1))), :);
    end
    [xBest, fitBest, pop, fit] = deOptimize( ...
        @(X) netLossOf(net, xMean, xStd, fTrain, X), ...
        lo, hi, nPop, nGen, useSeed);
    fprintf('  网络最优损失 %.6f\n', fitBest);
    [~, order] = sort(fit, 'ascend');
    keep = order(1:min(6, numel(order)));
    pool = [pool; pop(keep, :); xBest]; %#ok<AGROW>
    netLoss = [netLoss; fit(keep); fitBest]; %#ok<AGROW>
end
pool = [pool; seeds];
pool = unique(round(pool * 1000) / 1000, 'rows');
fprintf('传递矩阵初评 %d 个网络候选...\n', size(pool, 1));
info = physicsInfo(pool, modelScore.f, modelScore);
[~, order] = sort(rankKey(info), 'descend');
nKeep = min(8, numel(order));
chosen = pool(order(1:nKeep), :);
fprintf('网络候选里最好的 min(差值) %.4f\n', rankKeyAt(info, order(1)));

polished = chosen;
if physGen > 1
    fprintf('传递矩阵全局搜索（网络候选作为种子）...\n');
    nGlobal = 3;
    globalPop = max(physPop, 28);
    globalGen = max(physGen, 18);
    for restart = 1:nGlobal
        fprintf('  全局第 %d / %d 组\n', restart, nGlobal);
        seedG = [chosen; structuredSeeds(const)];
        [xG, ~, popG, fitG] = deOptimize( ...
            @(X) physLossOf(X, modelBand), lo, hi, globalPop, globalGen, seedG);
        [~, orderG] = sort(fitG, 'ascend');
        keepG = orderG(1:min(4, numel(orderG)));
        polished = [polished; popG(keepG, :); xG]; %#ok<AGROW>
        chosen = [xG; chosen]; %#ok<AGROW>
    end
end

infoPol = physicsInfo(polished, modelScore.f, modelScore);
[~, orderPol] = sort(rankKey(infoPol), 'descend');
nLocal = min(4, numel(orderPol));
localIn = polished(orderPol(1:nLocal), :);
for i = 1:nLocal
    fprintf('传递矩阵交替修正  %d / %d\n', i, nLocal);
    localIn(i, :) = alternatePolish(localIn(i, :), modelBand, const, max(physGen, 1));
end
polished = [polished; localIn];
finalInfo = physicsInfo(polished, modelScore.f, modelScore);
[~, orderFinal] = sort(rankKey(finalInfo), 'descend');
key = orderFinal(1);
scored = packDesign(polished(key, :), finalInfo, key);
nPool = min(12, numel(orderFinal));
scored.pool = polished(orderFinal(1:nPool), :);
scored.maeF = NaN;
scored.maeB = NaN;
end

function value = rankKeyAt(info, idx)
key = rankKey(info);
value = key(idx);
end

function x = alternatePolish(x, model, const, nGen)
lo = bounds(const, 'lo');
hi = bounds(const, 'hi');
rounds = 3;
if nGen <= 2
    rounds = 1;
    nPop = 6;
    nStep = 1;
else
    nPop = 18;
    nStep = max(8, round(nGen * 0.6));
end
slices = {13:20, 21:28, 1:12};
for roundId = 1:rounds
    for s = 1:numel(slices)
        idx = slices{s};
        x = optimizeSlice(x, idx, model, lo, hi, nPop, nStep);
    end
    info = physicsInfo(x, model.f, model);
    fprintf('    交替第 %d 轮  状态1差 %.4f  状态2差 %.4f\n', roundId, info.g1, info.g2);
    flushOut();
    if info.ok
        break;
    end
end
end

function x = optimizeSlice(x, idx, model, lo, hi, nPop, nGen)
base = x;
[z, ~] = deOptimize( ...
    @(Z) sliceLoss(Z, base, idx, model), ...
    lo(idx), hi(idx), nPop, nGen, x(idx));
x(idx) = z;
end

function loss = sliceLoss(Z, base, idx, model)
X = repmat(base, size(Z, 1), 1);
X(:, idx) = Z;
loss = physLossOf(X, model);
end

function loss = netLossOf(net, xMean, xStd, f, X)
d = X(:, 1:12);
[af, ab] = predictAbs(net, xMean, xStd, [d, X(:, 13:20); d, X(:, 21:28)]);
n = size(X, 1);
af1 = af(1:n, :);
ab1 = ab(1:n, :);
af2 = af(n + 1:end, :);
ab2 = ab(n + 1:end, :);
loss = gapLoss(f, af1, ab1, af2, ab2);
end

function info = physicsInfo(X, f, model)
d = X(:, 1:12);
[af, ab] = acousticChunk([d; d], [X(:, 13:20); X(:, 21:28)], f, model, 16);
n = size(X, 1);
info = gapInfo(f, af(1:n, :), ab(1:n, :), af(n + 1:end, :), ab(n + 1:end, :));
end

function x = physicsPolish(x, model, const, nPop, nGen)
lo = bounds(const, 'lo');
hi = bounds(const, 'hi');
span = hi - lo;
localLo = max(lo, x - 0.18 * span);
localHi = min(hi, x + 0.18 * span);
seeds = x;
[x, ~] = deOptimize(@(X) physLossOf(X, model), localLo, localHi, nPop, nGen, seeds);
x = clampDesign(x, const);
x = coordinatePolish(x, @(X) physLossOf(X, model), lo, hi);
end

function loss = physLossOf(X, model)
info = physicsInfo(X, model.f, model);
loss = info.loss;
end

function [xBest, fitBest, pop, fit] = deOptimize(evalBatch, lo, hi, nPop, nGen, seeds)
dim = numel(lo);
lo = lo(:).';
hi = hi(:).';
pop = repmat(lo, nPop, 1) + rand(nPop, dim) .* repmat(hi - lo, nPop, 1);
nSeed = min(size(seeds, 1), nPop);
if nSeed > 0
    pop(1:nSeed, :) = min(max(seeds(1:nSeed, :), lo), hi);
end
fit = evalBatch(pop);
F = 0.65;
CR = 0.9;
for gen = 1:nGen
    base = zeros(nPop, dim);
    diffA = zeros(nPop, dim);
    diffB = zeros(nPop, dim);
    for i = 1:nPop
        idx = randperm(nPop, 4);
        idx(idx == i) = [];
        idx = idx(1:3);
        base(i, :) = pop(idx(1), :);
        diffA(i, :) = pop(idx(2), :);
        diffB(i, :) = pop(idx(3), :);
    end
    mutant = base + F * (diffA - diffB);
    mutant = min(max(mutant, repmat(lo, nPop, 1)), repmat(hi, nPop, 1));
    cross = rand(nPop, dim) < CR;
    force = randi(dim, nPop, 1);
    for i = 1:nPop
        cross(i, force(i)) = true;
    end
    trial = pop;
    trial(cross) = mutant(cross);
    fitTrial = evalBatch(trial);
    better = fitTrial < fit;
    pop(better, :) = trial(better, :);
    fit(better) = fitTrial(better);
    [fitBest, ~] = min(fit);
    if gen == 1 || gen == nGen || mod(gen, 8) == 0
        fprintf('  第 %02d 代  最优损失 %.6f\n', gen, fitBest);
        flushOut();
    end
end
[fitBest, key] = min(fit);
xBest = pop(key, :);
end

function x = coordinatePolish(x, evalBatch, lo, hi)
scales = [1, 0.3];
baseStep = [1.2 * ones(1, 12), 5 * ones(1, 16)];
for s = 1:numel(scales)
    step = baseStep * scales(s);
    trials = zeros(0, numel(x));
    for j = 1:numel(x)
        xp = x;
        xm = x;
        xp(j) = min(hi(j), x(j) + step(j));
        xm(j) = max(lo(j), x(j) - step(j));
        trials = [trials; xp; xm]; %#ok<AGROW>
    end
    loss = evalBatch(trials);
    current = evalBatch(x);
    [bestLoss, key] = min(loss);
    if bestLoss + 1e-8 < current
        x = trials(key, :);
    end
end
end

function v = bounds(const, which)
if strcmp(which, 'lo')
    v = [const.dMin * ones(1, 12), const.Hmin * ones(1, 16)];
else
    v = [const.dMax * ones(1, 12), const.Hmax * ones(1, 16)];
end
end

function loss = gapLoss(f, af1, ab1, af2, ab2)
info = gapInfo(f, af1, ab1, af2, ab2);
loss = info.loss;
end

function info = gapInfo(f, af1, ab1, af2, ab2)
band = f >= 300 & f <= 500;
mf1 = mean(af1(:, band), 2);
mb1 = mean(ab1(:, band), 2);
mf2 = mean(af2(:, band), 2);
mb2 = mean(ab2(:, band), 2);
g1 = mf1 - mb1;
g2 = mb2 - mf2;
target = 0.42;
short = max(0, target - g1) .^ 2 + max(0, target - g2) .^ 2;
loss = short - 0.05 * min(g1, g2);
bad = ~all(isfinite(af1), 2) | ~all(isfinite(ab1), 2) ...
    | ~all(isfinite(af2), 2) | ~all(isfinite(ab2), 2);
loss(bad) = 1e3;
info.mf1 = mf1;
info.mb1 = mb1;
info.mf2 = mf2;
info.mb2 = mb2;
info.g1 = g1;
info.g2 = g2;
info.loss = loss;
info.ok = g1 > 0.4 & g2 > 0.4 & ~bad;
info.bad = bad;
end

function key = rankKey(info)
key = min(info.g1, info.g2);
key(info.bad) = -1e3;
end

function design = packDesign(x, info, key)
design.dMm = x(1:12);
design.H1mm = x(13:20);
design.H2mm = x(21:28);
design.mf1 = info.mf1(key);
design.mb1 = info.mb1(key);
design.mf2 = info.mf2(key);
design.mb2 = info.mb2(key);
design.g1 = info.g1(key);
design.g2 = info.g2(key);
design.ok = info.ok(key);
end

function report = bandReport(f, af1, ab1, af2, ab2)
info = gapInfo(f, af1, ab1, af2, ab2);
report = packDesign([zeros(1, 12), zeros(1, 8), zeros(1, 8)], info, 1);
report.ok = info.ok(1);
end

function yes = betterDesign(a, b)
yes = min(a.g1, a.g2) > min(b.g1, b.g2);
end

function printDesign(design, name)
fprintf('%s  状态1 正向 %.4f  反向 %.4f  正向高出 %.4f  %s\n', ...
    name, design.mf1, design.mb1, design.g1, passFail(design.g1 > 0.4));
fprintf('%s  状态2 正向 %.4f  反向 %.4f  反向高出 %.4f  %s\n', ...
    name, design.mf2, design.mb2, design.g2, passFail(design.g2 > 0.4));
if isfield(design, 'dMm')
    fprintf('孔径 (mm)，每行一个单胞：\n');
    disp(reshape(design.dMm, 3, 4).');
    fprintf('状态1 实际腔高 (mm)：\n');
    disp(reshape(design.H1mm, 2, 4).');
    fprintf('状态2 实际腔高 (mm)：\n');
    disp(reshape(design.H2mm, 2, 4).');
end
if isfield(design, 'Heq1mm')
    fprintf('状态1 等效腔高 (mm)：\n');
    disp(design.Heq1mm);
    fprintf('状态2 等效腔高 (mm)：\n');
    disp(design.Heq2mm);
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
set(0, 'defaultaxesfontname', fontName);
set(0, 'defaulttextfontname', fontName);
fig = figure('Color', 'w', 'Position', [80, 40, 980, 760], 'Visible', 'off');
set(fig, 'defaultaxesfontname', fontName);
set(fig, 'defaulttextfontname', fontName);
subplot(2, 1, 1);
drawState(f, af1, ab1, info.mf1, info.mb1, info.g1, ...
    '状态1：蓝实线 = 正向吸声，红虚线 = 反向吸声', true);
subplot(2, 1, 2);
drawState(f, af2, ab2, info.mf2, info.mb2, info.g2, ...
    '状态2：蓝实线 = 正向吸声，红虚线 = 反向吸声', false);
print(fig, figPath, '-dpng', '-r140');
close(fig);
end

function drawState(f, af, ab, mf, mb, gapValue, ~, forwardLeads)
hold on;
patch([300, 500, 500, 300], [0, 0, 1, 1], [1, 0.92, 0.55], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.45, 'HandleVisibility', 'off');
hF = plot(f, af, 'b-', 'LineWidth', 1.8);
hB = plot(f, ab, 'r--', 'LineWidth', 1.8);
if forwardLeads
    name = sprintf('状态1：300–500 Hz 正向 %.3f，反向 %.3f，正向高出 %.3f', mf, mb, gapValue);
else
    name = sprintf('状态2：300–500 Hz 正向 %.3f，反向 %.3f，反向高出 %.3f', mf, mb, gapValue);
end
grid on;
box on;
xlim([min(f), max(f)]);
ylim([0, 1]);
xticks(min(f):100:max(f));
yticks(0:0.2:1);
xlabel('频率 / Hz');
ylabel('吸声系数');
title(name, 'FontWeight', 'normal');
legend([hF, hB], {'正向吸声', '反向吸声'}, 'Location', 'northeast');
set(gca, 'FontName', 'WenQuanYi Micro Hei', 'FontSize', 13, 'LineWidth', 1.1, 'TickDir', 'in');
end

function writeCurves(path, f, af1, ab1, af2, ab2)
fid = fopen(path, 'w');
if fid < 0
    error('无法写入 %s', path);
end
fprintf(fid, 'f_Hz,state1_forward,state1_backward,state2_forward,state2_backward\n');
data = [f(:), af1(:), ab1(:), af2(:), ab2(:)];
for row = 1:size(data, 1)
    fprintf(fid, '%.0f,%.8f,%.8f,%.8f,%.8f\n', data(row, :));
end
fclose(fid);
end

function flushOut()
if exist('OCTAVE_VERSION', 'builtin')
    fflush(stdout);
else
    drawnow('update');
end
end
