% 单胞正向网络。保存为 ruanhua_shenjing.m 后直接运行。
% 输入：3 个孔径(mm)、2 个腔高(mm)、1 个频率(Hz)。
% 输出：A、B/Z0、C*Z0、D 的实部和虚部。
% 损失：归一化矩阵的均方误差 + 原始传递矩阵吸声的均方误差。

rng(1);

%% 1. 基本设置

f = 100:1:700;
omega = 2*pi*f;

Nf = numel(f);
Ns = 20000;

lb = [0.1, 0.1, 0.1, 10, 10];
ub = [8.0, 8.0, 8.0, 80, 80];

trainRatio = 0.70;
valRatio = 0.15;

clipPercentile = 99.5;
lambdaAlpha = 1;

numEpochs = 80;
miniBatchSize = 2048;
freqPerGeometry = 32;
initialLearnRate = 1e-4;

rho0 = 1.29;
c0 = 343;
eta = 1.89e-5;

Z0 = rho0 * c0;
k0 = omega / c0;

W = 30e-3;
L = 30e-3;
S = W * L;

t = [2, 2, 2] * 1e-3;
n_holes = [1, 1, 1];

a = 30;
hc = 20;
n0 = 4;
beta = 45;

names = { ...
    'Re(A)', 'Im(A)', ...
    'Re(B/Z0)', 'Im(B/Z0)', ...
    'Re(C*Z0)', 'Im(C*Z0)', ...
    'Re(D)', 'Im(D)'};

fprintf('\n========================================\n');
fprintf('FPN: dimensionless matrix + absorption loss\n');
fprintf('========================================\n');

%% 2. 拉丁超立方采样

fprintf('\nStep 1: LHS sampling...\n');

X_LHS = lhsdesign(Ns, 5, 'criterion', 'none');
G = single(lb + X_LHS .* (ub - lb));
clear X_LHS;

%% 3. 折纸等效腔高

h = double(G(:, 4:5)) / n0;
q = sqrt(max(0, hc ^ 2 - h .^ 2));

V1 = a .* h .* q / 16;
V2 = h .* (a - hc) .* q / (16 * tand(beta));
V3 = h .* q / 24 .* (hc / tand(beta) - q / 2);
V_mm3 = 8 * n0 .* (a ^ 2 .* h / 8 - V1 - V2 - V3);
H_eq = single(V_mm3 * 1e-9 / S);

if any(~isfinite(H_eq(:))) || any(H_eq(:) <= 0)
    error('Invalid equivalent cavity height.');
end

%% 4. 按几何划分

idx = randperm(Ns);
nTrain = round(Ns * trainRatio);
nVal = round(Ns * valRatio);

idTrain = idx(1:nTrain);
idVal = idx(nTrain + 1:nTrain + nVal);
idTest = idx(nTrain + nVal + 1:end);

fprintf('Training %d, validation %d, testing %d\n', ...
    numel(idTrain), numel(idVal), numel(idTest));

%% 5. 传递矩阵和原始吸声

fprintf('\nStep 2: Generating TMM data...\n');

RAW = cell(1, 8);
for k = 1:8
    RAW{k} = zeros(Ns, Nf, 'single');
end
ALF = zeros(Ns, Nf, 2, 'single');

geometryBatchSize = 100;

for firstID = 1:geometryBatchSize:Ns
    lastID = min(firstID + geometryBatchSize - 1, Ns);
    ids = firstID:lastID;

    [result, Af, Ab] = calculateTMMBatch( ...
        double(G(ids, :)), double(H_eq(ids, :)), ...
        omega, k0, rho0, eta, Z0, S, t, n_holes);

    for k = 1:8
        if any(~isfinite(result{k}(:)))
            error('TMM component %d contains NaN/Inf.', k);
        end
        RAW{k}(ids, :) = single(result{k});
    end

    ALF(ids, :, 1) = single(Af);
    ALF(ids, :, 2) = single(Ab);

    if mod(lastID, 1000) == 0 || lastID == Ns
        fprintf('TMM completed: %d/%d\n', lastID, Ns);
    end
end

clear result Af Ab;

[~, peakForward] = max(ALF(:, :, 1), [], 2);
[~, peakBackward] = max(ALF(:, :, 2), [], 2);

%% 6. 软处理和最小最大缩放

fprintf('\nStep 3: Preprocessing...\n');

b_soft = zeros(1, 8);
Ymin = zeros(1, 8);
Ymax = zeros(1, 8);

for k = 1:8
    values = abs(double(RAW{k}(idTrain, :)));
    b = prctile(values(:), clipPercentile);
    if ~isfinite(b) || b <= 0
        b = max(values(:));
        if b <= 0
            b = 1;
        end
    end
    b_soft(k) = b;

    trainSoft = softClipping(double(RAW{k}(idTrain, :)), b);
    Ymin(k) = min(trainSoft(:));
    Ymax(k) = max(trainSoft(:));
    if Ymax(k) <= Ymin(k)
        Ymax(k) = Ymin(k) + 1;
    end

    fprintf('%-10s | b=%+.4e | min=%+.4e | max=%+.4e\n', ...
        names{k}, b_soft(k), Ymin(k), Ymax(k));
    clear values trainSoft;
end

Xmin = [lb, f(1)];
Xmax = [ub, f(end)];
Xrange = Xmax - Xmin;
Yrange = Ymax - Ymin;

save('Origami_FPN_TMM_20000.mat', ...
    'G', 'H_eq', 'f', 'RAW', 'ALF', ...
    'peakForward', 'peakBackward', ...
    'idTrain', 'idVal', 'idTest', ...
    'b_soft', 'Ymin', 'Ymax', ...
    'Xmin', 'Xmax', 'Xrange', 'Yrange', ...
    'names', 'Z0', 'lambdaAlpha', ...
    '-v7.3');

%% 7. 网络

layers = [
    featureInputLayer(6, 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(300, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(500, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(500, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(100, 'Name', 'fc4')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(8, 'Name', 'output')
];

net = dlnetwork(layers);

%% 8. 固定验证样本：每隔 10 Hz，再加正反向吸声峰及其相邻点

peakIds = [peakNeighbors(peakForward(idVal), Nf); ...
    peakNeighbors(peakBackward(idVal), Nf)];
valFrequency = unique([1:10:Nf, peakIds(:)']);
nValGeometry = min(300, numel(idVal));
valGeometry = idVal(randperm(numel(idVal), nValGeometry));

[vg, vf] = ndgrid(valGeometry, valFrequency);
[XVal, YVal, AVal] = makeBatch( ...
    vg(:), vf(:), G, f, RAW, ALF, ...
    Xmin, Xrange, b_soft, Ymin, Yrange);
clear vg vf;

%% 9. 训练

fprintf('\nStep 5: Training...\n');

trainLossHistory = zeros(numEpochs, 1);
valLossHistory = zeros(numEpochs, 1);
averageGrad = [];
averageSqGrad = [];
iteration = 0;
bestValLoss = inf;
bestNet = net;

for epoch = 1:numEpochs
    epochStart = tic;
    learnRate = initialLearnRate * 0.5 ^ floor((epoch - 1) / 20);

    [trainG, trainF] = sampleTrainPoints( ...
        idTrain, peakForward, peakBackward, Nf, freqPerGeometry);

    numBatches = ceil(numel(trainG) / miniBatchSize);
    epochLoss = 0;

    for batch = 1:numBatches
        first = (batch - 1) * miniBatchSize + 1;
        last = min(batch * miniBatchSize, numel(trainG));

        [XBatch, YBatch, ABatch] = makeBatch( ...
            trainG(first:last), trainF(first:last), ...
            G, f, RAW, ALF, ...
            Xmin, Xrange, b_soft, Ymin, Yrange);

        dlX = dlarray(XBatch', 'CB');
        dlY = dlarray(YBatch', 'CB');
        dlA = dlarray(ABatch', 'CB');

        [loss, gradients] = dlfeval( ...
            @modelGradients, net, dlX, dlY, dlA, ...
            Ymin, Yrange, lambdaAlpha);

        iteration = iteration + 1;
        [net, averageGrad, averageSqGrad] = adamupdate( ...
            net, gradients, averageGrad, averageSqGrad, ...
            iteration, learnRate);

        lossValue = double(gather(extractdata(loss)));
        epochLoss = epochLoss + lossValue * (last - first + 1);
    end

    trainLossHistory(epoch) = epochLoss / numel(trainG);
    valLossHistory(epoch) = evaluateLoss( ...
        net, XVal, YVal, AVal, Ymin, Yrange, lambdaAlpha, miniBatchSize);

    if valLossHistory(epoch) < bestValLoss
        bestValLoss = valLossHistory(epoch);
        bestNet = net;
        save('Origami_FPN_Best.mat', ...
            'bestNet', 'bestValLoss', ...
            'Xmin', 'Xrange', 'b_soft', 'Ymin', 'Yrange', ...
            'lb', 'ub', 'f', 'Z0', 'lambdaAlpha', ...
            '-v7.3');
    end

    fprintf(['Epoch %3d/%3d | Train %.6e | Val %.6e | ' ...
        'LR %.2e | %.1f s\n'], ...
        epoch, numEpochs, ...
        trainLossHistory(epoch), valLossHistory(epoch), ...
        learnRate, toc(epochStart));
end

net = bestNet;

figure('Color', 'w', 'Position', [200 200 850 500]);
semilogy(1:numEpochs, trainLossHistory, 'b-', 'LineWidth', 1.8);
hold on;
semilogy(1:numEpochs, valLossHistory, 'r-', 'LineWidth', 1.8);
xlabel('Epoch');
ylabel('Loss');
legend('Training', 'Validation', 'Location', 'best');
title('Matrix loss + absorption loss');
grid on;
box on;
savefig('FPN_Loss.fig');

%% 10. 测试几何的声学曲线

fprintf('\nStep 6: Acoustic comparison...\n');

for sampleID = 1:3
    geomID = idTest(sampleID);
    geometryIDs = repmat(geomID, Nf, 1);
    frequencyIDs = (1:Nf)';

    [XSample, ~, ~] = makeBatch( ...
        geometryIDs, frequencyIDs, G, f, RAW, ALF, ...
        Xmin, Xrange, b_soft, Ymin, Yrange);

    YSamplePred = predictBatches(net, XSample, miniBatchSize);
    YPhysicalPred = double(YSamplePred) .* Yrange + Ymin;

    YPhysicalTrue = zeros(Nf, 8);
    YPhysicalSoft = zeros(Nf, 8);
    for k = 1:8
        YPhysicalTrue(:, k) = double(RAW{k}(geomID, :))';
        YPhysicalSoft(:, k) = softClipping(YPhysicalTrue(:, k), b_soft(k));
    end

    [Af, Rf, Tf, Ab, Rb, Tb] = acousticResponse(YPhysicalTrue);
    [AfSoft, RfSoft, TfSoft, AbSoft, RbSoft, TbSoft] = ...
        acousticResponse(YPhysicalSoft);
    [AfPred, RfPred, TfPred, AbPred, RbPred, TbPred] = ...
        acousticResponse(YPhysicalPred);

    fig = figure('Color', 'w', 'Position', [80 80 1400 750]);
    tl = tiledlayout(2, 3, 'TileSpacing', 'compact');

    trueData = {Af, Rf, Tf, Ab, Rb, Tb};
    softData = {AfSoft, RfSoft, TfSoft, AbSoft, RbSoft, TbSoft};
    predData = {AfPred, RfPred, TfPred, AbPred, RbPred, TbPred};
    plotNames = { ...
        'Forward absorption', 'Forward reflection', 'Forward transmission', ...
        'Backward absorption', 'Backward reflection', 'Backward transmission'};

    for j = 1:6
        nexttile(tl);
        plot(f, trueData{j}, 'k-', 'LineWidth', 1.8);
        hold on;
        plot(f, softData{j}, 'b--', 'LineWidth', 1.4);
        plot(f, predData{j}, 'r-.', 'LineWidth', 1.5);
        xlabel('Frequency (Hz)');
        ylabel('Coefficient');
        title(plotNames{j});
        ylim([-0.2, 1.2]);
        legend('Original TMM', 'Soft-clipped TMM', 'FPN prediction', ...
            'Location', 'best');
        grid on;
        box on;
    end

    title(tl, sprintf('Independent Test Geometry %d', geomID));
    savefig(fig, sprintf('FPN_Test_Geometry_%d.fig', sampleID));
end

save('Origami_FPN_Final_Results.mat', ...
    'net', 'trainLossHistory', 'valLossHistory', ...
    'idTrain', 'idVal', 'idTest', ...
    'Xmin', 'Xrange', 'Ymin', 'Yrange', ...
    'b_soft', 'lambdaAlpha', ...
    '-v7.3');

fprintf('\nAll calculations completed.\n');

%% 局部函数

function [result, Af, Ab] = calculateTMMBatch( ...
    G, H, omega, k0, rho0, eta, Z0, S, t, n_holes)

ng = size(G, 1);
nf = numel(omega);

omega = reshape(omega, 1, []);
k0 = reshape(k0, 1, []);

d = G(:, 1:3) * 1e-3;
H1 = H(:, 1);
H2 = H(:, 2);
phi = pi * d .^ 2 .* n_holes / (4 * S);

Zp = cell(1, 3);
for j = 1:3
    dj = d(:, j);
    phij = phi(:, j);
    y = (dj / 2) ./ sqrt(eta ./ (rho0 * omega));
    qm = y * sqrt(-1i);
    F = 1 - 2 * besselj(1, qm) ./ (qm .* besselj(0, qm));

    Zmass = 1i * rho0 * t(j) .* (omega ./ (phij .* F));
    Zviscous = sqrt(2) * eta .* (y ./ (phij .* dj));
    Zend = 1i * 0.85 * rho0 .* ((dj ./ phij) .* omega);
    Zp{j} = Zmass + Zviscous + Zend;
end

A = ones(ng, nf);
B = Zp{1};
C = zeros(ng, nf);
D = ones(ng, nf);

co = cos(H1 * k0);
si = sin(H1 * k0);
[A, B, C, D] = multiplyT(A, B, C, D, co, 1i * Z0 * si, 1i * si / Z0, co);
[A, B, C, D] = multiplyT(A, B, C, D, ones(ng, nf), Zp{2}, zeros(ng, nf), ones(ng, nf));

co = cos(H2 * k0);
si = sin(H2 * k0);
[A, B, C, D] = multiplyT(A, B, C, D, co, 1i * Z0 * si, 1i * si / Z0, co);
[A, B, C, D] = multiplyT(A, B, C, D, ones(ng, nf), Zp{3}, zeros(ng, nf), ones(ng, nf));

BZ = B / Z0;
CZ = C * Z0;

result = { ...
    real(A), imag(A), ...
    real(BZ), imag(BZ), ...
    real(CZ), imag(CZ), ...
    real(D), imag(D)};

Y = [real(A(:)), imag(A(:)), real(BZ(:)), imag(BZ(:)), ...
    real(CZ(:)), imag(CZ(:)), real(D(:)), imag(D(:))];
[Af, ~, ~, Ab] = acousticResponse(Y);
Af = reshape(Af, ng, nf);
Ab = reshape(Ab, ng, nf);
end

function [A, B, C, D] = multiplyT(A, B, C, D, a, b, c, d)
Anew = A .* a + B .* c;
Bnew = A .* b + B .* d;
Cnew = C .* a + D .* c;
Dnew = C .* b + D .* d;
A = Anew;
B = Bnew;
C = Cnew;
D = Dnew;
end

function Y = softClipping(X, b)
Y = X;
pos = X > b;
neg = X < -b;
xp = X(pos);
xn = X(neg);
Y(pos) = b + 2 * b .* (1 ./ (1 + exp(-(xp - b) ./ (2 * b))) - 0.5);
Y(neg) = -b - 2 * b .* (1 ./ (1 + exp((xn + b) ./ (2 * b))) - 0.5);
end

function [X, Y, Alpha] = makeBatch( ...
    geometryIDs, frequencyIDs, G, f, RAW, ALF, ...
    Xmin, Xrange, b_soft, Ymin, Yrange)

geometryIDs = geometryIDs(:);
frequencyIDs = frequencyIDs(:);
nb = numel(geometryIDs);
Ns = size(G, 1);

X = zeros(nb, 6, 'single');
X(:, 1:5) = G(geometryIDs, :);
X(:, 6) = single(reshape(f(frequencyIDs), [], 1));
X = single((double(X) - Xmin) ./ Xrange);

linearIDs = geometryIDs + (frequencyIDs - 1) * Ns;
Y = zeros(nb, 8, 'single');

for k = 1:8
    values = softClipping(double(RAW{k}(linearIDs)), b_soft(k));
    values = (values - Ymin(k)) / Yrange(k);
    Y(:, k) = single(values);
end

Alpha = zeros(nb, 2, 'single');
A1 = ALF(:, :, 1);
A2 = ALF(:, :, 2);
Alpha(:, 1) = A1(linearIDs);
Alpha(:, 2) = A2(linearIDs);
end

function [loss, gradients] = modelGradients( ...
    net, dlX, dlY, dlAlpha, yMin, yRange, lambdaAlpha)

predN = forward(net, dlX);
matrixLoss = mean((predN - dlY) .^ 2, 'all');

scale = dlarray(yRange(:), 'C');
shift = dlarray(yMin(:), 'C');
phys = predN .* scale + shift;
[af, ab] = alphaRows(phys);
ea2 = min((af - dlAlpha(1, :)) .^ 2, 1);
eb2 = min((ab - dlAlpha(2, :)) .^ 2, 1);
alphaLoss = 0.5 * (mean(ea2, 'all') + mean(eb2, 'all'));

loss = matrixLoss + lambdaAlpha * alphaLoss;
gradients = dlgradient(loss, net.Learnables);
end

function [af, ab] = alphaRows(P)
Ar = P(1, :); Ai = P(2, :);
Br = P(3, :); Bi = P(4, :);
Cr = P(5, :); Ci = P(6, :);
Dr = P(7, :); Di = P(8, :);

denR = Ar + Br + Cr + Dr;
denI = Ai + Bi + Ci + Di;
den2 = denR .^ 2 + denI .^ 2 + 1e-8;

rfR = Ar + Br - Cr - Dr;
rfI = Ai + Bi - Ci - Di;
Rf = (rfR .^ 2 + rfI .^ 2) ./ den2;
Tf = 4 ./ den2;
af = 1 - Rf - Tf;

rbR = -Ar + Br - Cr + Dr;
rbI = -Ai + Bi - Ci + Di;
Rb = (rbR .^ 2 + rbI .^ 2) ./ den2;

detR = Ar .* Dr - Ai .* Di - (Br .* Cr - Bi .* Ci);
detI = Ar .* Di + Ai .* Dr - (Br .* Ci + Bi .* Cr);
Tb = 4 * (detR .^ 2 + detI .^ 2) ./ den2;
ab = 1 - Rb - Tb;
end

function lossValue = evaluateLoss( ...
    net, X, Y, Alpha, yMin, yRange, lambdaAlpha, batchSize)

N = size(X, 1);
totalLoss = 0;

for first = 1:batchSize:N
    last = min(first + batchSize - 1, N);
    dlX = dlarray(X(first:last, :)', 'CB');
    dlY = predict(net, dlX);
    predN = double(gather(extractdata(dlY)))';
    target = double(Y(first:last, :));
    phys = predN .* yRange + yMin;
    [af, ab] = acousticResponse(phys);

    matrixLoss = mean((predN - target) .^ 2, 'all');
    ea = af - double(Alpha(first:last, 1));
    eb = ab - double(Alpha(first:last, 2));
    alphaLoss = 0.5 * (mean(min(ea .^ 2, 1)) + mean(min(eb .^ 2, 1)));
    totalLoss = totalLoss + (matrixLoss + lambdaAlpha * alphaLoss) * (last - first + 1);
end

lossValue = totalLoss / N;
end

function YPred = predictBatches(net, X, batchSize)
N = size(X, 1);
YPred = zeros(N, 8, 'single');
for first = 1:batchSize:N
    last = min(first + batchSize - 1, N);
    dlX = dlarray(X(first:last, :)', 'CB');
    dlY = predict(net, dlX);
    YPred(first:last, :) = single(gather(extractdata(dlY)))';
end
end

function [Af, Rf, Tf, Ab, Rb, Tb] = acousticResponse(Y)
Ar = Y(:, 1); Ai = Y(:, 2);
Br = Y(:, 3); Bi = Y(:, 4);
Cr = Y(:, 5); Ci = Y(:, 6);
Dr = Y(:, 7); Di = Y(:, 8);

denR = Ar + Br + Cr + Dr;
denI = Ai + Bi + Ci + Di;
den2 = denR .^ 2 + denI .^ 2 + 1e-8;

rfR = Ar + Br - Cr - Dr;
rfI = Ai + Bi - Ci - Di;
Rf = (rfR .^ 2 + rfI .^ 2) ./ den2;
Tf = 4 ./ den2;
Af = 1 - Rf - Tf;

rbR = -Ar + Br - Cr + Dr;
rbI = -Ai + Bi - Ci + Di;
Rb = (rbR .^ 2 + rbI .^ 2) ./ den2;

detR = Ar .* Dr - Ai .* Di - (Br .* Cr - Bi .* Ci);
detI = Ar .* Di + Ai .* Dr - (Br .* Ci + Bi .* Cr);
Tb = 4 * (detR .^ 2 + detI .^ 2) ./ den2;
Ab = 1 - Rb - Tb;
end

function [trainG, trainF] = sampleTrainPoints( ...
    idTrain, peakForward, peakBackward, Nf, freqPerGeometry)

idTrain = idTrain(:);
nTrain = numel(idTrain);
nFixed = 6;
nRand = freqPerGeometry - nFixed;

fixed = zeros(nTrain, nFixed);
fixed(:, 1:3) = peakNeighbors(peakForward(idTrain), Nf);
fixed(:, 4:6) = peakNeighbors(peakBackward(idTrain), Nf);
picked = [fixed, randi(Nf, nTrain, nRand)];

trainF = picked';
trainF = trainF(:);
trainG = repelem(idTrain, freqPerGeometry);
end

function ids = peakNeighbors(ip, Nf)
ip = ip(:);
ids = [min(max(ip - 1, 1), Nf), ip, min(max(ip + 1, 1), Nf)];
end
