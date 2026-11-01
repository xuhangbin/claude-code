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
