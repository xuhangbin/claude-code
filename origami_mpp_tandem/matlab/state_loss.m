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
