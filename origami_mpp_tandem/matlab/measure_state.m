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
