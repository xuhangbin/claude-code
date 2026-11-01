function [low, high] = parameter_bounds()
const = acoustic_constants();
low = [repmat(const.dMin, 1, 12), repmat(const.hMin, 1, 8)];
high = [repmat(const.dMax, 1, 12), repmat(const.hMax, 1, 8)];
end
