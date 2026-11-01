function unit = normalize_theta(theta)
[low, high] = parameter_bounds();
unit = (theta - low) ./ (high - low);
end
