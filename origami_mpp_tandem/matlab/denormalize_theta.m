function theta = denormalize_theta(unit)
[low, high] = parameter_bounds();
theta = low + unit .* (high - low);
end
