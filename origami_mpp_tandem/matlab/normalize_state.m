function x = normalize_state(state)
% [f_Hz, bandwidth_Hz, alpha_min, band_type] -> inverse-network input.
x = [(state(:, 1) - 100) / 600, state(:, 2) / 600, state(:, 3), state(:, 4)];
end
