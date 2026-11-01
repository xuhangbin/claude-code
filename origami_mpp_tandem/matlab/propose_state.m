function theta = propose_state(model, state)
%PROPOSE_STATE  Inverse network: acoustic state -> geometry (metres).
unit = mlp_apply(model.inverse, normalize_state(state));
theta = denormalize_theta(unit);
end

function y = mlp_apply(net, x)
h = x;
for layer = 1:numel(net.W) - 1
    z = h * net.W{layer} + net.b{layer};
    sig = 1 ./ (1 + exp(-z));
    h = z .* sig;
end
z = h * net.W{end} + net.b{end};
y = 1 ./ (1 + exp(-z));
end
