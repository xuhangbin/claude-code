function model = train_state_tandem(freq, fluid, nSamples)
%TRAIN_STATE_TANDEM  Train the tandem pair whose inverse input is a state.
%   State columns are [f_center_Hz, bandwidth_Hz, alpha_min, band_type].
%   band_type = -1 narrow (bandwidth below the given value), +1 wide
%   (bandwidth above it), 0 match that bandwidth. The curve itself is never
%   the network input.

if nargin < 3
    nSamples = 1200;
end
rng(0);
freq = freq(:);
[low, high] = parameter_bounds();
theta = low + rand(nSamples, 20) .* (high - low);
alpha = zeros(nSamples, numel(freq));
fprintf('Generating %d transfer-matrix spectra...\n', nSamples);
for i0 = 1:64:nSamples
    i1 = min(nSamples, i0 + 63);
    alpha(i0:i1, :) = mpp_absorption(theta(i0:i1, :), freq, fluid);
end
unit = normalize_theta(theta);
[fPeak, bandwidth, aPeak] = measure_state(freq, alpha, 0.5);
keep = aPeak >= 0.55;
bandType = zeros(nSamples, 1);
bandType(bandwidth < 80) = -1;
bandType(bandwidth >= 80) = 1;
states = [fPeak, max(bandwidth, 2), 0.5 * ones(nSamples, 1), bandType];
states = states(keep, :);

model.forward = init_net(20, [96 96], numel(freq));
model.inverse = init_net(4, [96 96], 20);
model.freq = freq;

fprintf('Training the forward network...\n');
model.forward = train_regression(model.forward, unit, alpha, 40, 1e-3);
fprintf('Training the inverse network...\n');
model.inverse = train_inverse(model.inverse, model.forward, states, freq, 40, 1e-3);
end

function net = init_net(nIn, hidden, nOut)
sizes = [nIn, hidden, nOut];
net.W = cell(1, numel(sizes) - 1);
net.b = cell(1, numel(sizes) - 1);
net.mW = cell(1, numel(sizes) - 1);
net.mB = cell(1, numel(sizes) - 1);
net.vW = cell(1, numel(sizes) - 1);
net.vB = cell(1, numel(sizes) - 1);
for layer = 1:numel(sizes) - 1
    net.W{layer} = randn(sizes(layer), sizes(layer + 1)) * sqrt(2 / sizes(layer));
    net.b{layer} = zeros(1, sizes(layer + 1));
    net.mW{layer} = zeros(size(net.W{layer}));
    net.mB{layer} = zeros(size(net.b{layer}));
    net.vW{layer} = net.mW{layer};
    net.vB{layer} = net.mB{layer};
end
net.step = 0;
end

function net = train_regression(net, x, y, epochs, lr)
n = size(x, 1);
for epoch = 1:epochs
    order = randperm(n);
    total = 0;
    batches = 0;
    for i0 = 1:64:n
        idx = order(i0:min(n, i0 + 63));
        [pred, cache] = mlp_forward(net, x(idx, :));
        err = pred - y(idx, :);
        total = total + mean(err(:).^2);
        batches = batches + 1;
        grad = 2 * err / size(err, 2);
        net = mlp_update(net, cache, grad, lr);
    end
    if epoch == 1 || mod(epoch, 10) == 0
        fprintf('  forward epoch %d  mse %.5f\n', epoch, total / batches);
    end
end
end

function net = train_inverse(inverse, forward, states, freq, epochs, lr)
n = size(states, 1);
for epoch = 1:epochs
    order = randperm(n);
    total = 0;
    seen = 0;
    for i0 = 1:64:n
        idx = order(i0:min(n, i0 + 63));
        batchState = states(idx, :);
        [unit, cacheInv] = mlp_forward(inverse, normalize_state(batchState));
        [spectrum, cacheFwd] = mlp_forward(forward, unit);
        [loss, gradSpectrum] = soft_state_loss(freq, spectrum, batchState);
        gradUnit = mlp_grad_input(forward, cacheFwd, gradSpectrum);
        inverse = mlp_update(inverse, cacheInv, gradUnit, lr);
        total = total + sum(loss);
        seen = seen + numel(loss);
    end
    if epoch == 1 || mod(epoch, 10) == 0
        fprintf('  inverse epoch %d  state loss %.4f\n', epoch, total / max(seen, 1));
    end
end
net = inverse;
end

function [y, cache] = mlp_forward(net, x)
nLayer = numel(net.W);
cache.x = cell(1, nLayer);
cache.z = cell(1, nLayer);
h = x;
for layer = 1:nLayer - 1
    cache.x{layer} = h;
    z = h * net.W{layer} + net.b{layer};
    cache.z{layer} = z;
    sig = 1 ./ (1 + exp(-z));
    h = z .* sig;
end
cache.x{nLayer} = h;
z = h * net.W{nLayer} + net.b{nLayer};
cache.z{nLayer} = z;
y = 1 ./ (1 + exp(-z));
cache.y = y;
end

function gradX = mlp_grad_input(net, cache, gradY)
[gradX, ~] = backprop(net, cache, gradY);
end

function net = mlp_update(net, cache, gradY, lr)
[~, grads] = backprop(net, cache, gradY);
batch = size(gradY, 1);
net.step = net.step + 1;
b1 = 0.9;
b2 = 0.999;
for layer = 1:numel(net.W)
    gW = grads.W{layer} / batch;
    gB = grads.b{layer} / batch;
    net.mW{layer} = b1 * net.mW{layer} + (1 - b1) * gW;
    net.mB{layer} = b1 * net.mB{layer} + (1 - b1) * gB;
    net.vW{layer} = b2 * net.vW{layer} + (1 - b2) * gW.^2;
    net.vB{layer} = b2 * net.vB{layer} + (1 - b2) * gB.^2;
    mW = net.mW{layer} / (1 - b1^net.step);
    mB = net.mB{layer} / (1 - b1^net.step);
    vW = net.vW{layer} / (1 - b2^net.step);
    vB = net.vB{layer} / (1 - b2^net.step);
    net.W{layer} = net.W{layer} - lr * mW ./ (sqrt(vW) + 1e-8);
    net.b{layer} = net.b{layer} - lr * mB ./ (sqrt(vB) + 1e-8);
end
end

function [gradX, grads] = backprop(net, cache, gradY)
nLayer = numel(net.W);
grads.W = cell(1, nLayer);
grads.b = cell(1, nLayer);
delta = gradY .* cache.y .* (1 - cache.y);
for layer = nLayer:-1:1
    grads.W{layer} = cache.x{layer}.' * delta;
    grads.b{layer} = sum(delta, 1);
    gradX = delta * net.W{layer}.';
    if layer > 1
        z = cache.z{layer - 1};
        sig = 1 ./ (1 + exp(-z));
        delta = gradX .* (sig .* (1 + z .* (1 - sig)));
    end
end
end
