function theta = refine_state(state, theta0, freq, fluid)
%REFINE_STATE  Differential evolution on the transfer-matrix model.
%   THETA0 from the inverse network is inserted into the initial population.
%   The cost is STATE_LOSS, so the search enforces the acoustic state.

[low, high] = parameter_bounds();
nParam = 20;
popSize = 36;
generations = 22;
population = low + rand(popSize, nParam) .* (high - low);
if ~isempty(theta0)
    population(1, :) = min(max(theta0, low), high);
end

cost = evaluate(population);
[bestCost, bestIdx] = min(cost);
best = population(bestIdx, :);
fprintf('  initial state loss %.4f\n', bestCost);

for generation = 1:generations
    trial = population;
    for i = 1:popSize
        ids = randperm(popSize, 3);
        mutant = population(ids(1), :) + 0.6 * (population(ids(2), :) - population(ids(3), :));
        cross = rand(1, nParam) < 0.7;
        cross(randi(nParam)) = true;
        candidate = population(i, :);
        candidate(cross) = mutant(cross);
        candidate = min(max(candidate, low), high);
        trial(i, :) = candidate;
    end
    trialCost = evaluate(trial);
    better = trialCost < cost;
    population(better, :) = trial(better, :);
    cost(better) = trialCost(better);
    [genBest, genIdx] = min(cost);
    if genBest < bestCost
        bestCost = genBest;
        best = population(genIdx, :);
    end
end
fprintf('  refined state loss %.4f\n', bestCost);
theta = best;

    function value = evaluate(batch)
        spectrum = mpp_absorption(batch, freq, fluid);
        value = state_loss(freq, spectrum, state);
    end
end
