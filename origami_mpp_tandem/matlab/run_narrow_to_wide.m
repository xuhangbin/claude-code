clear;
clc;
close all;

% 逆网络的输入是声学状态，不是一条吸声曲线。
% 状态写成一行四个数：[中心频率Hz, 带宽目标Hz, 最小吸声系数, 带宽类型]
% 带宽类型 -1：带宽要低于目标（窄带）
% 带宽类型 +1：带宽要高于目标（宽带）
%
% 带宽定义为：正向吸声系数不低于最小吸声系数的连续频率宽度。
% 下面两个目标落在题目要求的内部：
%   状态1 窄带：中心 400 Hz，带宽目标 40 Hz，因此会小于 50 Hz，峰值至少 0.5
%   状态2 宽带：中心 400 Hz，带宽目标 230 Hz，因此会大于 200 Hz，峰值至少 0.5

% 需要 MATLAB R2020b 或更新版本（用到 pagemtimes）。
% 若只把本脚本另存到别的文件夹，stinson_fluid 会找不到。
% 请改运行同目录下的 DualState_ATNN_28vars.m，那个文件自带全部子函数。
thisDir = fileparts(mfilename('fullpath'));
if ~isfile(fullfile(thisDir, 'stinson_fluid.m'))
    error(['找不到 stinson_fluid.m。请运行 DualState_ATNN_28vars.m，或把整个 matlab 文件夹设为当前文件夹。当前脚本目录：' thisDir]);
end
addpath(thisDir);

freq = (100:2:700).';
fluid = stinson_fluid(freq);

state = [
    400, 40,  0.5, -1
    400, 230, 0.5,  1
];
stateName = {'状态1 窄带', '状态2 宽带'};

model = train_state_tandem(freq, fluid, 1200);
save(fullfile(thisDir, 'state_tandem.mat'), 'model', 'state');

alpha = zeros(2, numel(freq));
found = zeros(2, 20);
for i = 1:2
    fprintf('%s\n', stateName{i});
    guess = propose_state(model, state(i, :));
    found(i, :) = refine_state(state(i, :), guess, freq, fluid);
    alpha(i, :) = mpp_absorption(found(i, :), freq, fluid);
    [fPeak, bandwidth, aPeak] = measure_state(freq, alpha(i, :), state(i, 3));
    fprintf('  峰值 %.1f Hz，带宽 %.1f Hz，峰值吸声 %.3f\n', fPeak, bandwidth, aPeak);
end

figure('Color', 'w', 'Position', [120 80 860 640]);
for i = 1:2
    subplot(2, 1, i);
    plot(freq, alpha(i, :), 'b-', 'LineWidth', 1.8);
    hold on;
    yline(state(i, 3), 'k--', 'LineWidth', 1.0);
    xline(state(i, 1), 'r--', 'LineWidth', 1.0);
    [fPeak, bandwidth, aPeak] = measure_state(freq, alpha(i, :), state(i, 3));
    title(sprintf('%s   峰值 %.0f Hz   带宽 %.0f Hz   \\alpha_{peak}=%.2f', ...
        stateName{i}, fPeak, bandwidth, aPeak), 'FontWeight', 'normal');
    xlabel('Frequency/Hz');
    ylabel('Absorption');
    xlim([freq(1) freq(end)]);
    ylim([0 1]);
    grid on;
    box on;
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14, 'LineWidth', 1.1, 'TickDir', 'in');
end

saveas(gcf, fullfile(thisDir, 'narrow_to_wide.png'));
save(fullfile(thisDir, 'narrow_to_wide_design.mat'), 'found', 'alpha', 'freq', 'state');
