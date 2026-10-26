% 训练正向网络，并画出四联精度图。
%
% (a) 训练集、验证集的均方误差随训练轮数变化
% (b) 三个数据集的平均相对误差分布
% (c) 验证集：网络预测值对传递矩阵真值
% (d) 测试集：网络预测值对传递矩阵真值
%
% 在 MATLAB 里把当前文件夹设为 acoustic_nn，然后运行本脚本。
% 不搜索孔径和腔深。神经网络本体在 origami_dual_mlp.m。

clear;
clc;
close all;

setenv('RUN_MODE', 'accuracy');
origami_dual_mlp;

fprintf('图：origami_nn_accuracy.png\n');
fprintf('Origin 数据：nn_accuracy_loss.csv，nn_accuracy_hist.csv，nn_accuracy_val.csv，nn_accuracy_test.csv\n');
