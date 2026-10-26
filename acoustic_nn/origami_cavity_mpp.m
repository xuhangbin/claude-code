clear;
clc;
close all;

%% =========================================================
% 1. 频率
% ==========================================================

f = 100:0.1:600;
omega = 2*pi*f;

%% =========================================================
% 2. 空气参数
% ==========================================================

rho_0 = 1.29;
c0 = 343;
eta_0 = 1.89e-5;

Z_0 = rho_0*c0;

%% =========================================================
% 3. 微穿孔板参数
% ==========================================================

% 厚度
t1 = 2e-3;
t2 = 2e-3;
t3 = 2e-3;

% 孔径
d1 = 2.75e-3;
d2 = 1.73e-3;
d3 = 0.42e-3;

% 孔数
n1 = 1;
n2 = 1;
n3 = 1;

%% =========================================================
% 4. 截面积
% ==========================================================

W = 30e-3;
L = 30e-3;

S0 = W*L;

%% =========================================================
% 5. 折纸腔参数
% ==========================================================

% 两个折纸腔的实际高度（m）。每个腔的允许范围是 0–54 mm。
H1 = 54e-3;
H2 = 42e-3;

% 折纸几何参数（mm）
a = 30;
hc = 20;
n0 = 3;
beta = 45;
Hmax_mm = 54;

%% =========================================================
% 6. 折纸腔体积与等效高度
%    直接在主程序中计算
% ==========================================================

% 实际高度转换为 mm
H1_mm = H1*1e3;
H2_mm = H2*1e3;

% 检查折纸高度范围
if H1_mm <= 0 || H1_mm > Hmax_mm || ...
   H2_mm <= 0 || H2_mm > Hmax_mm

    error('折纸腔实际高度必须满足 0 < H <= %.2f mm', Hmax_mm);

end

% ---------------------------------------------------------
% 第一折纸腔
% ---------------------------------------------------------

h1 = H1_mm/n0;

q1 = sqrt(hc^2-h1^2);

V11 = a*h1*q1/16;

V12 = h1*(a-hc)*q1/(16*tand(beta));

V13 = h1*q1/24*(hc/tand(beta)-q1/2);

% 第一折纸腔总体积（mm^3）
V1 = 8*n0*(a^2*h1/8-V11-V12-V13);

% ---------------------------------------------------------
% 第二折纸腔
% ---------------------------------------------------------

h2 = H2_mm/n0;

q2 = sqrt(hc^2-h2^2);

V21 = a*h2*q2/16;

V22 = h2*(a-hc)*q2/(16*tand(beta));

V23 = h2*q2/24*(hc/tand(beta)-q2/2);

% 第二折纸腔总体积（mm^3）
V2 = 8*n0*(a^2*h2/8-V21-V22-V23);

% ---------------------------------------------------------
% 计算等效高度
% ---------------------------------------------------------

S_mm2 = W*L*1e6;

H1_eq_mm = V1/S_mm2;
H2_eq_mm = V2/S_mm2;

% 转换为 m
H1_eq = H1_eq_mm*1e-3;
H2_eq = H2_eq_mm*1e-3;

fprintf('\n========== 折纸腔几何参数 ==========\n');

fprintf('第一腔实际高度 H1 = %.4f mm\n',H1_mm);
fprintf('第一腔体积 V1 = %.4f mm^3\n',V1);
fprintf('第一腔等效高度 H1_eq = %.4f mm\n\n',H1_eq_mm);

fprintf('第二腔实际高度 H2 = %.4f mm\n',H2_mm);
fprintf('第二腔体积 V2 = %.4f mm^3\n',V2);
fprintf('第二腔等效高度 H2_eq = %.4f mm\n',H2_eq_mm);

%% =========================================================
% 7. 穿孔率
% ==========================================================

phi1 = pi*d1^2*n1/(4*W*L);
phi2 = pi*d2^2*n2/(4*W*L);
phi3 = pi*d3^2*n3/(4*W*L);

%% =========================================================
% 8. 总长度
% ==========================================================

% 保留原代码的实际几何长度作为透射相位参考
L_total = t1+H1+t2+H2+t3;

%% =========================================================
% 9. 初始化
% ==========================================================

T11 = zeros(size(f));
T12 = zeros(size(f));
T21 = zeros(size(f));
T22 = zeros(size(f));

tau = zeros(size(f));

rf = zeros(size(f));
rb = zeros(size(f));

alpha_f = zeros(size(f));
alpha_b = zeros(size(f));

%% =========================================================
% 10. 频率循环
% ==========================================================

for ii = 1:length(f)

    w = omega(ii);

    k = w/c0;

    %% 微穿孔板参数

    y1 = (d1/2)/sqrt(eta_0/(rho_0*w));
    y2 = (d2/2)/sqrt(eta_0/(rho_0*w));
    y3 = (d3/2)/sqrt(eta_0/(rho_0*w));

    %% 第一块微穿孔板阻抗

    Z_p1 = ...
        1i*w*rho_0*t1/phi1 * ...
        ( ...
        1 - ...
        2*besselj(1,y1*sqrt(-1i)) / ...
        ( ...
        y1*sqrt(-1i) * ...
        besselj(0,y1*sqrt(-1i)) ...
        ) ...
        )^(-1) ...
        + sqrt(2)*eta_0*y1/(phi1*d1) ...
        + 1i*0.85*w*rho_0*d1/phi1;

    %% 第二块微穿孔板阻抗

    Z_p2 = ...
        1i*w*rho_0*t2/phi2 * ...
        ( ...
        1 - ...
        2*besselj(1,y2*sqrt(-1i)) / ...
        ( ...
        y2*sqrt(-1i) * ...
        besselj(0,y2*sqrt(-1i)) ...
        ) ...
        )^(-1) ...
        + sqrt(2)*eta_0*y2/(phi2*d2) ...
        + 1i*0.85*w*rho_0*d2/phi2;

    %% 第三块微穿孔板阻抗

    Z_p3 = ...
        1i*w*rho_0*t3/phi3 * ...
        ( ...
        1 - ...
        2*besselj(1,y3*sqrt(-1i)) / ...
        ( ...
        y3*sqrt(-1i) * ...
        besselj(0,y3*sqrt(-1i)) ...
        ) ...
        )^(-1) ...
        + sqrt(2)*eta_0*y3/(phi3*d3) ...
        + 1i*0.85*w*rho_0*d3/phi3;

    %% 微穿孔板传递矩阵

    T_p1 = [1, Z_p1;
            0, 1];

    T_p2 = [1, Z_p2;
            0, 1];

    T_p3 = [1, Z_p3;
            0, 1];

    %% =====================================================
    % 11. 第一折纸腔传递矩阵
    % ======================================================

    T_c1 = ...
        [cos(k*H1_eq),           1i*Z_0*sin(k*H1_eq);
         1i*sin(k*H1_eq)/Z_0,   cos(k*H1_eq)];

    %% =====================================================
    % 12. 第二折纸腔传递矩阵
    % ======================================================

    T_c2 = ...
        [cos(k*H2_eq),           1i*Z_0*sin(k*H2_eq);
         1i*sin(k*H2_eq)/Z_0,   cos(k*H2_eq)];

    %% =====================================================
    % 13. 总传递矩阵
    % ======================================================

    T_total = T_p1*T_c1*T_p2*T_c2*T_p3;

    T11(ii) = T_total(1,1);
    T12(ii) = T_total(1,2);
    T21(ii) = T_total(2,1);
    T22(ii) = T_total(2,2);

    %% =====================================================
    % 14. 共同分母
    % ======================================================

    D = ...
        T11(ii) ...
        + T12(ii)/Z_0 ...
        + T21(ii)*Z_0 ...
        + T22(ii);

    %% =====================================================
    % 15. 透射振幅系数
    % ======================================================

    tau(ii) = 2*exp(1i*k*L_total)/D;

    %% =====================================================
    % 16. 正向反射振幅系数
    % ======================================================

    rf(ii) = ...
        ( ...
          T11(ii) ...
        + T12(ii)/Z_0 ...
        - T21(ii)*Z_0 ...
        - T22(ii) ...
        )/D;

    %% =====================================================
    % 17. 反向反射振幅系数
    % ======================================================

    rb(ii) = ...
        ( ...
         -T11(ii) ...
        + T12(ii)/Z_0 ...
        - T21(ii)*Z_0 ...
        + T22(ii) ...
        )/D;

    %% =====================================================
    % 18. 正反向吸声系数
    % ======================================================

    alpha_f(ii) = ...
        1-abs(rf(ii))^2-abs(tau(ii))^2;

    alpha_b(ii) = ...
        1-abs(rb(ii))^2-abs(tau(ii))^2;

end

%% =========================================================
% 19. 反射率和透射率
% ==========================================================

R_f = abs(rf).^2;
R_b = abs(rb).^2;

T = abs(tau).^2;

%% =========================================================
% 20. 能量守恒检查
% ==========================================================

energy_forward = alpha_f+R_f+T;
energy_backward = alpha_b+R_b+T;

fprintf('\n========== 能量守恒检查 ==========\n');

fprintf('正向最大误差 = %.4e\n', ...
    max(abs(energy_forward-1)));

fprintf('反向最大误差 = %.4e\n', ...
    max(abs(energy_backward-1)));

%% =========================================================
% 21. 散射矩阵特征值
%
% S = [tau rb;
%      rf  tau]
%
% lambda+ = tau + sqrt(rf*rb)
% lambda- = tau - sqrt(rf*rb)
% ==========================================================

lambda_plus = zeros(size(f));
lambda_minus = zeros(size(f));

for ii = 1:length(f)

    delta = sqrt(rf(ii)*rb(ii));

    lambda_plus(ii) = tau(ii)+delta;

    lambda_minus(ii) = tau(ii)-delta;

end

%% =========================================================
% 22. 特征值实部、虚部及差值
% ==========================================================

Re_plus = real(lambda_plus);
Re_minus = real(lambda_minus);

Im_plus = imag(lambda_plus);
Im_minus = imag(lambda_minus);

EP_diff = abs(lambda_plus-lambda_minus);

%% =========================================================
% 23. 寻找实频率范围内特征值差最小点
% ==========================================================

[min_diff,idx] = min(EP_diff);

f_EP_candidate = f(idx);

fprintf('\n========== EP候选点 ==========\n');

fprintf('候选频率 = %.4f Hz\n',f_EP_candidate);

fprintf('特征值差 = %.8e\n',min_diff);

fprintf('|rf| = %.8e\n',abs(rf(idx)));

fprintf('|rb| = %.8e\n',abs(rb(idx)));

fprintf('lambda+ = %.6f%+.6fi\n', ...
    real(lambda_plus(idx)),imag(lambda_plus(idx)));

fprintf('lambda- = %.6f%+.6fi\n', ...
    real(lambda_minus(idx)),imag(lambda_minus(idx)));

fprintf('注意：特征值差最小不代表已证明存在EP。\n');

%% =========================================================
% 24. 绘图
% ==========================================================

figure('Color','w','Position',[160 60 1050 900]);

% ---------------------------------------------------------
% (a) 正向入射
% ---------------------------------------------------------

subplot(3,1,1);

plot(f,alpha_f,'b-','LineWidth',1.6);
hold on;

plot(f,R_f,'r--','LineWidth',1.6);

plot(f,T,'k:','LineWidth',1.6);

title('(a) Forward incidence');
ylabel('Coefficient');

legend('Absorption','Reflection','Transmission', ...
    'Location','northeast');

xlim([min(f),max(f)]);
ylim([0 1]);

grid on;
box on;

% ---------------------------------------------------------
% (b) 反向入射
% ---------------------------------------------------------

subplot(3,1,2);

plot(f,alpha_b,'b-','LineWidth',1.6);
hold on;

plot(f,R_b,'r--','LineWidth',1.6);

plot(f,T,'k:','LineWidth',1.6);

title('(b) Backward incidence');
ylabel('Coefficient');

legend('Absorption','Reflection','Transmission', ...
    'Location','northeast');

xlim([min(f),max(f)]);
ylim([0 1]);

grid on;
box on;

% ---------------------------------------------------------
% (c) 散射矩阵特征值
% ---------------------------------------------------------

subplot(3,1,3);

plot(f,Re_plus,'r-','LineWidth',1.4);
hold on;

plot(f,Re_minus,'b-','LineWidth',1.4);

plot(f,Im_plus,'m--','LineWidth',1.4);

plot(f,Im_minus,'c--','LineWidth',1.4);

plot([min(f),max(f)],[0 0],'k:');

title('(c) Scattering eigenvalues');

xlabel('Frequency (Hz)');
ylabel('Eigenvalue');

legend('Re(\lambda_+)','Re(\lambda_-)', ...
       'Im(\lambda_+)','Im(\lambda_-)', ...
       'Location','northeast');

xlim([min(f),max(f)]);

grid on;
box on;

%% =========================================================
% 25. 保存计算结果
% ==========================================================

outDir = fileparts(mfilename('fullpath'));
if isempty(outDir)
    outDir = pwd;
end
figPath = fullfile(outDir, 'origami_cavity_mpp.png');
print(gcf, figPath, '-dpng', '-r120');
fprintf('\n图已保存到 %s\n', figPath);

csvPath = fullfile(outDir, 'origami_cavity_mpp.csv');
if exist('table', 'file') == 2 || exist('table', 'builtin')
    Results = table( ...
        f(:), ...
        alpha_f(:),R_f(:),T(:), ...
        alpha_b(:),R_b(:), ...
        real(lambda_plus(:)),imag(lambda_plus(:)), ...
        real(lambda_minus(:)),imag(lambda_minus(:)), ...
        EP_diff(:), ...
        'VariableNames', { ...
        'Frequency_Hz', ...
        'Absorption_Forward', ...
        'Reflection_Forward', ...
        'Transmission', ...
        'Absorption_Backward', ...
        'Reflection_Backward', ...
        'Re_LambdaPlus', ...
        'Im_LambdaPlus', ...
        'Re_LambdaMinus', ...
        'Im_LambdaMinus', ...
        'EigenvalueDifference'});
    writetable(Results, csvPath);
else
    fid = fopen(csvPath, 'w');
    fprintf(fid, ['Frequency_Hz,Absorption_Forward,Reflection_Forward,' ...
        'Transmission,Absorption_Backward,Reflection_Backward,' ...
        'Re_LambdaPlus,Im_LambdaPlus,Re_LambdaMinus,Im_LambdaMinus,' ...
        'EigenvalueDifference\n']);
    fprintf(fid, '%.4f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8e\n', ...
        [f(:), alpha_f(:), R_f(:), T(:), alpha_b(:), R_b(:), ...
         real(lambda_plus(:)), imag(lambda_plus(:)), ...
         real(lambda_minus(:)), imag(lambda_minus(:)), EP_diff(:)].');
    fclose(fid);
end

fprintf('\n计算完成。\n');
