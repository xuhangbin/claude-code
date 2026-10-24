clear;
clc;
close all;

%% 1. 频率与空气参数

f = 100:1:500;
omega = 2*pi*f;
Nf = length(f);

rho0 = 1.29;
c0 = 343;
eta = 1.89e-5;
kappa = 0.026;
Cv = 712;
gamma = 1.4;

P0 = rho0*c0^2/gamma;
Z0 = rho0*c0;

nu = eta/rho0;
mu_bar = kappa/(rho0*Cv);

%% 2. 四个单胞结构参数

Nc = 4;


t = 1e-3 * [
    2 2 2;
    2 2 2;
    2 2 2;
    2 2 2
];

% 微穿孔板孔径，单位 m

d = 1e-3 * [
    2.43 1.6628 0.243;
    3.6 1.5 0.26;
    3.15 0.79 0.15;
    2.98 0.56 0.15
];

% 两个腔体深度，单位 m

H = 1e-3 * [
    47 47;
    47 47;
    47 47;
    47 47
];

% 单胞横截面尺寸

W = 30e-3;
L = 30e-3;

% 微穿孔板孔数

n_holes = [
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 1
];

% 穿孔率

phi = pi*d.^2.*n_holes/(4*W*L);

% 面积权重

area = W*L*ones(1,Nc);
weight = area/sum(area);

%% 3. 微穿孔板阻抗

Zp = zeros(Nc,3,Nf);

for s = 1:Nc

    for j = 1:3

        y = (d(s,j)/2)./sqrt(eta./(rho0*omega));

        q = y*sqrt(-1i);

        F = 1-2*besselj(1,q)./ ...
            (q.*besselj(0,q));

        Z_temp = ...
            1i*omega*rho0*t(s,j)./ ...
            (phi(s,j).*F) ...
            + sqrt(2)*eta*y./ ...
            (phi(s,j)*d(s,j)) ...
            + 1i*0.85*omega*rho0*d(s,j)/phi(s,j);

        Zp(s,j,:) = reshape(Z_temp,1,1,Nf);

    end

end

%% 4. Full Stinson Model

N = 200;

Sv = zeros(size(omega));
St = zeros(size(omega));

for m = 0:N

    am = (2*m+1)*pi/W;

    for n = 0:N

        bn = (2*n+1)*pi/L;

        lambda2 = am^2+bn^2;
        wmn = 1/(am^2*bn^2);

        Sv = Sv + wmn*lambda2./ ...
            (lambda2+1i*omega/nu);

        St = St + wmn*lambda2./ ...
            (lambda2+1i*omega/mu_bar);

    end

end

factor = 64/(W^2*L^2);

Fv = factor*Sv;
Ft = factor*St;

rho_eff = rho0./(1-Fv);

K_eff = gamma*P0./ ...
    (1+(gamma-1)*Ft);

k_eff = omega.*sqrt(rho_eff./K_eff);
Z_eff = sqrt(rho_eff.*K_eff);

%% 5. 四个单胞传递矩阵

T_cell = zeros(2,2,Nc,Nf);

for s = 1:Nc

    for ii = 1:Nf

        Tp1 = [1,Zp(s,1,ii);0,1];
        Tp2 = [1,Zp(s,2,ii);0,1];
        Tp3 = [1,Zp(s,3,ii);0,1];

        k = k_eff(ii);
        Z = Z_eff(ii);

        Tc1 = [
            cos(k*H(s,1)), ...
            1i*Z*sin(k*H(s,1));
            1i*sin(k*H(s,1))/Z, ...
            cos(k*H(s,1))
        ];

        Tc2 = [
            cos(k*H(s,2)), ...
            1i*Z*sin(k*H(s,2));
            1i*sin(k*H(s,2))/Z, ...
            cos(k*H(s,2))
        ];

        Tm = Tp1*Tc1*Tp2*Tc2*Tp3;

        T_cell(:,:,s,ii) = Tm;

    end

end

%% 6. 四单胞并联

T_total = zeros(2,2,Nf);

tau = zeros(1,Nf);
rf = zeros(1,Nf);
rb = zeros(1,Nf);

for ii = 1:Nf

    Y_total = zeros(2,2);

    for s = 1:Nc

        Tm = T_cell(:,:,s,ii);

        A = Tm(1,1);
        B = Tm(1,2);
        C = Tm(2,1);
        D = Tm(2,2);

        % 单胞导纳矩阵

        Ys = [
            D/B, C-A*D/B;
            1/B, -A/B
        ];

        % 面积加权并联

        Y_total = Y_total+weight(s)*Ys;

    end

    % 导纳矩阵恢复传递矩阵

    Y11 = Y_total(1,1);
    Y12 = Y_total(1,2);
    Y21 = Y_total(2,1);
    Y22 = Y_total(2,2);

    A = -Y22/Y21;
    B = 1/Y21;
    C = Y12-Y11*Y22/Y21;
    D = Y11/Y21;

    Tm = [A B;C D];

    T_total(:,:,ii) = Tm;

    % 散射参数
    % 适用于互易双端口

    den = A+B/Z0+C*Z0+D;

    tau(ii) = 2/den;

    rf(ii) = (A+B/Z0-C*Z0-D)/den;

    rb(ii) = (-A+B/Z0-C*Z0+D)/den;

end

%% 7. 正反向吸声

Rf = abs(rf).^2;
Rb = abs(rb).^2;
Tr = abs(tau).^2;

alpha_f = 1-Rf-Tr;
alpha_b = 1-Rb-Tr;

%% 8. 散射矩阵特征值

% 沿用原代码的排列方式：
%
% S = [tau rb;
%      rf  tau]
%
% 注意：这是重新排列的散射矩阵，
% 其特征值简并不等同于标准端口散射矩阵的EP。

delta = sqrt(rf.*rb);

lambda_plus = tau+delta;
lambda_minus = tau-delta;

EP_diff = abs(lambda_plus-lambda_minus);




%% 11. 四个相同单胞并联一致性验证

% 仅当四个单胞的参数完全相同时进行验证

same_t = all(all(abs(t-t(1,:))<1e-12));
same_d = all(all(abs(d-d(1,:))<1e-12));
same_H = all(all(abs(H-H(1,:))<1e-12));
same_n = all(all(n_holes==n_holes(1,:)));

if same_t && same_d && same_H && same_n

    max_error = 0;

    for ii = 1:Nf

        T_single = T_cell(:,:,1,ii);
        T_parallel = T_total(:,:,ii);

        error_now = max(abs(T_single(:)-T_parallel(:)));

        max_error = max(max_error,error_now);

    end

    fprintf('\n========== 并联一致性验证 ==========\n');
    fprintf('最大矩阵误差 = %.8e\n',max_error);

    if max_error < 1e-8
        fprintf('四单胞并联与单胞结果一致。\n');
    else
        warning('一致性验证未通过。');
    end

else

    fprintf('\n四单胞参数不同，跳过相同单胞验证。\n');

end

%% 12. 绘图

figure('Color','w','Position',[150 60 1050 900]);

subplot(2,1,1);

plot(f,alpha_f,'b-','LineWidth',1.7);
hold on;
plot(f,Rf,'r--','LineWidth',1.7);
plot(f,Tr,'k:','LineWidth',1.7);

title('(a) Forward incidence');
ylabel('Coefficient');

legend('Absorption','Reflection','Transmission', ...
    'Location','best');


grid on;
box on;

subplot(2,1,2);

plot(f,alpha_b,'b-','LineWidth',1.7);
hold on;
plot(f,Rb,'r--','LineWidth',1.7);
plot(f,Tr,'k:','LineWidth',1.7);

title('(b) Backward incidence');
ylabel('Coefficient');

legend('Absorption','Reflection','Transmission', ...
    'Location','best');

grid on;
box on;
