clear;
clc;
close all;

%% 1. 基本参数

f = 100:1:600;
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

W = 30e-3;
L = 30e-3;
S = W*L;

Nc = 4;

%% 2. 四单胞结构参数

% 三层微穿孔板厚度（m）
t = 2e-3*ones(Nc,3);

% 三层微穿孔板孔径（m）
d = 1e-3*repmat([2.75 1.73 0.42],Nc,1);

% 两个折纸腔实际高度（m）。神经网络搜索时每个腔取 0–54 mm。
H = 1e-3*47*ones(Nc,2);

% 穿孔数量
n_holes = ones(Nc,3);

% 穿孔率
phi = pi*d.^2.*n_holes/(4*S);

% 并联面积权重
weight = ones(1,Nc)/Nc;

%% 3. 折纸腔体积及等效高度

a = 30;       % mm
hc = 20;      % mm
n0 = 3;
beta = 45;    % degree

H_mm = H*1e3;
h = H_mm/n0;

if any(h(:)<=0) || any(h(:)>=hc)
    error('折纸腔高度超出几何模型允许范围。');
end

q = sqrt(hc^2-h.^2);

V11 = a.*h.*q/16;
V12 = h.*(a-hc).*q/(16*tand(beta));
V13 = h.*q/24.*(hc/tand(beta)-q/2);

V_mm3 = 8*n0.*(a^2.*h/8-V11-V12-V13);

if any(V_mm3(:)<=0)
    error('折纸腔体积出现非正值。');
end

% 等效腔高（m）
H_eq = V_mm3/(S*1e6)*1e-3;

fprintf('\n========== 折纸腔参数 ==========\n');

disp('实际腔高（mm）：');
disp(H_mm);

disp('等效腔高（mm）：');
disp(H_eq*1e3);

%% 4. 微穿孔板阻抗

Zp = complex(zeros(Nc,3,Nf));

for s = 1:Nc
    for j = 1:3

        y = (d(s,j)/2)./sqrt(eta./(rho0*omega));
        q = y*sqrt(-1i);

        F = 1-2*besselj(1,q)./(q.*besselj(0,q));

        Z = 1i*omega*rho0*t(s,j)./(phi(s,j).*F) ...
          + sqrt(2)*eta*y./(phi(s,j)*d(s,j)) ...
          + 1i*0.85*omega*rho0*d(s,j)/phi(s,j);

        Zp(s,j,:) = reshape(Z,1,1,Nf);
    end
end

%% 5. Full Stinson Model

N = 200;

Sv = zeros(1,Nf);
St = zeros(1,Nf);

for m = 0:N
    am = (2*m+1)*pi/W;

    for n = 0:N
        bn = (2*n+1)*pi/L;

        lambda2 = am^2+bn^2;
        wmn = 1/(am^2*bn^2);

        Sv = Sv + wmn*lambda2./(lambda2+1i*omega/nu);
        St = St + wmn*lambda2./(lambda2+1i*omega/mu_bar);
    end
end

Fv = 64*Sv/(W^2*L^2);
Ft = 64*St/(W^2*L^2);

rho_eff = rho0./(1-Fv);
K_eff = gamma*P0./(1+(gamma-1)*Ft);

k_eff = omega.*sqrt(rho_eff./K_eff);
Z_eff = sqrt(rho_eff.*K_eff);

%% 6. 四单胞传递矩阵及并联

rf = complex(zeros(1,Nf));
rb = complex(zeros(1,Nf));
tau = complex(zeros(1,Nf));

for ii = 1:Nf

    Y_total = complex(zeros(2,2));

    k = k_eff(ii);
    Z = Z_eff(ii);

    for s = 1:Nc

        % 三层微穿孔板
        Tp1 = [1,Zp(s,1,ii);0,1];
        Tp2 = [1,Zp(s,2,ii);0,1];
        Tp3 = [1,Zp(s,3,ii);0,1];

        % 两个折纸等效腔
        H1 = H_eq(s,1);
        H2 = H_eq(s,2);

        Tc1 = [cos(k*H1),1i*Z*sin(k*H1);
               1i*sin(k*H1)/Z,cos(k*H1)];

        Tc2 = [cos(k*H2),1i*Z*sin(k*H2);
               1i*sin(k*H2)/Z,cos(k*H2)];

        % 单胞总传递矩阵
        Tm = Tp1*Tc1*Tp2*Tc2*Tp3;

        A = Tm(1,1);
        B = Tm(1,2);
        C = Tm(2,1);
        D = Tm(2,2);

        % 单胞导纳矩阵
        Ys = [D/B,C-A*D/B;
              1/B,-A/B];

        Y_total = Y_total+weight(s)*Ys;
    end

    % 并联导纳矩阵恢复传递矩阵
    Y11 = Y_total(1,1);
    Y12 = Y_total(1,2);
    Y21 = Y_total(2,1);
    Y22 = Y_total(2,2);

    A = -Y22/Y21;
    B = 1/Y21;
    C = Y12-Y11*Y22/Y21;
    D = Y11/Y21;

    % 双端口散射参数
    den = A+B/Z0+C*Z0+D;

    tau(ii) = 2/den;
    rf(ii) = (A+B/Z0-C*Z0-D)/den;
    rb(ii) = (-A+B/Z0-C*Z0+D)/den;
end

%% 7. 正反向吸声、反射及透射

Rf = abs(rf).^2;
Rb = abs(rb).^2;
Tr = abs(tau).^2;

alpha_f = 1-Rf-Tr;
alpha_b = 1-Rb-Tr;

%% 8. 结果输出

fprintf('\n========== 100–600 Hz 吸声性能 ==========\n');

fprintf('正向平均吸声：%.4f\n',mean(alpha_f));
fprintf('反向平均吸声：%.4f\n',mean(alpha_b));

fprintf('正向最小吸声：%.4f\n',min(alpha_f));
fprintf('反向最小吸声：%.4f\n',min(alpha_b));

fprintf('正向最大吸声：%.4f\n',max(alpha_f));
fprintf('反向最大吸声：%.4f\n',max(alpha_b));

fprintf('平均方向差：%.4f\n',mean(alpha_f-alpha_b));

if any(alpha_f < -1e-6) || any(alpha_b < -1e-6)
    warning('存在负吸声系数，请检查模型的被动性。');
end

%% 9. 绘图

figure('Color','w','Position',[150 80 1000 800]);

A = [alpha_f;alpha_b];
R = [Rf;Rb];

titles = {'(a) Forward incidence','(b) Backward incidence'};

for i = 1:2

    subplot(2,1,i);

    hAbs = plot(f,A(i,:),'b-','LineWidth',1.8);
    hold on;

    hRef = plot(f,R(i,:),'r--','LineWidth',1.8);
    hTr = plot(f,Tr,'k:','LineWidth',1.8);

    title(titles{i},'FontWeight','normal');

    xlabel('Frequency/Hz');
    ylabel('Coefficient');

    legend([hAbs, hRef, hTr], {'Absorption','Reflection','Transmission'}, ...
        'Location','northeast');

    xlim([100 600]);
    ylim([0 1]);

    xticks(100:100:600);
    yticks(0:0.2:1);

    set(gca,'FontName','Times New Roman', ...
        'FontSize',16,'LineWidth',1.2,'TickDir','in');

    grid on;
    box on;
end

outDir = fileparts(mfilename('fullpath'));
if isempty(outDir)
    outDir = pwd;
end
print(gcf, fullfile(outDir, 'origami_four_cell.png'), '-dpng', '-r120');
fprintf('图已保存。\n');
