clear;
clc;
close all;

%% 1. 基本参数

f = 100:1:700;
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

% 三层微穿孔板厚度（m）。两态共用。
t = 2e-3*ones(Nc,3);

% 12 个孔径（m）。每行一个单胞，三列是三块板。两态共用。
d = 1e-3 * [
    8.000000  0.190362  2.937947;
    5.383374  2.741657  0.100000;
    0.100000  4.984316  8.000000;
    8.000000  8.000000  0.100000
];

% 状态 1：正向在一个频率上接近完美吸声，反向接近全反射。
H1 = 1e-3 * [
    62.900654  60.781378;
    18.939748  80.000000;
    72.038798  14.176110;
    80.000000  66.004906
];

% 状态 2：两侧对调。
H2 = 1e-3 * [
    10.000000  56.893990;
    22.703873  80.000000;
    75.716011  10.000000;
    10.000000  10.000000
];

n_holes = ones(Nc,3);
phi = pi*d.^2.*n_holes/(4*S);
weight = ones(1,Nc)/Nc;

%% 3. 折纸几何

a = 30;       % mm
hc = 20;      % mm
n0 = 4;
beta = 45;    % degree

%% 4. 微穿孔板阻抗（孔径不变，只算一次）

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

%% 5. Full Stinson Model（与腔高无关，只算一次）

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

%% 6. 两种腔高分别算正反向吸声

Hstate = cat(3,H1,H2);
alpha = zeros(2,2,Nf);
refl = zeros(2,2,Nf);
trans = zeros(2,Nf);

for state = 1:2
    H = Hstate(:,:,state);
    H_mm = H*1e3;
    h = H_mm/n0;

    if any(H_mm(:)<=0) || any(H_mm(:)>n0*hc)
        error('状态 %d 的腔高必须满足 0 < H <= %.0f mm。', state, n0*hc);
    end

    q = sqrt(max(0,hc^2-h.^2));
    V11 = a.*h.*q/16;
    V12 = h.*(a-hc).*q/(16*tand(beta));
    V13 = h.*q/24.*(hc/tand(beta)-q/2);
    V_mm3 = 8*n0.*(a^2.*h/8-V11-V12-V13);

    if any(V_mm3(:)<=0)
        error('状态 %d 的折纸腔体积出现非正值。', state);
    end

    H_eq = V_mm3*1e-9/S;

    fprintf('\n========== 状态 %d ==========\n', state);
    disp('实际腔高（mm）：');
    disp(H_mm);
    disp('等效腔高（mm）：');
    disp(H_eq*1e3);

    rf = complex(zeros(1,Nf));
    rb = complex(zeros(1,Nf));
    tau = complex(zeros(1,Nf));

    for ii = 1:Nf
        Y_total = complex(zeros(2,2));
        k = k_eff(ii);
        Z = Z_eff(ii);

        for s = 1:Nc
            Tp1 = [1,Zp(s,1,ii);0,1];
            Tp2 = [1,Zp(s,2,ii);0,1];
            Tp3 = [1,Zp(s,3,ii);0,1];

            Hc1 = H_eq(s,1);
            Hc2 = H_eq(s,2);
            Tc1 = [cos(k*Hc1),1i*Z*sin(k*Hc1);
                   1i*sin(k*Hc1)/Z,cos(k*Hc1)];
            Tc2 = [cos(k*Hc2),1i*Z*sin(k*Hc2);
                   1i*sin(k*Hc2)/Z,cos(k*Hc2)];

            Tm = Tp1*Tc1*Tp2*Tc2*Tp3;
            A = Tm(1,1);
            B = Tm(1,2);
            C = Tm(2,1);
            D = Tm(2,2);
            Ys = [D/B,C-A*D/B;1/B,-A/B];
            Y_total = Y_total+weight(s)*Ys;
        end

        Y11 = Y_total(1,1);
        Y12 = Y_total(1,2);
        Y21 = Y_total(2,1);
        Y22 = Y_total(2,2);

        A = -Y22/Y21;
        B = 1/Y21;
        C = Y12-Y11*Y22/Y21;
        D = Y11/Y21;

        den = A+B/Z0+C*Z0+D;
        tau(ii) = 2/den;
        rf(ii) = (A+B/Z0-C*Z0-D)/den;
        rb(ii) = (-A+B/Z0-C*Z0+D)/den;
    end

    alpha(state,1,:) = 1-abs(rf).^2-abs(tau).^2;
    alpha(state,2,:) = 1-abs(rb).^2-abs(tau).^2;
    refl(state,1,:) = abs(rf).^2;
    refl(state,2,:) = abs(rb).^2;
    trans(state,:) = abs(tau).^2;
end

%% 7. 单侧完美吸声频率点

% 状态1：正向吸声接近 1，同时反向反射接近 1。
score1 = squeeze(alpha(1,1,:)).' - squeeze(refl(1,1,:)).' - squeeze(trans(1,:)) ...
    + squeeze(refl(1,2,:)).' - squeeze(alpha(1,2,:)).';
[~, i1] = max(score1);
% 状态2：反向吸声接近 1，同时正向反射接近 1。
score2 = squeeze(alpha(2,2,:)).' - squeeze(refl(2,2,:)).' - squeeze(trans(2,:)) ...
    + squeeze(refl(2,1,:)).' - squeeze(alpha(2,1,:)).';
[~, i2] = max(score2);
fp = [f(i1), f(i2)];

fprintf('\n========== 单侧完美吸声 ==========\n');
fprintf('状态1  %.0f Hz  正向吸声 %.4f  反射 %.4f  透射 %.4f\n', ...
    f(i1), alpha(1,1,i1), refl(1,1,i1), trans(1,i1));
fprintf('状态1  %.0f Hz  反向吸声 %.4f  反射 %.4f  透射 %.4f\n', ...
    f(i1), alpha(1,2,i1), refl(1,2,i1), trans(1,i1));
fprintf('状态2  %.0f Hz  反向吸声 %.4f  反射 %.4f  透射 %.4f\n', ...
    f(i2), alpha(2,2,i2), refl(2,2,i2), trans(2,i2));
fprintf('状态2  %.0f Hz  正向吸声 %.4f  反射 %.4f  透射 %.4f\n', ...
    f(i2), alpha(2,1,i2), refl(2,1,i2), trans(2,i2));

%% 8. 绘图

titles = { ...
    '状态1 正向入射', '状态1 反向入射', ...
    '状态2 正向入射', '状态2 反向入射'};
side = [1 2 1 2];
stateId = [1 1 2 2];

figure('Color','w','Position',[40 40 1100 760]);
for p = 1:4
    subplot(2,2,p);
    st = stateId(p);
    sd = side(p);
    hold on;
    plot([fp(st) fp(st)], [0 1], 'Color', [0.55 0.55 0.55], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    hA = plot(f, squeeze(alpha(st,sd,:)), 'b-', 'LineWidth', 1.6);
    hR = plot(f, squeeze(refl(st,sd,:)), 'r--', 'LineWidth', 1.6);
    hT = plot(f, trans(st,:), 'k:', 'LineWidth', 1.6);
    grid on; box on;
    xlim([100 700]); ylim([0 1]);
    xticks(100:100:700); yticks(0:0.2:1);
    xlabel('Frequency/Hz'); ylabel('Coefficient');
    title(titles{p}, 'FontWeight', 'normal');
    legend([hA, hR, hT], {'Absorption', 'Reflection', 'Transmission'}, 'Location', 'best');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.0, 'TickDir', 'in');
end

outDir = fileparts(mfilename('fullpath'));
if isempty(outDir)
    outDir = pwd;
end
print(gcf, fullfile(outDir, 'ep_reversal_result.png'), '-dpng', '-r140');
fprintf('图已保存。\n');
