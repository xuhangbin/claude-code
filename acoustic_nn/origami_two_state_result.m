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
    4.140000  1.000000  3.831000;
    7.000000  6.772000  1.000000;
    1.000000  7.000000  6.943000;
    7.000000  1.000000  7.000000
];

% 状态 1 的 8 个实际腔高（m）。每行一个单胞的两个腔。范围 10–80 mm。
H1 = 1e-3 * [
    77.299000  10.577000;
    51.725000  80.000000;
    10.000000  25.723000;
    80.000000  53.396000
];

% 状态 2 只改这 8 个实际腔高（m）。
H2 = 1e-3 * [
    10.000000  74.637567;
    29.142631  29.102739;
    80.000000  77.321389;
    10.000000  80.000000
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
end

%% 7. 250–450 Hz 平均吸声

band = f>=250 & f<=450;
names = {'状态1','状态2'};

fprintf('\n========== 250–450 Hz ==========\n');
for state = 1:2
    af = squeeze(alpha(state,1,band)).';
    ab = squeeze(alpha(state,2,band)).';
    if state == 1
        gap = mean(af)-mean(ab);
        fprintf('%s  正向 %.4f  反向 %.4f  正向高出 %.4f\n', ...
            names{state}, mean(af), mean(ab), gap);
    else
        gap = mean(ab)-mean(af);
        fprintf('%s  正向 %.4f  反向 %.4f  反向高出 %.4f\n', ...
            names{state}, mean(af), mean(ab), gap);
    end
end

%% 8. 绘图

if ispc
    fontName = 'Microsoft YaHei';
else
    fontName = 'WenQuanYi Micro Hei';
end

figure('Color','w','Position',[80 40 980 760]);

for state = 1:2
    subplot(2,1,state);
    hold on;

    patch([250 450 450 250],[0 0 1 1],[1 0.92 0.55], ...
        'EdgeColor','none','FaceAlpha',0.45,'HandleVisibility','off');

    af = squeeze(alpha(state,1,:)).';
    ab = squeeze(alpha(state,2,:)).';
    hF = plot(f,af,'b-','LineWidth',1.8);
    hB = plot(f,ab,'r--','LineWidth',1.8);

    mf = mean(af(band));
    mb = mean(ab(band));
    if state == 1
        name = sprintf('状态1：250–450 Hz 正向 %.3f，反向 %.3f，正向高出 %.3f', ...
            mf, mb, mf-mb);
    else
        name = sprintf('状态2：250–450 Hz 正向 %.3f，反向 %.3f，反向高出 %.3f', ...
            mf, mb, mb-mf);
    end

    grid on;
    box on;
    xlim([100 600]);
    ylim([0 1]);
    xticks(100:100:600);
    yticks(0:0.2:1);
    xlabel('频率 / Hz');
    ylabel('吸声系数');
    title(name,'FontWeight','normal');
    legend([hF,hB],{'正向吸声','反向吸声'},'Location','northeast');
    set(gca,'FontName',fontName,'FontSize',13,'LineWidth',1.1,'TickDir','in');
end

outDir = fileparts(mfilename('fullpath'));
if isempty(outDir)
    outDir = pwd;
end
print(gcf, fullfile(outDir, 'origami_two_state_result.png'), '-dpng', '-r140');
fprintf('图已保存。\n');
