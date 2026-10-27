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
    0.100000  2.310443  8.000000;
    6.683513  0.100000  4.914359;
    6.047711  0.100000  7.131547;
    0.100000  0.100000  8.000000
];

% 两侧对调锁在 400 Hz。
H1 = 1e-3 * [
    10.000000  10.000000;
    79.811763  10.000000;
    79.969882  10.000000;
    35.271434  10.000000
];

H2 = 1e-3 * [
    34.365237  33.200337;
    10.000000  76.273805;
    10.000000  80.000000;
    50.708384  25.028648
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
rfC = complex(zeros(2,Nf));
rbC = complex(zeros(2,Nf));
tauC = complex(zeros(2,Nf));

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
    rfC(state,:) = rf;
    rbC(state,:) = rb;
    tauC(state,:) = tau;
end

%% 7. 同一频率上的两侧对调

af1 = squeeze(alpha(1,1,:)).';
ab1 = squeeze(alpha(1,2,:)).';
Rf1 = squeeze(refl(1,1,:)).';
Rb1 = squeeze(refl(1,2,:)).';
T1 = trans(1,:);
af2 = squeeze(alpha(2,1,:)).';
ab2 = squeeze(alpha(2,2,:)).';
Rf2 = squeeze(refl(2,1,:)).';
Rb2 = squeeze(refl(2,2,:)).';
T2 = trans(2,:);
joint = (af1-1).^2 + Rf1.^2 + T1.^2 + (Rb1-1).^2 + ab1.^2 ...
    + (ab2-1).^2 + Rb2.^2 + T2.^2 + (Rf2-1).^2 + af2.^2;
[~, iBest] = min(joint);
near = af1>=0.99 & Rf1<=0.01 & T1<=0.01 & Rb1>=0.99 & ab1<=0.01 ...
    & ab2>=0.99 & Rb2<=0.01 & T2<=0.01 & Rf2>=0.99 & af2<=0.01;
fp = [f(iBest), f(iBest)];

fprintf('\n========== 同一频率上的两侧对调 ==========\n');
fprintf('最好频率 %.0f Hz\n', f(iBest));
fprintf('状态1  正向吸声 %.4f  反射 %.4f  透射 %.4f\n', af1(iBest), Rf1(iBest), T1(iBest));
fprintf('状态1  反向吸声 %.4f  反射 %.4f  透射 %.4f\n', ab1(iBest), Rb1(iBest), T1(iBest));
fprintf('状态2  正向吸声 %.4f  反射 %.4f  透射 %.4f\n', af2(iBest), Rf2(iBest), T2(iBest));
fprintf('状态2  反向吸声 %.4f  反射 %.4f  透射 %.4f\n', ab2(iBest), Rb2(iBest), T2(iBest));
fprintf('接近这一标准的频率点 %d 个：', sum(near));
fprintf(' %.0f', f(near));
fprintf(' Hz\n');

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

%% 9. 并联后散射矩阵的两个特征值
% 四单元已经在上面合成一条双端口通道。散射矩阵为
%   S = [ rf , tau ;
%         tau, rb  ]
% 两个特征值
%   lambda± = (rf+rb)/2 ± sqrt( ((rf-rb)/2)^2 + tau^2 )
% 开方沿频率连续跟踪，避免主值开方把 λ+ 和 λ- 对调。

lamP = complex(zeros(2,Nf));
lamM = complex(zeros(2,Nf));
for state = 1:2
    rf = rfC(state,:);
    rb = rbC(state,:);
    tau = tauC(state,:);
    half = (rf+rb)/2;
    disc = sqrt(((rf-rb)/2).^2 + tau.^2);
    plus = half+disc;
    minus = half-disc;
    for ii = 2:Nf
        if abs(plus(ii)-plus(ii-1))+abs(minus(ii)-minus(ii-1)) ...
                > abs(plus(ii)-minus(ii-1))+abs(minus(ii)-plus(ii-1))
            tmp = plus(ii);
            plus(ii) = minus(ii);
            minus(ii) = tmp;
        end
    end
    lamP(state,:) = plus;
    lamM(state,:) = minus;
end

i400 = find(f==400, 1);
fprintf('\n========== 散射矩阵特征值 ==========\n');
for state = 1:2
    fprintf('状态%d  400 Hz\n', state);
    fprintf('  Re(λ+) % .6f   Re(λ-) % .6f\n', real(lamP(state,i400)), real(lamM(state,i400)));
    fprintf('  Im(λ+) % .6f   Im(λ-) % .6f\n', imag(lamP(state,i400)), imag(lamM(state,i400)));
end

eigTitle = {'状态1 实部', '状态1 虚部', '状态2 实部', '状态2 虚部'};
figure('Color','w','Position',[60 60 1100 760]);
for p = 1:4
    subplot(2,2,p);
    st = ceil(p/2);
    hold on;
    if mod(p,2)==1
        yP = real(lamP(st,:));
        yM = real(lamM(st,:));
        ylab = 'Real part';
    else
        yP = imag(lamP(st,:));
        yM = imag(lamM(st,:));
        ylab = 'Imaginary part';
    end
    hP = plot(f, yP, 'b-', 'LineWidth', 1.6);
    hM = plot(f, yM, 'r--', 'LineWidth', 1.6);
    yl = ylim;
    plot([400 400], yl, 'Color', [0.55 0.55 0.55], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    ylim(yl);
    grid on; box on;
    xlim([100 700]);
    xticks(100:100:700);
    xlabel('Frequency/Hz'); ylabel(ylab);
    title(eigTitle{p}, 'FontWeight', 'normal');
    legend([hP, hM], {'\lambda+', '\lambda-'}, 'Location', 'best');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.0, 'TickDir', 'in');
end
print(gcf, fullfile(outDir, 'ep_reversal_eigenvalues.png'), '-dpng', '-r140');
fprintf('图已保存。\n');
