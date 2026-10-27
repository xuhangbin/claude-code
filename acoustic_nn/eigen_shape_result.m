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

%% 2. 结构参数

t = 2e-3*ones(Nc,3);

d = 1e-3 * [
    0.10523839  2.30889639  7.99639151;
    6.67688764  0.10287452  4.90627569;
    6.05055847  0.10000000  7.12840978;
    0.17713759  0.21444739  7.16823169
];

H = 1e-3 * [
    10.00000000  10.00000000;
    79.79873067  10.00000000;
    79.97383299  10.00000000;
    29.26046627  10.00000000
];

n_holes = ones(Nc,3);
phi = pi*d.^2.*n_holes/(4*S);
weight = ones(1,Nc)/Nc;

a = 30;
hc = 20;
n0 = 4;
beta = 45;

%% 3. 微穿孔板阻抗

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

%% 4. Stinson

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

%% 5. 传递矩阵

H_mm = H*1e3;
h = H_mm/n0;
q = sqrt(max(0,hc^2-h.^2));
V11 = a.*h.*q/16;
V12 = h.*(a-hc).*q/(16*tand(beta));
V13 = h.*q/24.*(hc/tand(beta)-q/2);
V_mm3 = 8*n0.*(a^2.*h/8-V11-V12-V13);
H_eq = V_mm3*1e-9/S;

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
        Tc1 = [cos(k*Hc1),1i*Z*sin(k*Hc1);1i*sin(k*Hc1)/Z,cos(k*Hc1)];
        Tc2 = [cos(k*Hc2),1i*Z*sin(k*Hc2);1i*sin(k*Hc2)/Z,cos(k*Hc2)];
        Tm = Tp1*Tc1*Tp2*Tc2*Tp3;
        A = Tm(1,1); B = Tm(1,2); C = Tm(2,1); D = Tm(2,2);
        Ys = [D/B,C-A*D/B;1/B,-A/B];
        Y_total = Y_total+weight(s)*Ys;
    end
    Y11 = Y_total(1,1); Y12 = Y_total(1,2);
    Y21 = Y_total(2,1); Y22 = Y_total(2,2);
    A = -Y22/Y21; B = 1/Y21;
    C = Y12-Y11*Y22/Y21; D = Y11/Y21;
    den = A+B/Z0+C*Z0+D;
    tau(ii) = 2/den;
    rf(ii) = (A+B/Z0-C*Z0-D)/den;
    rb(ii) = (-A+B/Z0-C*Z0+D)/den;
end

%% 6. 特征值

delta = sqrt(rf.*rb);
lam_plus = tau+delta;
lam_minus = tau-delta;
for ii = 2:Nf
    if abs(lam_plus(ii)-lam_plus(ii-1))+abs(lam_minus(ii)-lam_minus(ii-1)) ...
            > abs(lam_plus(ii)-lam_minus(ii-1))+abs(lam_minus(ii)-lam_plus(ii-1))
        tmp = lam_plus(ii);
        lam_plus(ii) = lam_minus(ii);
        lam_minus(ii) = tmp;
    end
end

Frequency_Hz = f(:);
Re_plus = real(lam_plus(:));
Im_plus = imag(lam_plus(:));
Re_minus = real(lam_minus(:));
Im_minus = imag(lam_minus(:));

fprintf('工作区变量：Frequency_Hz, Re_plus, Im_plus, Re_minus, Im_minus\n');
fprintf('400 Hz 是第 301 行\n');
fprintf('400 Hz  Re+ %.4f  Re- %.4f  Im+ %.4f  Im- %.4f\n', ...
    Re_plus(301), Re_minus(301), Im_plus(301), Im_minus(301));

%% 7. 作图

figure('Color','w','Position',[80 40 520 640]);

subplot(2,1,1);
plot(f, Re_plus, 'Color', [0.90 0.45 0.10], 'LineWidth', 1.8);
hold on;
plot(f, Re_minus, 'Color', [0.20 0.55 0.55], 'LineWidth', 1.8);
grid on; box on;
xlim([100 700]); ylim([-1.05 1.05]);
xlabel('Frequency/Hz'); ylabel('Re(\lambda)');
legend('Re(\lambda_+)','Re(\lambda_-)','Location','northeast');
set(gca,'FontName','Times New Roman','FontSize',12,'LineWidth',1,'TickDir','in');

subplot(2,1,2);
plot(f, Im_plus, 'Color', [0.90 0.45 0.10], 'LineWidth', 1.8);
hold on;
plot(f, Im_minus, 'Color', [0.20 0.55 0.55], 'LineWidth', 1.8);
grid on; box on;
xlim([100 700]); ylim([-1.05 1.05]);
xlabel('Frequency/Hz'); ylabel('Im(\lambda)');
legend('Im(\lambda_+)','Im(\lambda_-)','Location','northeast');
set(gca,'FontName','Times New Roman','FontSize',12,'LineWidth',1,'TickDir','in');
