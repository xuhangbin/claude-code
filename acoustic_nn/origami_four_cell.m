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

t = 2e-3*ones(Nc,3);

% 微穿孔板孔径，单位 m
d = 1e-3 * [
    3.15 0.36 4.71;
    3.61 8.00 6.94;
    4.82 0.10 3.06;
    4.42 1.19 0.10
];

% 两个折纸腔实际高度，单位 m
H = 1e-3 * [
    10.64 59.00;
    42.79 59.00;
    36.74 59.00;
    20.06 0.50
];

n_holes = ones(Nc,3);

phi = pi*d.^2.*n_holes/(4*S);

weight = ones(1,Nc)/Nc;

%% 3. 折纸腔体积及等效高度

a = 30;       % mm
hc = 20;      % mm
n0 = 4;
beta = 45;    % degree

% 理论最大腔高
H_max = n0*hc;       % mm，即80 mm

% 腔体实际高度，单位 mm
H_mm = H*1e3;

% 腔高限制：0 < H <= 80 mm
if any(H_mm(:)<=0) || any(H_mm(:)>H_max)
    error('折纸腔高度必须满足 0 < H <= %.2f mm。',H_max);
end

% 每层折纸高度
h = H_mm/n0;

q = sqrt(max(0,hc^2-h.^2));

V11 = a.*h.*q/16;
V12 = h.*(a-hc).*q/(16*tand(beta));
V13 = h.*q/24.*(hc/tand(beta)-q/2);

V_mm3 = 8*n0.*(a^2.*h/8-V11-V12-V13);

if any(V_mm3(:)<=0)
    error('折纸腔体积出现非正值。');
end

% 折纸腔等效高度，单位 m
H_eq = V_mm3*1e-9/S;

%% 4. 微穿孔板声阻抗

Zp = complex(zeros(Nc,3,Nf));

for s = 1:Nc
    for j = 1:3

        y = (d(s,j)/2)./sqrt(eta./(rho0*omega));
        q_mpp = y*sqrt(-1i);

        F = 1-2*besselj(1,q_mpp)./ ...
            (q_mpp.*besselj(0,q_mpp));

        Z_temp = 1i*omega*rho0*t(s,j)./(phi(s,j).*F) ...
            +sqrt(2)*eta*y./(phi(s,j)*d(s,j)) ...
            +1i*0.85*omega*rho0*d(s,j)/phi(s,j);

        Zp(s,j,:) = reshape(Z_temp,1,1,Nf);
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

        Sv = Sv+wmn*lambda2./ ...
            (lambda2+1i*omega/nu);

        St = St+wmn*lambda2./ ...
            (lambda2+1i*omega/mu_bar);
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

        Tp1 = [1,Zp(s,1,ii);0,1];
        Tp2 = [1,Zp(s,2,ii);0,1];
        Tp3 = [1,Zp(s,3,ii);0,1];

        H1 = H_eq(s,1);
        H2 = H_eq(s,2);

        Tc1 = [cos(k*H1),1i*Z*sin(k*H1);
               1i*sin(k*H1)/Z,cos(k*H1)];

        Tc2 = [cos(k*H2),1i*Z*sin(k*H2);
               1i*sin(k*H2)/Z,cos(k*H2)];

        Tm = Tp1*Tc1*Tp2*Tc2*Tp3;

        A = Tm(1,1);
        B = Tm(1,2);
        C = Tm(2,1);
        D = Tm(2,2);

        Ys = [D/B,C-A*D/B;
              1/B,-A/B];

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

%% 7. 正反向吸声、反射及透射

Rf = abs(rf).^2;
Rb = abs(rb).^2;
Tr = abs(tau).^2;

alpha_f = 1-Rf-Tr;
alpha_b = 1-Rb-Tr;

%% 8. 绘图

figure('Color','w','Position',[150 80 1000 800]);

Absorption = [alpha_f;alpha_b];
Reflection = [Rf;Rb];

titles = {'(a) Forward incidence', ...
          '(b) Backward incidence'};

for i = 1:2

    subplot(2,1,i);

    plot(f,Absorption(i,:),'b-','LineWidth',1.8);
    hold on;

    plot(f,Reflection(i,:),'r--','LineWidth',1.8);
    plot(f,Tr,'k:','LineWidth',1.8);

    title(titles{i},'FontWeight','normal');

    xlabel('Frequency/Hz');
    ylabel('Coefficient');

    legend('Absorption','Reflection','Transmission', ...
        'Location','best');

    % 与前面的频率范围自动保持一致
    xlim([min(f) max(f)]);
    ylim([0 1]);

    xticks('auto');
    yticks(0:0.2:1);

    set(gca, ...
        'FontName','Times New Roman', ...
        'FontSize',16, ...
        'LineWidth',1.2, ...
        'TickDir','in');

    grid on;
    box on;
end

outDir = fileparts(mfilename('fullpath'));
if isempty(outDir)
    outDir = pwd;
end
print(gcf, fullfile(outDir, 'origami_four_cell.png'), '-dpng', '-r120');
