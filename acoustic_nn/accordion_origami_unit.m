function accordion_origami_unit(a, l, beta, theta, b)
% 手风琴折纸单胞
%
% 几何取自 Liu 等, International Journal of Mechanical Sciences 314 (2026) 111391,
% Fig. 1(d) 和式 (1)(2)。二维折痕是矩形管展开后的一圈侧壁，折起来是一层风箱。
%
%   a' = a - 2 l / tan(beta)     式 (1)，X 面板中间那段短谷折
%   h  = 2 l sin(theta / 2)      式 (2)，这一层折起来的高度
%
% l 是一块矩形板的长度，从外圈山折到中间折痕。一条单胞的纸条高度是 2l，
% 中间折痕把纸条分成上下两块板。theta = 180 度时两块板展平，h = 2l。
% theta 变小，外圈山折靠近，中间谷折向管内收，层高下降。
%
% 外圈山折、直谷、a' 和山折侧段的长度都保持不变。对角谷折在收拢时会略微伸长，
% theta = 180 度时与平面折痕一致。
%
% 输入（长度 mm，角度度）：
%   a      X 面板宽度，也是管截面的一条边
%   l      矩形板长度
%   beta   对角谷折和水平折痕的夹角
%   theta  两块矩形板之间的折叠角
%   b      直谷面板宽度，默认等于 a。Fig. 8 里 a 和 b 可以不同
%
% 直接运行 accordion_origami_unit
% 或者 accordion_origami_unit(70, 10, 45, 90)
% 或者 accordion_origami_unit(100, 10, 45, 80, 40)

if nargin < 1 || isempty(a), a = 70; end
if nargin < 2 || isempty(l), l = 10; end
if nargin < 3 || isempty(beta), beta = 45; end
if nargin < 4 || isempty(theta), theta = 90; end
if nargin < 5 || isempty(b), b = a; end

spec = creaseSpec(a, l, beta, b);
shape = foldedLayer(spec, theta);
printReport(spec, shape);

if exist('OCTAVE_VERSION', 'builtin')
    set(0, 'defaulttextfontname', 'WenQuanYi Micro Hei');
    set(0, 'defaultaxesfontname', 'WenQuanYi Micro Hei');
end

fig = figure('Color', 'w', 'Position', [40, 40, 1360, 820]);

axes('Position', [0.04, 0.60, 0.92, 0.34]);
drawCreasePattern(spec);
title('二维折痕    红实线山折，蓝虚线谷折');

axes('Position', [0.02, 0.06, 0.34, 0.48]);
drawLayer(shape, 0);
finishSolid(spec, shape.h, sprintf('单胞    θ = %.0f°    h = %.1f mm', theta, shape.h));

axes('Position', [0.34, 0.06, 0.32, 0.48]);
nStack = 4;
for iLayer = 0:(nStack - 1)
    drawLayer(shape, iLayer * shape.h);
end
finishSolid(spec, nStack * shape.h, sprintf('叠 %d 层    H = %.1f mm', nStack, nStack * shape.h));

axes('Position', [0.68, 0.08, 0.30, 0.44]);
drawSections(spec, comparisonAngles(theta), theta);
title('侧向剖面：外圈是山，中间是谷', 'Interpreter', 'none');

outDir = fileparts(mfilename('fullpath'));
if isempty(outDir)
    outDir = pwd;
end
figPath = fullfile(outDir, 'accordion_origami_unit.png');
drawnow;
print(fig, figPath, '-dpng', '-r140');
fprintf('图已保存到 %s\n', figPath);
end

function spec = creaseSpec(a, l, beta, b)
if a <= 0 || b <= 0 || l <= 0
    error('a、b、l 都要是正数。');
end
if beta <= 0 || beta >= 90
    error('beta 要在 0 到 90 度之间。');
end
flank = l / tand(beta);
aPrime = a - 2 * flank;
if aPrime <= 0
    error('a'' 必须是正数。请加大 beta，或减小 l / a，使 tan(beta) > 2l/a。');
end
spec = struct();
spec.a = a;
spec.b = b;
spec.l = l;
spec.beta = beta;
spec.flank = flank;
spec.aPrime = aPrime;
spec.diag = hypot(flank, l);
end

function shape = foldedLayer(spec, theta)
if theta <= 0 || theta >= 180
    error('theta 要在 0 到 180 度之间。');
end
l = spec.l;
flank = spec.flank;
a = spec.a;
b = spec.b;
h = 2 * l * sind(theta / 2);
s = l * cosd(theta / 2);
y2 = s * (2 * flank - s);
if y2 < 0
    error('这个 beta 和 theta 折不到一起。请减小 beta，或增大 theta。');
end
yIn = sqrt(y2);
if s >= a / 2
    error('左右两面的谷折交叉了。请加大 a，或增大 theta。');
end
if yIn >= b / 2
    error('前后两面的 X 谷折碰到了。请加大 b，或增大 beta。');
end

B = [
    0, 0, 0
    a, 0, 0
    a, b, 0
    0, b, 0
];
T = B;
T(:, 3) = h;
M = [
    s, 0, h / 2
    a - s, 0, h / 2
    a - s, b, h / 2
    s, b, h / 2
];
U = [flank, yIn, h / 2];
V = [a - flank, yIn, h / 2];
U2 = [a - flank, b - yIn, h / 2];
V2 = [flank, b - yIn, h / 2];

err = 0;
err = edgeErr(err, B(1, :), B(2, :), a);
err = edgeErr(err, B(2, :), B(3, :), b);
err = edgeErr(err, B(3, :), B(4, :), a);
err = edgeErr(err, B(4, :), B(1, :), b);
err = edgeErr(err, T(1, :), T(2, :), a);
err = edgeErr(err, T(2, :), T(3, :), b);
err = edgeErr(err, T(3, :), T(4, :), a);
err = edgeErr(err, T(4, :), T(1, :), b);
for i = 1:4
    err = edgeErr(err, B(i, :), M(i, :), l);
    err = edgeErr(err, M(i, :), T(i, :), l);
end
err = edgeErr(err, M(2, :), M(3, :), b);
err = edgeErr(err, M(4, :), M(1, :), b);
err = edgeErr(err, M(1, :), U, flank);
err = edgeErr(err, V, M(2, :), flank);
err = edgeErr(err, M(3, :), U2, flank);
err = edgeErr(err, V2, M(4, :), flank);
err = edgeErr(err, U, V, spec.aPrime);
err = edgeErr(err, U2, V2, spec.aPrime);

vol = 0;
scale = a * l * l;
vol = faceVol(vol, B(1, :), B(2, :), V, U, scale);
vol = faceVol(vol, T(1, :), U, V, T(2, :), scale);
vol = faceVol(vol, B(3, :), B(4, :), V2, U2, scale);
vol = faceVol(vol, T(3, :), U2, V2, T(4, :), scale);
vol = faceVol(vol, B(2, :), B(3, :), M(3, :), M(2, :), scale);
vol = faceVol(vol, T(2, :), M(2, :), M(3, :), T(3, :), scale);
vol = faceVol(vol, B(4, :), B(1, :), M(1, :), M(4, :), scale);
vol = faceVol(vol, T(4, :), M(4, :), M(1, :), T(1, :), scale);

v1 = B(2, :) - M(2, :);
v2 = T(2, :) - M(2, :);
cosang = dot(v1, v2) / (norm(v1) * norm(v2));
cosang = max(-1, min(1, cosang));
angleDeg = acosd(cosang);

shape = struct();
shape.theta = theta;
shape.h = h;
shape.s = s;
shape.yIn = yIn;
shape.B = B;
shape.T = T;
shape.M = M;
shape.U = U;
shape.V = V;
shape.U2 = U2;
shape.V2 = V2;
shape.edgeError = err;
shape.flatError = vol;
shape.angleDeg = angleDeg;
shape.diagFold = norm(U - B(1, :));
shape.diagFlat = spec.diag;
if err > 1e-6 || vol > 1e-6 || abs(angleDeg - theta) > 1e-4
    error('折叠几何没有闭合。');
end
end

function printReport(spec, shape)
fprintf('X 面板宽度 a = %.3f mm\n', spec.a);
fprintf('直谷面板宽度 b = %.3f mm\n', spec.b);
fprintf('矩形板长度 l = %.3f mm（纸条高度 2l = %.3f mm）\n', spec.l, 2 * spec.l);
fprintf('夹角 beta = %.2f 度\n', spec.beta);
fprintf('折叠角 theta = %.2f 度\n', shape.theta);
fprintf('中间谷折 a'' = %.3f mm\n', spec.aPrime);
fprintf('山折侧段 = %.3f mm\n', spec.flank);
fprintf('层高 h = %.3f mm\n', shape.h);
fprintf('直谷内收 = %.3f mm\n', shape.s);
fprintf('X 谷内收 = %.3f mm\n', shape.yIn);
fprintf('板间夹角 = %.3f 度（误差 %.3e 度）\n', shape.angleDeg, abs(shape.angleDeg - shape.theta));
fprintf('保持长度的折痕误差 = %.3e mm\n', shape.edgeError);
fprintf('面板平面误差 = %.3e\n', shape.flatError);
fprintf('对角谷折：平面长度 %.3f mm，折叠后 %.3f mm\n', shape.diagFlat, shape.diagFold);
end

function drawCreasePattern(spec)
a = spec.a;
b = spec.b;
l = spec.l;
flank = spec.flank;
aPrime = spec.aPrime;
beta = spec.beta;
red = [0.78, 0.08, 0.10];
blue = [0.08, 0.32, 0.75];
H = 2 * l;
widths = [a, b, a, b];
kinds = {'x', 'plain', 'x', 'plain'};
hold on;

plot([0, 0], [0, H], 'Color', red, 'LineWidth', 1.6);
x = 0;
for iPanel = 1:4
    w = widths(iPanel);
    if strcmp(kinds{iPanel}, 'x')
        p = x + flank;
        q = x + w - flank;
        plot([x, p], [l, l], 'Color', red, 'LineWidth', 1.6);
        plot([q, x + w], [l, l], 'Color', red, 'LineWidth', 1.6);
        plot([p, q], [l, l], 'Color', blue, 'LineStyle', '--', 'LineWidth', 1.5);
        plot([x, p], [0, l], 'Color', blue, 'LineStyle', '--', 'LineWidth', 1.4);
        plot([x, p], [H, l], 'Color', blue, 'LineStyle', '--', 'LineWidth', 1.4);
        plot([x + w, q], [0, l], 'Color', blue, 'LineStyle', '--', 'LineWidth', 1.4);
        plot([x + w, q], [H, l], 'Color', blue, 'LineStyle', '--', 'LineWidth', 1.4);
    else
        plot([x, x + w], [l, l], 'Color', blue, 'LineStyle', '--', 'LineWidth', 1.5);
    end
    x = x + w;
    plot([x, x], [0, H], 'Color', red, 'LineWidth', 1.6);
end
totalW = x;
plot([0, totalW], [0, 0], 'Color', red, 'LineWidth', 1.6);
plot([0, totalW], [H, H], 'Color', red, 'LineWidth', 1.6);

dim = [0.15, 0.15, 0.15];
drawDoubleArrow([0, a], [H, H] + 0.55 * l, dim);
text(a / 2, H + 1.15 * l, 'a', 'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', dim);
drawDoubleArrow([a, a + b], [H, H] + 0.55 * l, dim);
text(a + b / 2, H + 1.15 * l, 'b', 'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', dim);

drawDoubleArrow([0, 0] - 0.7 * l, [0, l], dim);
text(-1.5 * l, l / 2, 'l', 'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', dim);

drawDoubleArrow([flank, flank + aPrime], [0, 0] - 0.55 * l, dim);
text(flank + aPrime / 2, -1.25 * l, 'a''', 'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', dim);

pArc = [flank, l];
radius = 0.48 * l;
ang = linspace(pi - beta * pi / 180, pi, 24);
plot(pArc(1) + radius * cos(ang), pArc(2) + radius * sin(ang), 'Color', dim, 'LineWidth', 0.9);
text(0.34 * flank, l + 0.42 * l, 'β', ...
    'FontSize', 11, 'Color', dim, 'Interpreter', 'none');

plot([0.72 * totalW, 0.78 * totalW], [H, H] + 1.7 * l, 'Color', red, 'LineWidth', 1.6);
text(0.79 * totalW, H + 1.7 * l, '山折', 'FontSize', 10, 'Color', red, 'VerticalAlignment', 'middle');
plot([0.72 * totalW, 0.78 * totalW], [H, H] + 2.5 * l, 'Color', blue, 'LineStyle', '--', 'LineWidth', 1.5);
text(0.79 * totalW, H + 2.5 * l, '谷折', 'FontSize', 10, 'Color', blue, 'VerticalAlignment', 'middle');

xlim([-2.2 * l, totalW + 0.6 * l]);
ylim([-1.8 * l, H + 3.1 * l]);
axis equal;
axis off;
end

function drawLayer(shape, z0)
red = [0.78, 0.08, 0.10];
blue = [0.08, 0.32, 0.75];
lower = [0.97, 0.90, 0.74];
upper = [0.73, 0.77, 0.84];
shift = [0, 0, z0];
B = shape.B + shift;
T = shape.T + shift;
M = shape.M + shift;
U = shape.U + shift;
V = shape.V + shift;
U2 = shape.U2 + shift;
V2 = shape.V2 + shift;
hold on;

paint(B(1, :), M(1, :), U, [], lower);
paint(T(1, :), U, M(1, :), [], upper);
paint(B(2, :), V, M(2, :), [], lower);
paint(T(2, :), M(2, :), V, [], upper);
paint(B(1, :), B(2, :), V, U, lower);
paint(T(1, :), U, V, T(2, :), upper);

paint(B(3, :), M(3, :), U2, [], lower);
paint(T(3, :), U2, M(3, :), [], upper);
paint(B(4, :), V2, M(4, :), [], lower);
paint(T(4, :), M(4, :), V2, [], upper);
paint(B(3, :), B(4, :), V2, U2, lower);
paint(T(3, :), U2, V2, T(4, :), upper);

paint(B(2, :), B(3, :), M(3, :), M(2, :), lower);
paint(T(2, :), M(2, :), M(3, :), T(3, :), upper);
paint(B(4, :), B(1, :), M(1, :), M(4, :), lower);
paint(T(4, :), M(4, :), M(1, :), T(1, :), upper);

for i = 1:4
    j = mod(i, 4) + 1;
    crease(B(i, :), B(j, :), red, '-', 1.8);
    crease(T(i, :), T(j, :), red, '-', 1.8);
    crease(B(i, :), M(i, :), red, '-', 1.3);
    crease(M(i, :), T(i, :), red, '-', 1.3);
end
crease(M(1, :), U, red, '-', 1.7);
crease(V, M(2, :), red, '-', 1.7);
crease(M(3, :), U2, red, '-', 1.7);
crease(V2, M(4, :), red, '-', 1.7);

crease(U, V, blue, '--', 1.6);
crease(U2, V2, blue, '--', 1.6);
crease(B(1, :), U, blue, '--', 1.4);
crease(T(1, :), U, blue, '--', 1.4);
crease(B(2, :), V, blue, '--', 1.4);
crease(T(2, :), V, blue, '--', 1.4);
crease(B(3, :), U2, blue, '--', 1.4);
crease(T(3, :), U2, blue, '--', 1.4);
crease(B(4, :), V2, blue, '--', 1.4);
crease(T(4, :), V2, blue, '--', 1.4);
crease(M(2, :), M(3, :), blue, '--', 1.5);
crease(M(4, :), M(1, :), blue, '--', 1.5);
end

function finishSolid(spec, zTop, name)
a = spec.a;
b = spec.b;
pad = 0.08 * max(a, b);
xlim([-pad, a + pad]);
ylim([-pad, b + pad]);
zlim([0, max(zTop, spec.l) * 1.08]);
daspect([1, 1, 1]);
view(18, -62);
axis off;
title(name, 'Interpreter', 'none');
end

function drawSections(spec, thetas, current)
red = [0.78, 0.08, 0.10];
blue = [0.08, 0.32, 0.75];
fillColor = [0.97, 0.91, 0.78];
nLayer = 3;
gap = 0.28 * spec.a;
hold on;
for iTheta = 1:numel(thetas)
    theta = thetas(iTheta);
    h = 2 * spec.l * sind(theta / 2);
    s = spec.l * cosd(theta / 2);
    x0 = (iTheta - 1) * (spec.a + gap);
    px = x0;
    pz = 0;
    for iLayer = 0:(nLayer - 1)
        z0 = iLayer * h;
        px = [px, x0 + s, x0]; %#ok<AGROW>
        pz = [pz, z0 + h / 2, z0 + h]; %#ok<AGROW>
    end
    patch([px, x0], [pz, 0], fillColor, 'EdgeColor', 'none');
    qx = x0 + spec.a;
    qz = 0;
    for iLayer = 0:(nLayer - 1)
        z0 = iLayer * h;
        qx = [qx, x0 + spec.a - s, x0 + spec.a]; %#ok<AGROW>
        qz = [qz, z0 + h / 2, z0 + h]; %#ok<AGROW>
    end
    patch([qx, x0 + spec.a], [qz, 0], fillColor, 'EdgeColor', 'none');
    plot(px, pz, 'Color', red, 'LineWidth', 1.7);
    plot(qx, qz, 'Color', red, 'LineWidth', 1.7);
    for iLayer = 0:(nLayer - 1)
        zMid = iLayer * h + h / 2;
        plot([x0 + s, x0 + spec.a - s], [zMid, zMid], 'Color', blue, 'LineStyle', '--', 'LineWidth', 1.3);
    end
    if abs(theta - current) < 1e-6
        label = sprintf('当前  θ = %.0f°\nh = %.1f mm', theta, h);
    else
        label = sprintf('θ = %.0f°\nh = %.1f mm', theta, h);
    end
    text(x0 + spec.a / 2, nLayer * h + 0.55 * spec.l, label, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 9, 'Interpreter', 'none');
end
axis equal;
axis off;
end

function ths = comparisonAngles(theta)
base = [50, theta, 130];
ths = unique(base(:)');
extras = [70, 90, 110, 150, 40];
k = 1;
while numel(ths) < 3
    if all(abs(ths - extras(k)) > 5)
        ths(end + 1) = extras(k); %#ok<AGROW>
    end
    k = k + 1;
end
ths = sort(ths(1:3));
end

function paint(p, q, r, s, color)
if isempty(s)
    pts = [p; q; r];
else
    pts = [p; q; r; s];
end
patch('XData', pts(:, 1), 'YData', pts(:, 2), 'ZData', pts(:, 3), ...
    'FaceColor', color, 'EdgeColor', 'none');
end

function crease(p, q, color, style, width)
plot3([p(1), q(1)], [p(2), q(2)], [p(3), q(3)], ...
    'Color', color, 'LineStyle', style, 'LineWidth', width);
end

function drawDoubleArrow(xs, ys, color)
plot(xs, ys, 'Color', color, 'LineWidth', 0.8);
drawHead(xs(1), ys(1), xs(2), ys(2), color);
drawHead(xs(2), ys(2), xs(1), ys(1), color);
end

function drawHead(xTip, yTip, xFrom, yFrom, color)
d = [xFrom - xTip, yFrom - yTip];
n = hypot(d(1), d(2));
if n < 1e-9
    return;
end
d = d / n;
side = [-d(2), d(1)];
len = 2.4;
wid = 0.85;
p1 = [xTip, yTip];
p2 = p1 + len * d + wid * side;
p3 = p1 + len * d - wid * side;
patch([p1(1), p2(1), p3(1)], [p1(2), p2(2), p3(2)], color, ...
    'EdgeColor', 'none', 'FaceColor', color);
end

function err = edgeErr(err, p, q, target)
err = max(err, abs(norm(p - q) - target));
end

function err = faceVol(err, p, q, r, s, scale)
vol = abs(dot(cross(q - p, r - p), s - p)) / scale;
err = max(err, vol);
end
