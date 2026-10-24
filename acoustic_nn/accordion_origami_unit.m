function accordion_origami_unit(a, l, aPrime, theta)
% 手风琴折纸单胞
%
% 左图是二维折痕，右图是折成管后的一个单胞。
% 实线红是山折，虚线蓝是谷折，和示意图一致。
%
% 纸条高度 l，沿周向分成四块等宽面板，宽度都是 a：
%   第 1、3 块是带对角谷折的侧面板
%   第 2、4 块是只有一条水平谷折的顶、底面板
% 四块面板收成一根直管，端面是菱形。
%
% 输入（长度单位 mm，角度单位度）：
%   a       每块面板的宽度，也是端面菱形的边长
%   l       单胞沿管轴的长度，也是纸条高度
%   aPrime  侧面板中段谷折的长度，要满足 0 < aPrime < a
%   theta   端面菱形的内角，顶边和侧边的夹角。90 度是方管
%
% 由上面四个量决定：
%   beta = atan(l / (a - aPrime))     对角谷折和水平折痕的夹角
%   h    = a * sin(theta)             管的总高度
%
% 直接运行 accordion_origami_unit
% 或者 accordion_origami_unit(30, 22, 12, 60)

if nargin < 1 || isempty(a), a = 30; end
if nargin < 2 || isempty(l), l = 22; end
if nargin < 3 || isempty(aPrime), aPrime = 12; end
if nargin < 4 || isempty(theta), theta = 60; end

if a <= 0 || l <= 0
    error('a 和 l 都要是正数。');
end
if aPrime <= 0 || aPrime >= a
    error('aPrime 要大于 0，并且小于 a。');
end
if theta <= 0 || theta >= 180
    error('theta 要在 0 到 180 度之间。');
end

pattern = creasePattern(a, l, aPrime);
shape = foldedTube(pattern, theta);
shape.edgeError = max(pattern.edgeError, shape.edgeError);

fprintf('面板宽度 a = %.3f mm\n', a);
fprintf('单胞长度 l = %.3f mm\n', l);
fprintf('中段谷折 a'' = %.3f mm\n', aPrime);
fprintf('对角谷折夹角 beta = %.2f 度\n', pattern.betaDeg);
fprintf('端面夹角 theta = %.2f 度\n', theta);
fprintf('折叠高度 h = %.3f mm\n', shape.h);
fprintf('端面水平偏距 = %.3f mm\n', shape.shift);
fprintf('折痕长度误差 = %.3e mm\n', shape.edgeError);
fprintf('端面夹角误差 = %.3e 度\n', shape.angleError);

fig = figure('Color', 'w', 'Position', [60, 60, 1180, 540]);

subplot(1, 2, 1);
drawCreasePattern(pattern);
title('2D crease pattern');

subplot(1, 2, 2);
drawFolded(shape);
title('3D folded unit cell');

outDir = fileparts(mfilename('fullpath'));
if isempty(outDir)
    outDir = pwd;
end
figPath = fullfile(outDir, 'accordion_origami_unit.png');
drawnow;
print(fig, figPath, '-dpng', '-r140');
fprintf('图已保存到 %s\n', figPath);
end

function pattern = creasePattern(a, l, aPrime)
beta = atan(l / (a - aPrime));
halfFlank = (a - aPrime) / 2;
names = {'side', 'top', 'side', 'bottom'};
panels = struct('name', {}, 'u0', {}, 'kind', {});
for iPanel = 1:4
    panels(iPanel).name = names{iPanel};
    panels(iPanel).u0 = (iPanel - 1) * a;
    panels(iPanel).kind = names{iPanel};
end

pattern = struct();
pattern.a = a;
pattern.l = l;
pattern.aPrime = aPrime;
pattern.betaDeg = beta * 180 / pi;
pattern.halfFlank = halfFlank;
pattern.panels = panels;
pattern.diagLength = hypot((a - aPrime) / 2, l / 2);
pattern.edgeError = 0;
end

function shape = foldedTube(pattern, thetaDeg)
a = pattern.a;
l = pattern.l;
theta = thetaDeg * pi / 180;
h = a * sin(theta);
shift = -a * cos(theta);

topLeft = [-a / 2, h / 2];
topRight = [a / 2, h / 2];
bottomRight = [a / 2 + shift, -h / 2];
bottomLeft = [-a / 2 + shift, -h / 2];

sections = [
    topLeft;
    topRight;
    bottomRight;
    bottomLeft
];

faces = zeros(4, 4, 3);
for iFace = 1:4
    i2 = mod(iFace, 4) + 1;
    p0 = [sections(iFace, :), 0];
    p1 = [sections(i2, :), 0];
    q0 = p0 + [0, 0, l];
    q1 = p1 + [0, 0, l];
    % 存成 x y z。端面坐标原来是 x z，管轴放到 y。
    faces(iFace, 1, :) = [p0(1), p0(3), p0(2)];
    faces(iFace, 2, :) = [p1(1), p1(3), p1(2)];
    faces(iFace, 3, :) = [q1(1), q1(3), q1(2)];
    faces(iFace, 4, :) = [q0(1), q0(3), q0(2)];
end

shape = struct();
shape.a = a;
shape.l = l;
shape.aPrime = pattern.aPrime;
shape.betaDeg = pattern.betaDeg;
shape.thetaDeg = thetaDeg;
shape.h = h;
shape.shift = shift;
shape.faces = faces;
shape.sections = sections;
shape.edgeError = tubeEdgeError(faces, pattern);
shape.angleError = abs(sectionAngleDeg(sections) - thetaDeg);
end

function err = tubeEdgeError(faces, pattern)
a = pattern.a;
l = pattern.l;
err = 0;
for iFace = 1:4
    p = squeeze(faces(iFace, :, :));
    len = [
        norm(p(2, :) - p(1, :));
        norm(p(3, :) - p(2, :));
        norm(p(4, :) - p(3, :));
        norm(p(1, :) - p(4, :))
    ];
    err = max(err, abs(len(1) - a));
    err = max(err, abs(len(2) - l));
    err = max(err, abs(len(3) - a));
    err = max(err, abs(len(4) - l));
end
for iFace = [2, 4]
    p = squeeze(faces(iFace, :, :));
    err = max(err, sideCreaseError(p, pattern));
end
end

function err = sideCreaseError(p, pattern)
a = pattern.a;
aPrime = pattern.aPrime;
flank = (a - aPrime) / 2;
sP = flank / a;
sQ = (flank + aPrime) / a;
err = 0;
err = max(err, abs(norm(pointOnFace(p, sP, 0.5) - pointOnFace(p, 0, 0.5)) - flank));
err = max(err, abs(norm(pointOnFace(p, sQ, 0.5) - pointOnFace(p, sP, 0.5)) - aPrime));
err = max(err, abs(norm(pointOnFace(p, 1, 0.5) - pointOnFace(p, sQ, 0.5)) - flank));
err = max(err, abs(norm(pointOnFace(p, sP, 0.5) - pointOnFace(p, 0, 1)) - pattern.diagLength));
err = max(err, abs(norm(pointOnFace(p, sQ, 0.5) - pointOnFace(p, 1, 1)) - pattern.diagLength));
err = max(err, abs(norm(pointOnFace(p, sP, 0.5) - pointOnFace(p, 0, 0)) - pattern.diagLength));
err = max(err, abs(norm(pointOnFace(p, sQ, 0.5) - pointOnFace(p, 1, 0)) - pattern.diagLength));
end

function deg = sectionAngleDeg(sections)
vTop = sections(1, :) - sections(2, :);
vSide = sections(3, :) - sections(2, :);
cosang = dot(vTop, vSide) / (norm(vTop) * norm(vSide));
cosang = max(min(cosang, 1), -1);
deg = acos(cosang) * 180 / pi;
end

function drawCreasePattern(pattern)
a = pattern.a;
l = pattern.l;
aPrime = pattern.aPrime;
hold on;

for iPanel = 1:4
    u0 = pattern.panels(iPanel).u0;
    rectangle('Position', [u0, 0, a, l], 'EdgeColor', [0.75, 0.12, 0.12], 'LineWidth', 1.6);
    if strcmp(pattern.panels(iPanel).kind, 'side')
        drawSideCreases(u0, a, l, aPrime);
    else
        plot([u0, u0 + a], [l / 2, l / 2], 'b--', 'LineWidth', 1.6);
    end
end

dimY = l + 0.16 * l + 1.2;
drawDoubleArrow(0, a, dimY, dimY, [0, 0, 0]);
text(a / 2, dimY + 0.12 * l, 'a', 'HorizontalAlignment', 'center', ...
    'Interpreter', 'none', 'FontSize', 12);

drawDoubleArrow(-0.08 * a, -0.08 * a, 0, l, [0, 0, 0]);
text(-0.18 * a, l / 2, 'l', 'HorizontalAlignment', 'right', ...
    'Interpreter', 'none', 'FontSize', 12);

uMid = (a - aPrime) / 2;
yPrime = -0.14 * l;
drawDoubleArrow(uMid, uMid + aPrime, yPrime, yPrime, [0, 0, 0]);
text(a / 2, yPrime - 0.12 * l, 'a''', 'HorizontalAlignment', 'center', ...
    'Interpreter', 'none', 'FontSize', 12);

drawBetaArc(a, l, aPrime, pattern.betaDeg);

axis equal;
xlim([-0.45 * a, 4 * a + 0.2 * a]);
ylim([-0.5 * l, l + 0.55 * l]);
grid on;
box on;
xlabel('u (mm)');
ylabel('v (mm)');
end

function drawSideCreases(u0, a, l, aPrime)
flank = (a - aPrime) / 2;
p = u0 + flank;
q = u0 + flank + aPrime;
plot([u0, p], [l / 2, l / 2], '-', 'Color', [0.75, 0.12, 0.12], 'LineWidth', 1.6);
plot([q, u0 + a], [l / 2, l / 2], '-', 'Color', [0.75, 0.12, 0.12], 'LineWidth', 1.6);
plot([p, q], [l / 2, l / 2], 'b--', 'LineWidth', 1.6);
plot([u0, p], [l, l / 2], 'b--', 'LineWidth', 1.6);
plot([u0 + a, q], [l, l / 2], 'b--', 'LineWidth', 1.6);
plot([u0, p], [0, l / 2], 'b--', 'LineWidth', 1.6);
plot([u0 + a, q], [0, l / 2], 'b--', 'LineWidth', 1.6);
end

function drawBetaArc(a, l, aPrime, betaDeg)
flank = (a - aPrime) / 2;
cx = 2 * a + flank;
cy = l / 2;
radius = 0.22 * min(a, l);
phi = (0:2:betaDeg) * pi / 180;
plot(cx + radius * cos(phi), cy + radius * sin(phi), 'k-', 'LineWidth', 1.0);
text(cx + radius * 1.15, cy + radius * 0.55, 'beta', ...
    'Interpreter', 'none', 'FontSize', 12);
end

function drawFolded(shape)
hold on;
faceColor = [
    0.93, 0.94, 0.95;
    0.98, 0.86, 0.82;
    0.78, 0.80, 0.82;
    0.98, 0.86, 0.82
];
for iFace = 1:4
    p = squeeze(shape.faces(iFace, :, :));
    patch('Vertices', p, 'Faces', [1, 2, 3, 4], ...
        'FaceColor', faceColor(iFace, :), 'FaceAlpha', 0.72, ...
        'EdgeColor', 'none');
    drawFaceBoundary(p);
    if mod(iFace, 2) == 0
        drawSideCreases3(p, shape.a, shape.aPrime);
    else
        drawPlainCrease3(p);
    end
end

drawHeightArrow(shape);
drawWidthArrow(shape);
drawThetaArc(shape);

axis equal;
grid on;
box on;
view(52, 24);
xlabel('x (mm)');
ylabel('y (mm)');
zlabel('z (mm)');
end

function drawFaceBoundary(p)
loop = [p; p(1, :)];
plot3(loop(:, 1), loop(:, 2), loop(:, 3), '-', ...
    'Color', [0.75, 0.12, 0.12], 'LineWidth', 1.7);
end

function drawPlainCrease3(p)
s = pointOnFace(p, 0, 0.5);
e = pointOnFace(p, 1, 0.5);
plot3([s(1), e(1)], [s(2), e(2)], [s(3), e(3)], 'b--', 'LineWidth', 1.5);
end

function drawSideCreases3(p, a, aPrime)
flank = (a - aPrime) / 2;
sP = flank / a;
sQ = (flank + aPrime) / a;
p0 = pointOnFace(p, 0, 0.5);
pP = pointOnFace(p, sP, 0.5);
pQ = pointOnFace(p, sQ, 0.5);
p1 = pointOnFace(p, 1, 0.5);
plotSegment(p0, pP, [0.75, 0.12, 0.12], '-');
plotSegment(pQ, p1, [0.75, 0.12, 0.12], '-');
plotSegment(pP, pQ, [0.1, 0.35, 0.85], '--');
plotSegment(pointOnFace(p, 0, 1), pP, [0.1, 0.35, 0.85], '--');
plotSegment(pointOnFace(p, 1, 1), pQ, [0.1, 0.35, 0.85], '--');
plotSegment(pointOnFace(p, 0, 0), pP, [0.1, 0.35, 0.85], '--');
plotSegment(pointOnFace(p, 1, 0), pQ, [0.1, 0.35, 0.85], '--');
end

function q = pointOnFace(p, s, t)
top = (1 - s) * p(1, :) + s * p(2, :);
bottom = (1 - s) * p(4, :) + s * p(3, :);
q = (1 - t) * top + t * bottom;
end

function plotSegment(p, q, color, style)
plot3([p(1), q(1)], [p(2), q(2)], [p(3), q(3)], style, ...
    'Color', color, 'LineWidth', 1.5);
end

function drawHeightArrow(shape)
xs = reshape(shape.faces(:, :, 1), [], 1);
x = min(xs) - 0.28 * shape.a;
zTop = shape.h / 2;
zBot = -shape.h / 2;
y = shape.l / 2;
plot3([x, x], [y, y], [zBot, zTop], 'k-', 'LineWidth', 1.0);
plot3(x, y, zTop, 'k^', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
plot3(x, y, zBot, 'kv', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
text(x, y, zTop + 0.12 * shape.h, 'h', 'Interpreter', 'none', ...
    'FontSize', 12, 'HorizontalAlignment', 'center');
end

function drawWidthArrow(shape)
p = squeeze(shape.faces(1, :, :));
z = shape.h / 2 + 0.14 * shape.h;
y = shape.l * 0.15;
x1 = min(p(:, 1));
x2 = max(p(:, 1));
plot3([x1, x2], [y, y], [z, z], 'k-', 'LineWidth', 1.0);
plot3(x1, y, z, 'k<', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
plot3(x2, y, z, 'k>', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
text((x1 + x2) / 2, y, z + 0.08 * shape.h, 'a', ...
    'Interpreter', 'none', 'FontSize', 12, 'HorizontalAlignment', 'center');
end

function drawThetaArc(shape)
p = squeeze(shape.faces(1, :, :));
right = squeeze(shape.faces(2, :, :));
origin = p(3, :);
vTop = p(4, :) - p(3, :);
vSide = right(3, :) - right(4, :);
vTop = vTop / norm(vTop);
vSide = vSide / norm(vSide);
radius = 0.28 * shape.a;
n = 16;
pts = zeros(n, 3);
for k = 1:n
    t = (k - 1) / (n - 1);
    d = (1 - t) * vTop + t * vSide;
    d = d / norm(d);
    pts(k, :) = origin + radius * d;
end
plot3(pts(:, 1), pts(:, 2), pts(:, 3), 'k-', 'LineWidth', 1.0);
mid = pts(round(n / 2), :);
text(mid(1) + 0.12 * shape.a, mid(2), mid(3), 'theta', ...
    'Interpreter', 'none', 'FontSize', 12);
end

function drawDoubleArrow(x1, x2, y1, y2, color)
plot([x1, x2], [y1, y2], '-', 'Color', color, 'LineWidth', 1.0);
d = [x2 - x1, y2 - y1];
n = norm(d);
if n < 1e-9
    return;
end
d = d / n;
left = [-d(2), d(1)];
head = 0.07 * max(abs(x2 - x1) + abs(y2 - y1), 1);
drawHead(x1, y1, d, left, head, color);
drawHead(x2, y2, -d, left, head, color);
end

function drawHead(x, y, d, left, head, color)
p1 = [x, y];
p2 = [x, y] + head * d + 0.45 * head * left;
p3 = [x, y] + head * d - 0.45 * head * left;
plot([p2(1), p1(1), p3(1)], [p2(2), p1(2), p3(2)], '-', 'Color', color, 'LineWidth', 1.0);
end
