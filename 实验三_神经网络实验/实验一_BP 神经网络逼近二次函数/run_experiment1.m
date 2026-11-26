%% 实验一：BP 神经网络逼近二次函数
% 对照《神经网络实验.pdf》第 6–9 页。在 MATLAB 编辑器中运行整个脚本。
outputFolder = fileparts(mfilename('fullpath'));

%% 初始化权值与训练样本
% 讲义使用 rands 初始化，未提供原始权值或随机种子。
% 为复现第 9 页的趋势，下列初始权值由图中误差曲线估计。
learningRate = 0.05;
momentum = 0.05;
sampleIndex = 1:500;
x = sampleIndex / 100;
y = 0.3 * x.^2 - 2 * x - 1;

initialW2 = [0.26653769; -0.44882971; -0.87118000; 0.24736646; -0.41141352; -0.15487366];
initialW1 = [0.56516904  0.18477757   0.34557093  -0.78842148  0.58799215  -0.43827121
             0.74890885  0.46648931  -0.35680774   0.85700127  0.73301323   0.99999999];
w2 = initialW2;
w1 = initialW1;
previousW2 = w2;
previousW1 = w1;
yn = zeros(size(y));
e = zeros(size(y));
jacobianX = zeros(size(y));

%% 逐样本前向计算和带动量的权值更新
for k = sampleIndex
    % 按讲义构造两个输入，第二个输入是当前样本的真实输出。
    sampleInput = [x(k); y(k)];
    hidden = 1 ./ (1 + exp(-(w1.' * sampleInput)));
    hiddenSlope = hidden .* (1 - hidden);
    yn(k) = w2.' * hidden;
    e(k) = y(k) - yn(k);

    % 先更新输出层，再用本步的新输出层权值更新输入至隐藏层权值。
    nextW2 = w2 + learningRate * e(k) * hidden + momentum * (w2 - previousW2);
    hiddenDelta = learningRate * e(k) * (hiddenSlope .* nextW2);
    nextW1 = w1 + sampleInput * hiddenDelta.' + momentum * (w1 - previousW1);

    % 依照第 8 页，用本步前向传播的斜率和更新后的权值记录灵敏度。
    jacobianX(k) = nextW1(1, :) * (nextW2 .* hiddenSlope);
    previousW2 = w2;
    previousW1 = w1;
    w2 = nextW2;
    w1 = nextW1;
end

%% 按讲义第 9 页绘制两幅结果图
fig = figure('Name', '实验一 BP 神经网络函数逼近', 'NumberTitle', 'off', ...
    'Color', 'white', 'Position', [100 100 1100 390], ...
    'DefaultAxesFontName', 'Arial', 'DefaultAxesFontSize', 12);
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'loose');

fitAxes = nexttile(layout);
plot(fitAxes, sampleIndex, y, 'r-', 'LineWidth', 2.4);
hold(fitAxes, 'on');
plot(fitAxes, sampleIndex, yn, 'b--', 'LineWidth', 2.4);
hold(fitAxes, 'off');
title(fitAxes, 'BP神经网络系统逼近', 'FontName', 'Microsoft YaHei');
xlabel(fitAxes, 'times');
ylabel(fitAxes, 'y and yn');
legend(fitAxes, {'y', 'yn'}, 'Location', 'northeast', 'Color', 'white');
ylim(fitAxes, [-4.5 -0.5]);
yticks(fitAxes, -4.5:0.5:-0.5);

errorAxes = nexttile(layout);
plot(errorAxes, sampleIndex, e, 'r-', 'LineWidth', 1.25);
title(errorAxes, '误差曲线', 'FontName', 'Microsoft YaHei');
xlabel(errorAxes, 'times');
ylabel(errorAxes, 'error');
ylim(errorAxes, [-0.35 0.1]);
yticks(errorAxes, -0.35:0.05:0.1);

for ax = [fitAxes errorAxes]
    xlim(ax, [0 500]);
    xticks(ax, 0:100:500);
    grid(ax, 'on');
    box(ax, 'on');
    ax.GridAlpha = 0.55;
    ax.Toolbar.Visible = 'off';
end

[minError, minSample] = min(e);
fprintf('第 1 点误差 %.6f，第 %d 点达到最小误差 %.6f，第 500 点误差 %.6f。\n', ...
    e(1), minSample, minError, e(end));
exportgraphics(layout, fullfile(outputFolder, '实验一结果.png'), ...
    'Resolution', 180, 'BackgroundColor', 'white');
