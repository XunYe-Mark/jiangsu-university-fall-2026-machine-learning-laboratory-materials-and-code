%% 实验一：线性支持向量机二分类
% 对应讲义第 7–11 页。需要 Statistics and Machine Learning Toolbox。

%% 1. 生成两类训练样本
previousRng = rng;
rng(0, 'twister');

% 讲义未注明随机状态。将其散点与 MATLAB 随机序列匹配后，
% 找到下列起点，使生成的样本位置与参考图对应。
sampleOffset = 2241;
randn(sampleOffset, 1);

nPerClass = 100;
sigma = 0.75;
dataClass1 = sigma * randn(nPerClass, 2) + [1, 1];
dataClass2 = sigma * randn(nPerClass, 2) + [-1, -1];
rng(previousRng);
data = [dataClass1; dataClass2];
groups = [ones(nPerClass, 1); -ones(nPerClass, 1)];

%% 2. 训练线性 SVM 并预测
SVMModel = fitcsvm(data, groups, 'KernelFunction', 'linear', 'BoxConstraint', 1);
supportVectors = SVMModel.SupportVectors;

newData = [0.5, 0.5; -0.5, -0.5];
predictedLabels = predict(SVMModel, newData);
predictionText = sprintf( ...
    'Predicted class labels for the new data points:\n    %d\n    %d', ...
    predictedLabels);
fprintf('%s\n', predictionText);

%% 3. 按讲义绘制训练数据
trainingFigure = figure('Name', 'Training Data', 'NumberTitle', 'off', ...
    'Color', 'white', 'WindowStyle', 'normal', 'Position', [80, 150, 640, 520]);
trainingAxes = axes(trainingFigure, 'Position', [0.13, 0.14, 0.80, 0.78]);
hold(trainingAxes, 'on');
class1 = scatter(trainingAxes, dataClass1(:, 1), dataClass1(:, 2), 36, 'r', 'o');
class2 = scatter(trainingAxes, dataClass2(:, 1), dataClass2(:, 2), 36, 'b', 'o');
formatAxes(trainingAxes, 'Training Data', 'off');
legend(trainingAxes, [class1, class2], {'Class 1', 'Class 2'}, ...
    'Location', 'northeast');

%% 4. 按讲义绘制并保存实验结果
resultFigure = figure('Name', 'SVM with linear kernel', 'NumberTitle', 'off', ...
    'Color', 'white', 'WindowStyle', 'normal', 'Position', [160, 100, 700, 650]);
resultAxes = axes(resultFigure, 'Position', [0.14, 0.12, 0.76, 0.67]);

% gscatter 按标签排序，所以 -1 对应红色叉号，+1 对应蓝色空心圆。
gscatter(resultAxes, data(:, 1), data(:, 2), groups, 'rb', 'xo', [], 'off');
hold(resultAxes, 'on');
supportMarker = plot(resultAxes, supportVectors(:, 1), supportVectors(:, 2), ...
    'ko', 'MarkerSize', 8);
formatAxes(resultAxes, 'SVM with linear kernel', 'on');
resultLegend = legend(resultAxes, supportMarker, 'Support Vectors', 'Location', 'east');
drawnow;
resultLegend.Position(2) = resultLegend.Position(2) - 0.03;
resultAxes.Toolbar.Visible = 'off';
annotation(resultFigure, 'textbox', [0.07, 0.84, 0.86, 0.12], ...
    'String', predictionText, 'Interpreter', 'none', 'EdgeColor', 'none', ...
    'FontName', 'Arial', 'FontSize', 13, 'Color', 'black');

outputFile = fullfile(fileparts(mfilename('fullpath')), '实验一结果.png');
exportgraphics(resultFigure, outputFile, 'Resolution', 180, ...
    'BackgroundColor', 'white', 'Padding', 14);

function formatAxes(ax, plotTitle, boxState)
% 设置讲义两张图共用的坐标范围、刻度和标签。
set(ax, 'XLim', [-4, 4], 'YLim', [-4, 3], 'XTick', -4:4, 'YTick', -4:3, ...
    'Box', boxState, 'TickDir', 'in', 'FontName', 'Arial', 'FontSize', 10, ...
    'XColor', 'black', 'YColor', 'black', 'Color', 'white');
grid(ax, 'off');
title(ax, plotTitle, 'FontWeight', 'normal', 'Color', 'black');
xlabel(ax, 'Feature 1');
ylabel(ax, 'Feature 2');
end
