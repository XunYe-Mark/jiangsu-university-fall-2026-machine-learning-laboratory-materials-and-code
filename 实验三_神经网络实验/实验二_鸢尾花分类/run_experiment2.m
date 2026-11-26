%% 实验二：用 BP 神经网络对鸢尾花分类
% 在 MATLAB 编辑器中点击“运行”。
% 需要 Deep Learning Toolbox，以及用于示例数据和混淆矩阵的 Statistics and Machine Learning Toolbox。

%% 按类别随机划分训练集和测试集
rng(42, 'twister');
irisData = load('fisheriris', 'meas', 'species');
classNames = {'setosa', 'versicolor', 'virginica'};
numClasses = numel(classNames);
samplesPerClass = 25;
trainIndices = zeros(numClasses * samplesPerClass, 1);
testIndices = zeros(size(trainIndices));
classIds = zeros(size(irisData.species));

for classId = 1:numClasses
    indices = find(strcmp(irisData.species, classNames{classId}));
    classIds(indices) = classId;
    shuffled = indices(randperm(numel(indices)));
    rows = (classId - 1) * samplesPerClass + (1:samplesPerClass);
    trainIndices(rows) = shuffled(1:samplesPerClass);
    testIndices(rows) = shuffled(samplesPerClass + 1:end);
end

%% 归一化和目标编码
% 样本按列排列。按训练集各特征的最小值和最大值映射到 [0,1]，
% 测试集应用同一组参数，因此测试样本超出训练范围时可落在区间外。
[trainInputs, scaling] = mapminmax(irisData.meas(trainIndices, :).', 0, 1);
testInputs = mapminmax('apply', irisData.meas(testIndices, :).', scaling);
testLabels = classIds(testIndices);
% 一位有效编码：该样本所属类对应的输出为 1，其余输出为 0。
trainTargets = double((1:numClasses).' == classIds(trainIndices).');

%% 按指导书的结构和参数训练
% 4 个输入，两个隐藏层分别有 10、5 个节点，输出层有 3 个节点。
% 使用三个 tansig 层、批量梯度下降和均方误差，无额外预处理或再次划分。
% 单独固定初始化种子，使其独立于数据划分中随机数的调用。
rng(42, 'twister');
% 用 feedforwardnet 搭建与讲义 newff 示例相同的网络。
net = feedforwardnet([10 5], 'traingd');
net.layers{3}.transferFcn = 'tansig';
net.inputs{1}.processFcns = {};
net.outputs{3}.processFcns = {};
net.divideFcn = 'dividetrain';
net.performFcn = 'mse';
net = configure(net, trainInputs, trainTargets);
net.trainParam.show = 50;
net.trainParam.epochs = 500;
net.trainParam.goal = 0.01;
net.trainParam.lr = 0.01;
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = false;
[net, trainingRecord] = train(net, trainInputs, trainTargets);

%% 测试与准确率
scores = net(testInputs);
[~, predictedLabels] = max(scores, [], 1);
predictedLabels = predictedLabels.';
numCorrect = nnz(predictedLabels == testLabels);
accuracy = 100 * numCorrect / numel(testLabels);
confusion = confusionmat(testLabels, predictedLabels, 'Order', 1:numClasses);
finalMSE = trainingRecord.perf(end);

fprintf('\n实验二  鸢尾花 BP 分类\n');
fprintf('编号与类别  1 = %s，2 = %s，3 = %s\n', classNames{:});
fprintf('每类训练 %d 个、测试 %d 个样本，训练集与测试集各 %d 个。\n', ...
    samplesPerClass, samplesPerClass, numel(testLabels));
fprintf('训练轮数 %d，最终训练 MSE %.6f，目标 MSE %.2f。\n', ...
    trainingRecord.epoch(end), finalMSE, net.trainParam.goal);
fprintf('测试准确率 %.2f%%（%d/%d）。\n', accuracy, numCorrect, numel(testLabels));
fprintf('混淆矩阵（行 = 真实类别，列 = 预测类别，顺序 1、2、3）\n');
disp(confusion);

%% 绘图并保存结果
fig = figure('Name', '实验二｜鸢尾花分类', 'NumberTitle', 'off', ...
    'Color', 'white', 'Position', [100 100 1110 500]);
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'loose', 'Padding', 'compact');
fontName = 'Microsoft YaHei';
title(layout, '实验二　鸢尾花 BP 神经网络分类', 'FontName', fontName, 'FontSize', 17);

lossAxes = nexttile(layout);
semilogy(lossAxes, trainingRecord.epoch, trainingRecord.perf, ...
    'LineWidth', 2, 'Color', [0.05 0.41 0.55]);
yline(lossAxes, net.trainParam.goal, '--', '训练目标 0.01', ...
    'Color', [0.72 0.27 0.17], 'LabelHorizontalAlignment', 'left', 'FontName', fontName);
set(lossAxes, 'FontName', fontName, 'FontSize', 11, 'Layer', 'top');
grid(lossAxes, 'on');
title(lossAxes, sprintf('训练误差（最终 MSE %.4f）', finalMSE));
xlabel(lossAxes, '训练轮数');
ylabel(lossAxes, '均方误差（MSE）');

matrixAxes = nexttile(layout);
imagesc(matrixAxes, confusion, [0 samplesPerClass]);
axis(matrixAxes, 'image');
fade = linspace(0, 1, 256).';
colormap(matrixAxes, 1 - fade .* [0.91 0.49 0.36]);
classTickLabels = string(1:numClasses) + " " + string(classNames);
set(matrixAxes, 'XTick', 1:numClasses, 'YTick', 1:numClasses, ...
    'XTickLabel', classTickLabels, 'YTickLabel', classTickLabels, ...
    'FontName', fontName, 'FontSize', 11, 'TickLength', [0 0]);
for trueClass = 1:numClasses
    for predictedClass = 1:numClasses
        count = confusion(trueClass, predictedClass);
        textColor = [0.1 0.2 0.3];
        if count > samplesPerClass / 2
            textColor = [1 1 1];
        end
        text(matrixAxes, predictedClass, trueClass, num2str(count), ...
            'Color', textColor, 'FontSize', 14, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center');
    end
end
title(matrixAxes, sprintf('测试准确率 %.2f%%（%d/%d）', accuracy, numCorrect, numel(testLabels)));
xlabel(matrixAxes, '预测类别');
ylabel(matrixAxes, '真实类别');

resultFile = fullfile(fileparts(mfilename('fullpath')), '鸢尾花分类结果.png');
exportgraphics(layout, resultFile, 'Resolution', 180);
