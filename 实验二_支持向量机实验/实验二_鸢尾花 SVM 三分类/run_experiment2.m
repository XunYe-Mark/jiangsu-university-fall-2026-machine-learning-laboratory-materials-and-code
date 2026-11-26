%% 实验二：鸢尾花 SVM 三分类
% 需要 Statistics and Machine Learning Toolbox。

%% 1. 加载数据并训练分类模型
irisData = load('fisheriris', 'meas', 'species');
features = irisData.meas;
species = irisData.species;
classNames = {'setosa', 'versicolor', 'virginica'};

% 各二分类 SVM 用其训练数据计算标准化参数。
svmTemplate = templateSVM('Standardize', true);
model = fitcecoc(features, species, 'Learners', svmTemplate, ...
    'ClassNames', classNames, 'Coding', 'onevsone');

%% 2. 进行 10 折交叉验证
foldCount = 10;
seed = 0;
previousRng = rng;
rng(seed, 'twister');
crossValidatedModel = crossval(model, 'KFold', foldCount);
rng(previousRng);
classificationLoss = kfoldLoss(crossValidatedModel);

%% 3. 预测讲义给定样本
% 特征顺序为萼片长度、萼片宽度、花瓣长度、花瓣宽度，单位为 cm。
newData = [5.1 3.5 1.4 0.2; ...
           6.7 3.0 5.2 2.3];
predictedLabels = predict(model, newData);

%% 4. 显示并保存结果
resultText = sprintf([ ...
    '实验二　鸢尾花 SVM 三分类\n' ...
    '数据集：fisheriris（%d 个样本，%d 个特征，%d 个类别）\n' ...
    '模型：线性 SVM，特征标准化，一对一 ECOC\n' ...
    '%d 折交叉验证（随机种子 %d）\n' ...
    '分类损失：%.6f（误分类率 %.2f%%）\n' ...
    '分类准确率：%.2f%%\n\n' ...
    '讲义指定样本的预测（萼片长、萼片宽、花瓣长、花瓣宽，单位 cm）\n' ...
    '样本 1  [%.1f, %.1f, %.1f, %.1f]  %s\n' ...
    '样本 2  [%.1f, %.1f, %.1f, %.1f]  %s\n' ...
    '这两个样本可在原数据集中找到，用于演示预测。模型性能由交叉验证评估。\n'], ...
    size(features, 1), size(features, 2), numel(classNames), ...
    foldCount, seed, classificationLoss, 100 * classificationLoss, ...
    100 * (1 - classificationLoss), newData(1, :), predictedLabels{1}, ...
    newData(2, :), predictedLabels{2});

fprintf('%s', resultText);
resultPath = fullfile(fileparts(mfilename('fullpath')), '实验二结果.txt');
[fileID, message] = fopen(resultPath, 'wt', 'n', 'UTF-8');
if fileID == -1
    error('无法打开结果文件 %s：%s', resultPath, message);
end
fprintf(fileID, '%s', resultText);
fclose(fileID);
