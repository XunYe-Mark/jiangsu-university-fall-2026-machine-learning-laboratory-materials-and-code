function bestFeature=chooseFeature(dataset,~)
% 选择信息增益最大的属性特征
%%%%%=================  下面语句请同学们完成，循环每一个特征计算信息增益，选取信息增益最大的特征  ==================

[N, M] = size(dataset);
M = M - 1;

% 以第一个样本的类别为 1，另一类为 0。
y = strcmp(dataset(:, M+1), dataset(1, M+1));
x = dataset(:, 1:M);
gain = zeros(1, M);
Ent_D = calShannonEnt(y);

for i = 1:M
    temp = tabulate(x(:, i));
    value = temp(:, 1);
    count = cell2mat(temp(:, 2));
    Kind_Num = length(value);
    Ent = zeros(Kind_Num, 1);
    for j = 1:Kind_Num
        Ent(j) = calShannonEnt(y(strcmp(x(:, i), value(j))));
    end
    gain(i) = Ent_D - count' / N * Ent;
end

% 信息增益最大值并列时，从并列特征中随机选一个。
max_gain = find(gain == max(gain));
choose = randi(length(max_gain));
bestFeature = max_gain(choose);


%%%%===================================================================================
 

