function shannonEnt = calShannonEnt(y)
% 计算信息熵


%%%%%================= 下面语句请同学们完成，计算所输入样本集的信息熵 ==================

N = length(y);
P_T = sum(y) / N;
P_F = (N - sum(y)) / N;

% 当某一类的概率为 0 时，样本全属同一类，信息熵为 0。
if P_T == 0 || P_F == 0
    shannonEnt = 0;
    return
end
shannonEnt = -(P_T * log2(P_T) + P_F * log2(P_F));

%%%%%%===============================================================================



