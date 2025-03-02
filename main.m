warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
tic
% restoredefaultpath

%% 导入数据
respre = parquetread('data.parquet');
respre = table2array(respre);
%disp(respre);


%%  碱基转化为数值并形成新矩阵
res = zeros(size(respre, 1), length(respre{1, 1})+1); % 预分配
for i = 1:size(respre, 1)
    seq = respre{i, 1}; % 提取 DNA 序列
    result=respre{i, 2};
    num_seq = convert_seq_to_num(seq,result); % 转换为数值
    res(i, :) = num_seq; % 存入矩阵
end
%disp(res);


%%  数据分析
num_size = 0.8;                              % 训练集占数据集比例
num1_size = 0.9;                             % 验证集占数据集比例
outdim = 1;                                  % 最后一列为输出
num_samples = size(res, 1);                  % 样本个数
res = res(randperm(num_samples), :);         % 打乱数据集（不希望打乱时，注释该行）
num_train_s = ceil(num_size * num_samples)+1; % 训练集样本个数
num_val_s = ceil(num1_size * num_samples)+1; % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度

%%  划分训练集和测试集
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_val = res(num_train_s + 1: num_val_s, 1: f_)';
T_val = res(num_train_s + 1: num_val_s, f_ + 1: end)';
V = size(P_val, 2);

P_test = res(num_val_s + 1: end, 1: f_)';
T_test = res(num_val_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  数据平铺
p_train =  double(reshape(P_train, f_, 1, 1, M));
p_val = double(reshape(P_val, f_, 1, 1, V));
p_test  =  double(reshape(P_test , f_, 1, 1, N));

t_train1 =  double(T_train)';
t_val1 =  double(T_val)';
t_test1  =  double(T_test )';

t_train =  categorical(t_train1)';
t_val =  categorical(t_val1)';
t_test  =  categorical(t_test1)';
%%  数据格式转换
for i = 1 : M
    Lp_train{i, 1} = p_train(:, :, 1, i);
end

for i = 1 : V
    Lp_val{i, 1} = p_val(:, :, 1, i);
end

for i = 1 : N
    Lp_test{i, 1}  = p_test( :, :, 1, i);
end
    
%%  建立模型
lgraph = layerGraph();                                                 % 建立空白网络结构

tempLayers = [
    sequenceInputLayer([f_, 1, 1], "Name", "sequence")               % 建立输入层，输入数据结构为[f_, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = convolution2dLayer([5,1], 32, "Padding","same","Name", "conv_1");         % 卷积层 卷积核[1, 1] 步长[1, 1] 通道数 32
lgraph = addLayers(lgraph,tempLayers);                                 % 将上述网络结构加入空白结构中
 
tempLayers = [
    reluLayer("Name", "relu_1")                                        % 激活层
    convolution2dLayer([5,1], 64, "Padding","same","Name", "conv_2")                   % 卷积层 卷积核[1, 1] 步长[1, 1] 通道数 64
    reluLayer("Name", "relu_2")];                                      % 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    globalAveragePooling2dLayer("Name", "gapool")                      % 全局平均池化层
    fullyConnectedLayer(16, "Name", "fc_2")                            % SE注意力机制，通道数的1 / 4
    reluLayer("Name", "relu_3")                                        % 激活层
    fullyConnectedLayer(64, "Name", "fc_3")                            % SE注意力机制，数目和通道数相同
    sigmoidLayer("Name", "sigmoid")];                                  % 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的注意力
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
    flattenLayer("Name", "flatten")                                    % 网络铺平层
    lstmLayer(16, "Name", "lstm", "OutputMode", "last")                 % lstm层
    fullyConnectedLayer(2, "Name", "fc")                               % 2个类别输出
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classificationoutput")];
    %regressionLayer("Name", "regressionoutput")];                      % 回归层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入;
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
                                                                       % 折叠层输出 连接 反折叠层输入  
lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % 卷积层输出 链接 激活层
lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % 卷积层输出 链接 全局平均池化
lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % 激活层输出 链接 相乘层
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % 全连接输出 链接 相乘层
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % 点乘输出
%%  参数设置
options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MaxEpochs', 100, ...                 % 最大迭代次数
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 5e-4, ...          % 初始学习率为0.01
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.5, ...        % 学习率下降因子 0.5
    'LearnRateDropPeriod', 150, ...        % 经过700次训练后 学习率为 0.01 * 0.1
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'ValidationData',{Lp_val,t_val}, ...   %'ValidationFrequency', 10, ...
    'ValidationPatience', 20, ...
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);

%%  训练模型

net = trainNetwork(Lp_train, t_train, lgraph, options);

%%  模型预测
T_sim1pre = predict(net, Lp_train);
T_sim2pre = predict(net, Lp_test );
T_sim1 = T_sim1pre(:,2) > 0.5; % 选择 class_1 的概率大于 0.5 则预测为 class_1 (1)，否则为 class_0 (0)
T_sim2 = T_sim2pre(:,2) > 0.5; % 同理，测试集
T_test = reshape(T_test, [], 1);
T_sim2 = reshape(T_sim2, [], 1);
accuracy_train = sum(T_sim1 == T_train) / M; % 训练集准确率
accuracy_test = sum(T_sim2 == T_test) / N;   % 测试集准确率
disp(size(T_test));  % 输出 T_test 的维度
disp(size(T_sim2));
%%  数据反归一化
%T_sim1 = mapminmax('reverse', T_sim1', ps_output);
%T_sim2 = mapminmax('reverse', T_sim2', ps_output);
%T_sim1=double(T_sim1);
%T_sim2=double(T_sim2);
%%  显示网络结构
 analyzeNetwork(net)

%% 测试集结果

T_sim2_cat = categorical(T_sim2, [0, 1]);  % [0, 1] 是类别标签

% 如果 T_test 是数值型（0/1），将其转换为 categorical 类型
T_test_cat = categorical(T_test, [0, 1]);
figure;
plotconfusion(T_test_cat, T_sim2_cat, '分类结果');

%%  混淆矩阵和相关指标
% 计算混淆矩阵
confMat = confusionmat(T_test_cat, T_sim2_cat);  % 预测与真实标签

% 从混淆矩阵中提取 True Positive, False Positive, True Negative, False Negative
TP = confMat(2, 2); % True Positive
FP = confMat(1, 2); % False Positive
TN = confMat(1, 1); % True Negative
FN = confMat(2, 1); % False Negative

% 计算 Precision, Recall, F1-score
precision = TP / (TP + FP);   % 精确率
recall = TP / (TP + FN);      % 召回率
f1_score = 2 * (precision * recall) / (precision + recall);  % F1-score

%%  输出分类指标
disp(['-----------------------分类指标--------------------------'])
disp(['训练集准确率: ', num2str(accuracy_train * 100), '%']);
disp(['测试集准确率: ', num2str(accuracy_test * 100), '%']);
disp(['精确率 (Precision): ', num2str(precision)]);
disp(['召回率 (Recall): ', num2str(recall)]);
disp(['F1-score: ', num2str(f1_score)]);
%%  输出测试集准确率
disp(['测试集分类准确率为：', num2str(accuracy_test * 100), '%']);

%%  分类指标记录 储存用
classification_results = [accuracy_train, accuracy_test, precision, recall, f1_score];
%%  碱基数值映射函数
function num_seq = convert_seq_to_num(seq,result)
    % 将 DNA 序列转换为数值表示
    num_seq = zeros(1, length(seq)+1); % 预分配数组
    for i = 1:length(seq)
        switch seq(i)
            case 'A'
                num_seq(i) = 1;
            case 'T'
                num_seq(i) = 2;
            case 'C'
                num_seq(i) = 3;
            case 'G'
                num_seq(i) = 4;
            otherwise
                num_seq(i) = NaN; % 处理未知碱基
        end
    end
    num_seq(length(seq)+1)=result-'0';
end

