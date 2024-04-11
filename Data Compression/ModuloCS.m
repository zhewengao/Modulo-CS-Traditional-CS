
clear;
clc;
close all;


% 设置MNIST数据集的路径
mnistTrainPath = '/Users/kevingao/Desktop/Data Compression/mnist_dataset/mnist_train.csv';


% 读取训练数据
trainData = readtable(mnistTrainPath);
trainLabels = trainData{:, 1}; % 第一列是标签
trainImages = table2array(trainData(:, 2:end)); % 第二列到最后一列是图像像素值

% 打印trainImages的形状
disp(['trainImages的形状：', num2str(size(trainImages))]);


% 转换图像数据为28x28x1的形状，并调整图像方向
trainImagesplay = reshape(trainImages', [28, 28, 1, size(trainImages, 1)]);

% 选择要显示的图像的索引
index = 200; % 选择第100张图像

% 显示选定的训练图像
figure('Position', [100, 100, 800, 600]); % Adjust the position and size as needed
imshow(trainImagesplay(:,:,1,index)', 'InitialMagnification', 'fit');
title(['Label: ', num2str(trainLabels(index))]);
% 使用transpose函数进行转置
trainImages = transpose(trainImages);
% 缩放图像到[0, 1]
min_val = min(trainImages(:));
max_val = max(trainImages(:));
trainImages = (trainImages - min_val) / (max_val - min_val);


% 打印转置后的trainImages的形状
disp(['转置后 trainImages的形状：', num2str(size(trainImages))]);


% 设置矩阵的列数
N = 784;
% 设置行数的范围
M_values = [400, 500];

% 初始化一个cell数组来存储生成的矩阵A
A_storage = cell(1, length(M_values));

% 初始化一个cell数组来存储投影结果y
y_storage = cell(1, length(M_values));

% 选择要投影的列索引
col_index = 200; % 选择第200列

% 选择要投影的列
x = trainImages(:, col_index);
s = nnz(x);

% 生成A矩阵并进行投影操作
for i = 1:length(M_values)

    M = M_values(i);

    % 从均值为0，标准差为1/sqrt(M)的高斯分布中抽取随机矩阵
    A = randn(M, N) / sqrt(M);
    % 确保矩阵是可逆的
    while rank(A) < min(M, N)
        A = randn(M, N)/ sqrt(M);
    end
    disp(rank(A));

    % 对每一列进行L2范数归一化
    A = A ./ vecnorm(A, 2, 1);
  
    % 存储生成的矩阵A
    A_storage{i} = A;
    
    
    % 计算投影
    y = A * x;
    
    % 存储投影结果y
    y_storage{i} = y;
    
    % 显示投影结果的长度
    disp(['当M为', num2str(M_values(i)), '时，投影的长度：', num2str(norm(y))]);
end


M_test = 500; % 测试M为200的情况
index_test = find(M_values == M_test);
M = index_test;

z = y_storage{M} - floor(y_storage{M}); % 选择第m个投影结果的小数部分作为z

% 定义整数线性规划问题
n = 784;
f = [ones(n, 1); ones(n, 1);zeros(M_test,1)]; % 目标函数系数
intcon = 2 * n + 1:2 * n + M_test;  % 整数变量索引

% 等式约束
x_plus = optimvar('x_plus', n, 1, 'Type', 'continuous', 'LowerBound', 0);
x_minus = optimvar('x_minus', n, 1, 'Type', 'continuous', 'LowerBound', 0);
v = optimvar('v', M_test, 1, 'Type', 'integer');

% 打印矩阵的形状
disp(['形状 of [A_storage{M}, -A_storage{M}, -eye(M_test)]: ', num2str(size([A_storage{M}, -A_storage{M}, -eye(M_test)]))]);
A_eq = [A_storage{M}, -A_storage{M}, -eye(M_test)];
b_eq = z;

% 检查 A_eq 和 b_eq 的形状
disp(['形状 of A_eq: ', num2str(size(A_eq))]);
disp(['形状 of b_eq: ', num2str(size(b_eq))]);

% 非负约束
lb = [zeros(n, 1); zeros(n, 1)];

% 求解整数线性规划问题
[x_opt, fval] = intlinprog(f,intcon, [], [], A_eq, b_eq, lb);

% 提取解
x_plus = x_opt(1:n);
x_minus = x_opt(n+1:2*n);
v = x_opt(2*n+1:end);

% 计算x
x = x_opt(1:n) -x_opt(n+1:2*n);

res = nnz(x);
disp(['x中非0元素的个数：', num2str(res)]);


% 将x reshape为28x28图像
reconstructed_image = reshape(x, [28, 28]);

% 显示重构的图像
figure('Position', [100, 100, 800, 600]); % Adjust the position and size as needed
imshow(reconstructed_image', 'InitialMagnification', 'fit');
title('Reconstructed Image');

% 显示结果
disp('Optimal v (integer part):');
disp(v');

disp(['Optimal objective value: ', num2str(fval)]);