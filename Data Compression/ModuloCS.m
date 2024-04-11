
clear all;



% 设置MNIST数据集的路径
mnistTrainPath = 'E:\data compression\project matlab\mnist_train.csv';
%mnistTrainPath = 'E:\data compression\project matlab\mnist_test.csv';

% 读取训练数据
trainData = readtable(mnistTrainPath);
trainLabels = trainData{:, 1}; % 第一列是标签
trainImages = table2array(trainData(:, 2:end)); % 第二列到最后一列是图像像素值
%disp(trainImages)
% 打印trainImages的形状
disp(['trainImages的形状：', num2str(size(trainImages))]);


% 转换图像数据为28x28x1的形状，并调整图像方向
trainImagesplay = reshape(trainImages', [28, 28, 1, size(trainImages, 1)]);

% 选择要显示的图像的索引
index = 123; % 选择第100张图像

% % 显示选定的训练图像
% figure;
% imshow(trainImagesplay(:,:,1,index)');
% title(['Label: ', num2str(trainLabels(index))]);

% 使用transpose函数进行转置
%threshold
%trainImages(trainImages <=128) = 0;

trainImages = transpose(trainImages);
% 缩放图像到[0, 1]
% min_val = min(trainImages(:));
% max_val = max(trainImages(:));
% trainImages = (trainImages - min_val) / (max_val - min_val);
trainImages = (trainImages/255);
% 显示选定的训练图像
figure;
imshow(trainImagesplay(:,:,1,index)','InitialMagnification', 'fit');
title(['Label: ', num2str(trainLabels(index))]);


% 打印转置后的trainImages的形状
disp(['转置后 trainImages的形状：', num2str(size(trainImages))]);


% 设置矩阵的列数
N = 784;
% 设置行数的范围
M_values = [200,250,300,350,400,450,500,550,600,650,700];

% 初始化一个cell数组来存储生成的矩阵A
A_storage = cell(1, length(M_values));

% 初始化一个cell数组来存储投影结果y
y_storage = cell(1, length(M_values));

% 选择要投影的列索引
col_index = index; % 选择第100列

% 选择要投影的列
x = trainImages(:, col_index);
s = nnz(x);
%disp(x)
%disp(['x中非0元素的个数：', s);




% 生成A矩阵并进行投影操作
for i = 1:length(M_values)
    M = M_values(i);
    
    % 从均值为0，标准差为1/sqrt(M)的高斯分布中抽取随机矩阵
    A = random_gaussian_matrix(M,784,M);
    % 确保矩阵是可逆的
    while rank(A) < min(M, N)
       A = random_gaussian_matrix(M,784,M);
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

% % 测试：调用不同M值下的投影结果y
% M_test = 200; % 测试M为200的情况
% index_test = find(M_values == M_test);
% 
% if ~isempty(index_test)
%     disp(['当M为', num2str(M_test), '时，投影结果y为：']);
%     disp(y_storage{index_test});
% else
%     disp(['没有找到M为', num2str(M_test), '的投影结果y。']);
% end
% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 计算z

M_test =700; % 测试M为200的情况
index_test = find(M_values == M_test);
M = index_test;

z = y_storage{M} - floor(y_storage{M}); % 选择第m个投影结果的小数部分作为z
%disp(['当M为',z])
%v = y_storage{M} - z




% MILP probelm
n = 784;
f = [ones(n, 1); ones(n, 1);zeros(M_test,1)]; 
intcon = 2*n + 1:2*n + M_test; 

% equality constraint

x_plus = optimvar('x_plus', n, 1);
x_minus = optimvar('x_minus', n, 1);
v = optimvar('v', M_test, 1, 'Type', 'integer');

% 打印矩阵的形状
%disp(['形状 of [A_storage{M}, -A_storage{M}, -eye(M_test)]: ', num2str(size([A_storage{M}, -A_storage{M}, -eye(M_test)]))]);
%disp(['形状 of transpose([ x_plus; x_minus; v_var]): ', num2str(size([x_plus; x_minus; v]))]);


%A_eq = [A_storage{M}, -A_storage{M}, -eye(M_test)] * [x_plus; x_minus; v];
A_eq = [A_storage{M}, -A_storage{M}, -eye(M_test)];
b_eq = z;

% 
disp(['形状 of A_eq: ', num2str(size(A_eq))]);
disp(['形状 of b_eq: ', num2str(size(b_eq))]);

%A_eq_double = double(value(A_eq));

% non negetive constrain
lb = [zeros(n, 1); zeros(n, 1)];



% % 求解整数线性规划问题
% [x_opt, fval] = intlinprog(f,intcon, [], [], A_eq, b_eq,lb);

options = optimoptions('intlinprog', 'MaxTime', 600); 


% 求解整数线性规划问题
%[x_opt, fval] = intlinprog(f, intcon, [], [], A_eq, b_eq, lb);

[x_opt, fval] = intlinprog(f, intcon,[], [], A_eq, b_eq, lb, [], [], options);

%the solution
disp(x_opt)
x_plus = x_opt(1:n);
x_minus = x_opt(n+1:2*n);
v = x_opt(2*n+1:end);


% get x_reconstructed
x_reconstruted = x_opt(1:n) -x_opt(n+1:2*n);

res = nnz(x_reconstruted);
disp(['x中非0元素的个数：', num2str(res)]);
error = (x_reconstruted - x)*255;
MSE = mean(error.^2);
% disp(['error='])
% disp(error)
% disp(MSE)
disp(['当M为', num2str(M_test), '时，MSE为：', num2str(MSE)]);

% 将x reshape为28x28图像
reconstructed_image = reshape(x_reconstruted, [28, 28]);

% 显示重构的图像
figure;
%imshow(reconstructed_image');
imshow(reconstructed_image', 'InitialMagnification', 'fit');
title('Reconstructed Image');
% 
% % % 显示结果
% % disp('Optimal x+:');
% % disp(x_plus');
% % 
% % disp('Optimal x-:');
% % disp(x_minus');
% % 
% % disp('Optimal v (integer part):');
% % disp(v');
% % 111111111111111111
% disp(['Optimal objective value: ', num2str(fval)]);
% % 计算MSE并重构图像
% for m = 1:length(M_values)
%     M_test = M_values(m);
%     index_test = find(M_values == M_test);
%     M = index_test;
% 
%     z = y_storage{M} - floor(y_storage{M});
% 
%     % 定义整数线性规划问题
%     n = 784;
%     f = [ones(n, 1); ones(n, 1); zeros(M_test, 1)];
%     intcon = 2 * n + 1:2 * n + M_test;
% 
%     x_plus = optimvar('x_plus', n, 1);
%     x_minus = optimvar('x_minus', n, 1);
%     v = optimvar('v', M_test, 1, 'Type', 'integer');
% 
%     A_eq = [A_storage{M}, -A_storage{M}, -eye(M_test)];
%     b_eq = z;
% 
%     lb = [zeros(n, 1); zeros(n, 1)];
% 
%     options = optimoptions('intlinprog', 'MaxTime', 300);
% 
%     [x_opt, ~] = intlinprog(f, intcon, [], [], A_eq, b_eq, lb, [], [], options);
% 
%     x_reconstructed = x_opt(1:n) - x_opt(n + 1:2 * n);
% 
%     error = (x_reconstructed - x)*255;
%     MSE = mean(error.^2);
%     disp(['当M为', num2str(M_test), '时，MSE为：', num2str(MSE)]);
% 
%     % 将x reshape为28x28图像
%     reconstructed_image = reshape(x_reconstructed, [28, 28]);
% 
%     % 显示重构的图像
%     figure;
%     imshow(reconstructed_image');
%     title(['Reconstructed Image for M = ', num2str(M_test)]);
% end
