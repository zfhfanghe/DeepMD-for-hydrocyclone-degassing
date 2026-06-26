% bubble_radius.m
% 计算 40 个 H2 分子气泡在水/气界面张力作用下的稳态平衡半径，并绘制收敛情况

% ———— 参数设定 ————
N      = 60;                 % 分子数
k_B    = 1.380649e-23;       % 波尔兹曼常数 (J/K)
gamma  = 0.072;              % 水/气界面张力 (N/m)
T      = 298.15;                % 温度 (K)，可根据需要修改
P_out  = 111457;          % 外部压力 (Pa)，1 atm
delta  = 0.3e-9;                % Tolman修正系数

% ———— 初始猜测与迭代设定 ————
r0      = 2e-9;              % 初始猜测半径 (m)，约 2 nm
max_iter = 1000;             % 最大迭代次数
tol      = 1e-12;            % 收敛阈值

% ———— 收敛迭代求解 (固定点迭代) ————
r_vals = zeros(max_iter,1);
res     = zeros(max_iter,1);
r_vals(1) = r0;
for k = 2:max_iter
    % 更新公式：r = (3*N*k_B*T / (4*pi*(P_out + 2*gamma/(r+2*delta))))^(1/3)
    r_vals(k) = (3*N*k_B*T / (4*pi*(P_out + 2*gamma./(r_vals(k-1)+2*delta))))^(1/3);
    res(k)     = abs(r_vals(k) - r_vals(k-1));
    if res(k) < tol
        break;
    end
end
iter = k;
r_sol = r_vals(iter);

% ———— 输出结果 ————
fprintf('收敛至 tol = %.1e ，耗时迭代 %d 次\n', tol, iter);
fprintf('稳态平衡气泡半径：%.5e m (%.5f nm)\n', r_sol, r_sol*1e9);

% ———— 绘制收敛情况 ————
figure;
subplot(2,1,1);
plot(1:iter, r_vals(1:iter)*1e9, '-o', 'MarkerSize',4, 'LineWidth',1);
xlabel('迭代次数');
ylabel('半径 (nm)');
title('气泡半径收敛曲线');
grid on;

subplot(2,1,2);
semilogy(2:iter, res(2:iter), '-o', 'MarkerSize',4, 'LineWidth',1);
xlabel('迭代次数');
ylabel('残差 |r_k - r_{k-1}|');
title('收敛残差曲线');
grid on;
