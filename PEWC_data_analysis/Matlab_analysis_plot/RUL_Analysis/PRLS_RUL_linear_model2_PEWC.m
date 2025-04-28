% This program is used to traking the RUL by projected RLS
% The model is choose as linear model (three order)
% y=phi+ax+b*1/(x+gamma)
clc; clear; close all;
addpath("..\Helpler_functions\");
%% load timelist 

PEWC_timelist_name="motor_time_list2_0208to0217PEWC";
file_path = fullfile('../../time_list_extraction/RUL_2/timelist_data/parquet/', PEWC_timelist_name.append('.parquet'));

warning('off','MATLAB:table:ModifiedVarnames');
PEWC_time_list = parquetread(file_path);
warning('on','MATLAB:table:ModifiedVarnames');

%% load and show HI
% time = 1:length(PEWC_time_list.ElapsedTime)';
time = PEWC_time_list.ElapsedTime';

torque_timelist=PEWC_time_list.torque_time_list';

% reference health state 
health_ref= mean(torque_timelist(1:10));

EMA_HI=get_EMA(torque_timelist,0.1,health_ref);
HI_raw=torque_timelist;

figure(); hold on; grid minor;
plot(time, EMA_HI);plot(time, HI_raw);
xlabel('sample');ylabel('HI [p.u.]');


%% RUL setting 
% determine RUL threshold (drop by 20%)
RUL_thres=health_ref*0.8;

% if the data set not reach the RUL, choose the maximum HI for thres 
if max(EMA_HI)<RUL_thres
    RUL_thres=max(EMA_HI);
end 

% find failure index 
Fail_idx_MA=find(EMA_HI<=RUL_thres);
Fail_idx_MA=Fail_idx_MA(1);

% RUL trggering (MK test)
RUL_start_thres=health_ref*0.95;% satrt thres (drop by 2%)
RUL_start_idx=find(EMA_HI<=RUL_start_thres);
RUL_start_idx=RUL_start_idx(1); % index of ruls start (HI)

% initial constants 
phi0=EMA_HI(RUL_start_idx); C1=0; x_P0=0;

% find the strat index 
% tend detection by MK test 
MK_length=50;                           % MK window length 
[S,Z] = rolling_MK(EMA_HI,MK_length,0); % MK test 
MK_thres=3;                             % MK confidence value for triggering 

% two-step RUL triggering 
RUL_start_idx=find(abs(Z)>MK_thres & EMA_HI'<RUL_start_thres);
RUL_start_idx=RUL_start_idx(1);

%% RLS initialization 
% Model setting 
lin_mod = 'phi0+alpha*x+beta/(x+gamma)';

ft = fittype(lin_mod, 'independent', 'x', 'coefficients', {'phi0', 'alpha', 'beta', 'gamma'});

% fitting parameter setting 
phi0_min=-inf;  phi0_max=inf;   phi0_init=health_ref;
alpha_min=-inf; alpha_max=0;    alpha_init=-1;
beta_min=-inf;   beta_max=inf;   beta_init=1;
gamma_min=0;    gamma_max=inf;  gamma_init=1;

% NLR boundary condition 
lower_bounds = [phi0_min, alpha_min, beta_min, gamma_min];
upper_bounds = [phi0_max, alpha_max, beta_max, gamma_max];

fo = fitoptions('Method', 'NonlinearLeastSquares', ...
                  'Lower', lower_bounds, ...
                  'Upper', upper_bounds, ...
                  'StartPoint', [phi0_init, alpha_init, beta_init, gamma_init]);

[phi0, alpha, beta, gamma, NLR_curv, f1] = fitAndPlotLinearModel(lin_mod, ft, fo, time, EMA_HI,...
    RUL_start_idx, MK_length, Fail_idx_MA, RUL_start_thres, Z, MK_thres);
disp(f1);

%% EKF estimation 
Pk=eye(3)*10; %initial covariance 
Qk=diag([0.01;0.01;0.01]);
Rk=0.05;
x0=[EMA_HI(RUL_start_idx);C1*exp(x_P0*MK_length);x_P0];
[X_est] = EKF_2para_exp_model(x0,Pk,EMA_HI(RUL_start_idx:Fail_idx_MA)-phi0,time ,Qk,Rk);

%% RLS tracking 
% Set RUL triggering value
RUL_duration_idx=(RUL_start_idx+1):Fail_idx_MA;
t=time((RUL_start_idx+1):Fail_idx_MA)-time(RUL_start_idx-MK_length);% time steps from RUL trigger point 
N=length(t);
RLS_bond_mult=0.1;
% exponetial RLS tracking (begining by NRstart)


% lin_mod = 'phi0+alpha*x+beta*exp(-(x+gamma))';
%%% very important! t0 of RUL should be zero  
Beta=0.99;
h_exp=[ones(1,N)' t' ((t+gamma).^-1)'];

%fake HI
% EMA_HI=2*exp(0.001*(1:length(EMA_HI)));
P0=5*eye(length(3));% initial covariance matrix
x_initial=[phi0;alpha;beta]; % RLS initial guess by NLR

% RLS boundary 
RLS_bond_L=[0; -3;0];
RLS_bond_U=[log(C1*2); -x_P0*0.1];

% RUL estimation
[X_exp,y_hat_exp,P,MSE_exp]=RLS_functionsContainerPEWC.RLS_covR_bond_3( ...
    EMA_HI(RUL_duration_idx),h_exp,Beta,P0,500000,x_initial,RLS_bond_L, RLS_bond_U);

% plot the estimate parameters 
help_plot_RLS_para(y_hat_exp, EMA_HI, X_exp,RUL_start_idx, h_exp, time);

%% plot the curve fit
plot_HI_fit(time, EMA_HI, NLR_curv, RUL_duration_idx, y_hat_exp, X_est, phi0, Fail_idx_MA, RUL_thres, Beta, RUL_start_idx, MK_length)

%% RUL estimation and animation  
% calculate the absolute time point of estimated RUL
tic;
fail_t_from_0=plotIntersectionAnimation(X_exp, y_hat_exp, MK_length, gamma, RUL_thres, h_exp, EMA_HI, time, RUL_start_idx);
toc

%% show result 

fail_time=time(Fail_idx_MA);

RLS_RUL_boundary=2.5*fail_t_from_0(1)-time(RUL_duration_idx);

RLS_RUL=fail_t_from_0-time(RUL_duration_idx);

RLS_RUL_bound=min(RLS_RUL, RLS_RUL_boundary);

True_RUL=ones(1,length(RUL_duration_idx))*fail_time-time(RUL_duration_idx);

% 建立新圖形，設定白色背景及適當的尺寸
figure('Color', 'w', 'Position', [100, 320, 700, 380]);
hold on;
grid on;

% 取出 RUL 擬合用的時間索引，並轉換為欄向量
t = time(RUL_duration_idx);
t = t(:);

% 將 True_RUL 與 RLS_RUL 轉換為欄向量
True_RUL_col = True_RUL(:);
RLS_RUL_col = RLS_RUL(:);

% 計算 True_RUL 的上下誤差界線：±20%
upper_bound = True_RUL_col * 1.2;
lower_bound = True_RUL_col * 0.8;

% 利用 fill 填入誤差區間，設定淺粉紅色區塊，並調整透明度
fill([t; flipud(t)], [upper_bound; flipud(lower_bound)], [1 0.8 0.8], ...
     'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Error band');

% 計算 SMAPE
SMAPE = mean( abs(RLS_RUL_col - True_RUL_col) ./ ((abs(RLS_RUL_col) + abs(True_RUL_col)) / 2) ) * 100;
SMAPE_bound = mean( abs(RLS_RUL_bound(:) - True_RUL_col) ./ ((abs(RLS_RUL_bound(:)) + abs(True_RUL_col)) / 2) ) * 100;

% 定義 RLS RUL 的圖例字串，將 SMAPE 值納入其中
display_RLS = sprintf('RLS RUL (SMAPE: %.2f%%)', SMAPE);
display_RLS_bound = sprintf('P-RLS RUL (SMAPE: %.2f%%)', SMAPE_bound);

% 繪製 RLS_RUL 曲線，使用柔和藍色實線
plot(t, RLS_RUL_col, 'LineWidth', 1.5, 'LineStyle', '--', 'Color', [0.2, 0.4, 0.8], 'DisplayName', display_RLS);
plot(t, RLS_RUL_bound, 'LineWidth', 1.5, 'Color', [0.2, 0.4, 0.8], 'DisplayName', display_RLS_bound);

% 繪製 True_RUL 曲線，使用柔和紅色實線
plot(t, True_RUL_col, 'LineWidth', 1.5, 'Color', [0.8, 0.2, 0.2], 'DisplayName', 'True RUL');


xlabel('Elapsed Time [min]', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('RUL [min]', 'Interpreter', 'latex', 'FontSize', 14);

% 設定座標軸範圍
axis([-inf inf 0 inf]);

% 加入圖例，使用 LaTeX 格式
lgd = legend('Location', 'best', 'Interpreter', 'latex');
set(lgd, 'FontName', 'Times New Roman', 'FontSize', 12);

%% zoom in 
zoom_idx1=length(true_RUL_MA)-50+1;zoom_idx2=length(true_RUL_MA);
figure();hold on;grid on;
plot(RUL);plot(RUL_EKF);plot(true_RUL_MA);
fill([(1:length(true_RUL_MA)) fliplr(1:length(true_RUL_MA))], [alphaPlus fliplr(alphaMinus)], ...
    'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
axis([max(zoom_idx1,1) zoom_idx2 0 max(true_RUL_MA)]);
title(append(' zoom in RUL estimation of ',PEWC_timelist_name));
legend( 'P-RLS RUL', 'True RUL','EKF RUL','P-RLS boundary', '+-10% confidence interval');

%% functions

% Nonlinear regression 
function [phi0, alpha, beta, gamma, NLR_curv, f1] = fitAndPlotLinearModel(lin_mod, ft, fo, time, EMA_HI, RUL_start_idx, MK_length, Fail_idx_MA, RUL_start_thres, Z, MK_thres)
% fitAndPlotLinearModel 以非線性最小平方法擬合線性模型，並繪製觸發結果圖
%
% 輸入參數：
%   lin_mod        - 模型函式字串，例: 'phi0+alpha*x+beta/(x+gamma)'
%   fo             - fitoptions 物件，用於指定擬合選項
%   startPoints    - 初始參數向量，用於擬合
%   time           - 時間向量
%   EMA_HI         - 與 time 對應的資料向量 (例如 HI)
%   RUL_start_idx  - 開始擬合資料的索引 (用於選取資料區段)
%   MK_length      - 擬合資料點數 (例如取最後 MK_length 個點)
%   Fail_idx_MA    - 失效點索引 (用於圖形標示)
%   RUL_start_thres- RUL 閾值 (水平線)
%   Z              - 第二張圖的資料向量
%   MK_thres       - MK 閾值 (第二張圖中使用)
%
% 輸出參數：
%   phi0, alpha, beta, gamma - 擬合後的模型參數
%
% 模型形式：f(x)=phi0+alpha*x+beta/(x+gamma)

    % 將初始參數存入 fitoptions 中
    % fo.StartPoint = startPoints;

    % 建立 fittype 物件
    % ft = fittype(lin_mod, 'options', fo);
    
    % 擬合資料：選取 time 與 EMA_HI 中符合條件的資料段
    % 此處假設擬合使用 time 的前 MK_length 個點，以及 EMA_HI 中對應區間
    x_fit = (time(RUL_start_idx-MK_length+1:RUL_start_idx)-time(RUL_start_idx-MK_length+1))';
    y_fit = EMA_HI(RUL_start_idx-MK_length+1:RUL_start_idx)';
    
    f1 = fit(x_fit, y_fit, ft, fo);

    % 取得擬合後的參數
    phi0 = f1.phi0;
    alpha = f1.alpha;
    beta  = f1.beta;
    gamma = f1.gamma;
    
    % 繪製觸發結果圖
    figure();
    % 第一個子圖：HI 資料與擬合線
    subplot(2,1,1);
    hold on;
    plot(time, EMA_HI, 'LineWidth',1.5);
    
    % 產生擬合曲線（利用模型公式）
    % 注意：此處用 1:MK_length 當作 x 軸座標，與擬合資料相符
    y_fit_line = phi0 + alpha*(x_fit) + beta./(x_fit+gamma);
    plot(x_fit+time(RUL_start_idx-MK_length+1), y_fit_line, 'LineWidth', 1.5);
    
    % 繪製垂直與水平標線
    xline(time(RUL_start_idx), 'r--');
    xline(time(Fail_idx_MA), 'k--');
    yline(RUL_start_thres, 'r--');
    title('HI');
    grid minor;
    hold off;
    
    % 第二個子圖：Z 資料與 MK 閾值
    subplot(2,1,2);
    hold on;
    plot(time, Z, 'LineWidth',1.5);
    xline(time(RUL_start_idx), 'r--');
    yline(MK_thres, 'r--');
    xline(time(Fail_idx_MA), 'k--');
    title(append('MK value, window length: ', string(MK_length)));
    grid minor;
    hold off;
    
    % retrun the NLR fitted curve
    NLR_curv=[(x_fit+time(RUL_start_idx-MK_length+1)) y_fit_line];
end

% plot RLS parameter traking result 
function help_plot_RLS_para(y_hat, EMA_HI, x_rls,RUL_start_idx, h_input,time)

figure(); 

% fitted curve 

subplot(4,1,1); hold on;
plot(time,EMA_HI); 
plot(time((RUL_start_idx+1):(RUL_start_idx+length(y_hat))),y_hat); 
subtitle('y/hat');

% model parameter1 
subplot(4,1,2);hold on;
plot((x_rls(1,:)));plot((x_rls(1,:)).*h_input(:,1)');
subtitle('phi0');

% model parameter2  
subplot(4,1,3);hold on;
plot((x_rls(2,:)));plot((x_rls(2,:)).*h_input(:,2)');
subtitle('alpha');

% model parameter3  
subplot(4,1,4);hold on;
plot((x_rls(3,:)));plot((x_rls(2,:)).*h_input(:,2)');
subtitle('beta');

end 

% RUL calculate animation 
function [fail_t_from_0]=plotIntersectionAnimation(X_exp, RLS_fit, MK_length, gamma, RUL_thres, h_exp, EMA_HI,time,RUL_start_idx)
% plotIntersectionAnimation 以動畫方式繪製模型函數與 RUL_thres 間的交點
%
% 輸入參數：
%   X_exp      - 3 x N 的矩陣，每列分別代表模型參數 (phi0, alpha, beta)
%   MK_length  - 一個常數，用於模型內的常數項
%   gamma      - 一個常數，用於模型內的常數項
%   RUL_thres  - 模型水平線的值 (RUL 閾值)
%   h_exp      - 與迭代次數相關的向量 (用來決定迭代次數)
%
% 此函式在每次迭代中，根據目前的參數建立模型函數
%   f(x) = phi0 + alpha*x + alpha*MK_length + beta/(x+MK_length+gamma)
% 並求解 f(x)==RUL_thres 的交點，選取 x>0 的最小值作為交點，
% 然後以動畫方式繪製模型曲線、RUL_thres 水平線以及交點。

    % 設定 x 軸繪圖範圍
    x_min = 0;
    x_max = 12000;  % xlimit
    x_vals = linspace(x_min, x_max, 1000);
    
    % save the RUL estimation result 
    fail_t_from_0=zeros(1,length(h_exp));

    figure;
    for i_RLS = 1:length(h_exp)
        % 定義符號變數
        syms x

        % 根據當前迭代的參數建立模型函數
        % 注意：此處使用 X_exp(2,i_RLS) 出現兩次，請確認是否符合您的模型需求
        time_offset=time(RUL_start_idx)-time(RUL_start_idx);
        f_sym = X_exp(1,i_RLS) + X_exp(2,i_RLS)*x + X_exp(3,i_RLS)./(x+gamma);

        % 建立方程式 f(x) = RUL_thres
        eqn = f_sym == RUL_thres;

        % 求解方程式
        sol = vpasolve(eqn, x);
        sol_values = double(sol);
        % 選取大於 0 的解
        sol_positive = sol_values(sol_values > 0);
        if isempty(sol_positive)
            intersection_x = NaN;
        else
            intersection_x = min(sol_positive);
        end

        % 將符號函數轉換為匿名函數以便繪圖
        f_handle = matlabFunction(f_sym, 'Vars', x);
        y_vals = f_handle(x_vals);

        % 繪圖：模型函數、RUL_thres 水平線
        plot(x_vals+time(RUL_start_idx-MK_length+1), y_vals, 'b-', 'LineWidth', 2);
        hold on;
        plot(x_vals, RUL_thres*ones(size(x_vals)), 'r--', 'LineWidth', 1.5);

        % 標示交點（如果存在）
        if ~isnan(intersection_x)
            intersection_y = f_handle(intersection_x);
            % shift to the absolute time point 
            intersection_x=intersection_x+time(RUL_start_idx-MK_length+1);
            plot(intersection_x, intersection_y, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
            % 顯示交點座標
            text(intersection_x, intersection_y, sprintf(' (%.2f, %.2f)', intersection_x, intersection_y), ...
                'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
        end
        
        % update the absolute RUL time 
        fail_t_from_0(i_RLS)=intersection_x;
        
        % plot RLs fit 
        plot(time(RUL_start_idx+1:RUL_start_idx+i_RLS),RLS_fit(1:i_RLS),'r--');
        plot(time(RUL_start_idx+i_RLS), RLS_fit(i_RLS), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
         
        % plot the HI 
        plot(time(1:RUL_start_idx+i_RLS), EMA_HI(1:+RUL_start_idx+i_RLS));

        hold off;
        xlabel('x');
        ylabel('f(x)');
        title(sprintf('Iteration %d', i_RLS));
        legend('模型函數', 'RUL\_thres', '交點', 'Location', 'best');
        grid on;

        axis([0 time(end) RUL_thres max(EMA_HI)])

        % 暫停一段時間以呈現動畫效果
        pause(0.025);
    end
end

% plot the curve fitting result 
function plot_HI_fit(time, EMA_HI, NLR_curv, RUL_duration_idx, y_hat_exp, X_est, phi0, Fail_idx_MA, RUL_thres, Beta, RUL_start_idx, MK_length)
% plotHI - 繪製 HI 相關曲線圖
%
% 說明:
%   此函式依照預先設定的格式，繪製原始 HI 曲線、非線性迴歸初始結果、RLS 擬合結果、
%   以及 EKF 狀態估計結果，並在圖中標示故障位置與 RUL 門檻。
%
% 輸入參數:
%   time           - 時間向量或樣本索引
%   EMA_HI         - 原始 HI 資料
%   NLR_curv       - 非線性迴歸曲線資料，必須為兩欄矩陣，第一欄為 x 軸資料，第二欄為 y 軸資料
%   RUL_duration_idx - RLS 擬合與 EKF 狀態估計的索引
%   y_hat_exp      - RLS 擬合的 HI 結果向量
%   X_est          - EKF 狀態估計結果，第一欄為估計的 HI 狀態（注意：此向量長度需比 y_hat_exp 多一點）
%   phi0           - EKF 模型中的常數項
%   Fail_idx_MA    - 故障位置對應的索引
%   RUL_thres      - RUL 門檻值
%   Beta           - RLS 擬合參數，用於圖例顯示
%   RUL_start_idx  - RUL 起始索引，用於設定軸範圍（選用）
%   MK_length      - 用於設定軸範圍的參數（選用）
%
% 範例:
%   plotHI(time, EMA_HI, NLR_curv, RUL_duration_idx, y_hat_exp, X_est, phi0, Fail_idx_MA, RUL_thres, Beta, RUL_start_idx, MK_length);

    % 建立新圖形，設定白色背景及適當的尺寸
    f = figure('Color', 'w', 'Position', [100, 320, 700, 380]);
    hold on;


    % 繪製非線性迴歸初始結果（柔和紅色實線）
    plot(NLR_curv(:,1), NLR_curv(:,2), 'LineWidth', 1.5, 'Color', [0.8, 0.2, 0.2], 'DisplayName', 'NLR initial');

    % RLS 擬合處理：
    % 將 y_hat_exp 的第一個值調整成 NLR_curv 的最後一個點以連接兩條曲線
    y_hat_exp(1) = NLR_curv(end);

    % 繪製 RLS 擬合結果（黑色虛線）
    plot(time(RUL_duration_idx), y_hat_exp, 'LineWidth', 1, 'LineStyle', '-', 'Color', 'k', ...
     'Marker', 'o', 'MarkerIndices', 1:5:length(time(RUL_duration_idx)), ...
     'DisplayName', append('RLS fitted ', '\beta=', string(Beta)));

    % 繪製 EKF 狀態估計結果（柔和紅色虛線）
    plot(time(RUL_duration_idx), X_est(1:end-1,1) + phi0, 'LineWidth', 1, 'LineStyle', '-', 'Color', [0.8, 0.2, 0.2], ...
     'Marker', 'o', 'MarkerIndices', 1:5:length(time(RUL_duration_idx)), ...
     'DisplayName', append('RLS fitted ', '\beta=', string(Beta)));

    % 繪製原始 HI 曲線（柔和藍色實線）
    plot(time, EMA_HI, 'LineWidth', 2, 'Color', [0.2, 0.4, 0.8], 'DisplayName', 'HI');

    % 在故障位置處加入垂直虛線
    xline(time(Fail_idx_MA), 'LineWidth', 1.5, 'LineStyle', '--', 'Color', 'k');
    % 取得目前 y 軸範圍
    yl = ylim;

    % 在垂直線旁加入文字標記，將文字放在圖的上方偏右位置
    text(time(Fail_idx_MA), yl(2) - 0.05*(yl(2)-yl(1)), 'Failure time ', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
        'Interpreter', 'latex');

    % 在 RUL 門檻處加入水平虛線
    yline(RUL_thres, 'LineWidth', 1.5, 'LineStyle', '--', 'Color', 'k');
    % 取得目前 x 軸範圍
    xl = xlim;
    % 在水平線旁加入文字標記，將文字放在圖的左側偏上位置
    text(xl(1) + 0.05*(xl(2)-xl(1)), RUL_thres, 'Failure threshold', ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
        'Interpreter', 'latex');


    % 設定軸標籤，使用 LaTeX 格式
    xlabel('Elapsed Time [min]', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('Torque [Nm]', 'Interpreter', 'latex', 'FontSize', 14);

    % 啟用格線與邊框，並調整座標軸屬性
    grid on;
    box on;
    set(gca, 'FontSize', 12, 'LineWidth', 1, 'TickLabelInterpreter', 'latex');

    % 設定圖例，並使用 LaTeX 格式
   legend({'True Toruque', 'NLR region', sprintf('$RLS fitted  (\\beta=%.2f$)', Beta), 'EKF fitted'},...
       'Location', 'best', 'Interpreter', 'latex');

    % 若需要設定軸範圍，可取消下列註解
    % axis([RUL_start_idx-MK_length+1, Fail_idx_MA, -inf, inf]);
end
