%% 範例程式：利用 ARIMA 預測時序資料下降到特定值的時間

% 清除環境變數
clear; clc; close all;

%% 1. 產生模擬資料
rng(1);  % 設定隨機數種子，確保重現性
T = 100;              % 資料點數
t = (1:T)';           % 時間軸
% 模擬一個具有下降趨勢且有隨機擾動的時序資料
y = 50 - 0.3*t + randn(T,1);

%% 2. ARIMA 模型建構與估計
% 選擇 ARIMA(1,1,1) 模型 (常數項設為 0)
Mdl = arima('Constant',0,'D',1,'AR',0.5,'MA',0.3);
% 使用資料估計模型參數
EstMdl = estimate(Mdl, y);

%% 3. 利用模型進行預測
numForecastSteps = 50;   % 預測未來 50 個時間點
[YF, YMSE] = forecast(EstMdl, numForecastSteps, 'Y0', y);

%% 4. 判斷預測何時下降到特定門檻
target = 30;  % 設定目標門檻值
belowTargetIdx = find(YF < target, 1, 'first');
if isempty(belowTargetIdx)
    fprintf('在未來 %d 個時間點內，預測資料均未下降到 %.2f 以下。\n', numForecastSteps, target);
else
    % 預測下降到門檻的時刻 (以資料原有時間點編號計算)
    forecastTime = T + belowTargetIdx;
    fprintf('預測時序資料在時間點 %d 下降到 %.2f 以下。\n', forecastTime, target);
end

%% 5. 繪圖：觀察原始資料與預測結果
figure;
hold on;
% 繪製原始時序資料
plot(t, y, 'b-o', 'DisplayName', 'Observed');
% 繪製預測資料
tForecast = (T+1):(T+numForecastSteps);
plot(tForecast, YF, 'r-*', 'DisplayName', 'Forecast');
% 繪製預測與原始資料間的連線（如果有下降到門檻）
if ~isempty(belowTargetIdx)
    plot([T, T+belowTargetIdx], [y(end), YF(belowTargetIdx)], 'k--', 'LineWidth',2, ...
         'DisplayName', 'Transition');
end
% 繪製目標門檻線
yline(target, 'k:', 'LineWidth', 1.5, 'DisplayName', 'Threshold');

xlabel('Time');
ylabel('Value');
title('ARIMA 預測及下降到特定門檻值的時間');
legend('Location', 'best');
grid on;
hold off;
