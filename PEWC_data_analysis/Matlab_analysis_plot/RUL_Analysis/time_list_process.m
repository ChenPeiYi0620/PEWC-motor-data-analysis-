clc; clear; close all;
% warning('off','MATLAB:table:ModifiedVarnames');
% dir='../../time_list_extraction/motor_time_list2_2.parquet'
% PEWC_data_0208_0217_2_2 = parquetread('../../time_list_extraction/RUL_2/motor_time_list2_2.parquet');
% PEWC_data_0208_0217_5_2 = parquetread('../../time_list_extraction/motor_time_list5_2.parquet');
% warning('on','MATLAB:table:ModifiedVarnames');

%% RUL predition from time data 

% 設定目錄路徑
folderPath = '../../time_list_extraction/RUL_5/';
filePattern = fullfile(folderPath, '*.csv');

csvFiles= dir(filePattern);

first_data=readtable(fullfile(folderPath, csvFiles(1).name));
initial_time=first_data.TimeStamps(1);

time_list_summary=first_data.TimeStamps-initial_time;
torque_summary=first_data.torque_time_list;

% 遍歷後續每個檔案
for k = 2:length(csvFiles)
    % 完整檔案路徑
    baseFileName = csvFiles(k).name;
    fullFileName = fullfile(folderPath, baseFileName);

    % 顯示檔名
    fprintf('正在處理檔案: %s\n', fullFileName);

    % 讀取 CSV（依資料內容調整）
    data = readtable(fullFileName);
    time_list_summary=[time_list_summary; data.TimeStamps-initial_time];
    torque_summary=[torque_summary; data.torque_time_list];

end

%%

figure(); 
plot(time_list_summary/60, torque_summary);
ylabel('Nm'); xlabel('time elapsed[min]');
title('Motor 4 torque over time');
grid on; 

%%











%% plot time list 

% 建立新圖形，設定白色背景及適當的尺寸
f = figure('Color', 'w', 'Position', [100, 320, 700, 380]);
hold on;
time1=PEWC_data_0208_0217_2_2.ElapsedTime;
time2=PEWC_data_0208_0217_5_2.ElapsedTime;

% 繪製第一條曲線：柔和藍色實線
plot(time1, PEWC_data_0208_0217_2_2.torque_time_list, ...
    'LineWidth', 1.5, 'Color', [0.2, 0.4, 0.8], 'DisplayName', 'Data 2\_2');

% 繪製第二條曲線：柔和紅色虛線
plot(time2, PEWC_data_0208_0217_5_2.torque_time_list, ...
    'LineWidth', 1.5, 'LineStyle', '-', 'Color', [0.8, 0.2, 0.2], 'DisplayName', 'Data 5\_2');

% 設定軸標籤，使用 LaTeX 格式
xlabel('Elapsed Time [min]', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Torque [Nm]', 'Interpreter', 'latex', 'FontSize', 14);

% 設定圖例，並使用 LaTeX 格式
legend({'M2 torque output', 'M4 torque output'}, 'Location', 'best', 'Interpreter', 'latex');

% 啟用格線與邊框，調整座標軸字型及線寬
grid on;
box on;
set(gca, 'FontSize', 12, 'LineWidth', 1, 'TickLabelInterpreter', 'latex');

% 將 Unix 時間轉換成 datetime，並指定格式（例如 'yyyy-MM-dd HH:mm:ss'）
startDate1 = datetime(PEWC_data_0208_0217_2_2.TimeStamps(1), 'ConvertFrom', 'posixtime');
endDate1   = datetime(PEWC_data_0208_0217_2_2.TimeStamps(end), 'ConvertFrom', 'posixtime');

% 轉換為字串
startStr1 = datestr(startDate1, 'yyyy-mm-dd');
endStr1   = datestr(endDate1, 'yyyy-mm-dd ');

% 繪圖後取得預設的 xtick 位置與標籤
default_xticks = get(gca, 'XTick');
default_xticklabels = get(gca, 'XTickLabel');
% default_xticklabels{1}=[default_xticklabels{1},': ',startStr1];
% default_xticklabels{end}=[default_xticklabels{end},': ',endStr1];

default_xticklabels{1}=[startStr1];
default_xticklabels{end}=[endStr1];

% 使用 xtick 與 xticklabel 在 x 軸的頭尾位置標示日期
% 此處假設 x 軸的資料單位為 minutes，並以資料的第一與最後一筆數值作為 xtick 位置
ax = gca;
ax.TickLabelInterpreter = 'tex';
xticks(default_xticks);
xticklabels(default_xticklabels);
axis('tight')
