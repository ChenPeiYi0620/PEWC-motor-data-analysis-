% This script is used to reralize the proposed nameplate IM estimation
% method ofor Flux, Torque and Speed 
% Paper name is "Measurement Analysis and Efficiency Estimation of Three Phase Induction Machines Using Instantaneous Electrical Quantities"
clc; close all; clear; 
%% import and plot  sample data and initialize parameters 
% smaple_file='../test_data/RUL_Data_2_2752.parquet';
% smaple_file='../test_data/RUL_Data_5_2754.parquet';
smaple_file='../test_data/RUL_Data_2_4831.parquet';
% smaple_file='../test_data/RUL_Data_2_4837.parquet';
% smaple_file='../test_data/RUL_Data_2_4838.parquet';


sample_data= parquetread(smaple_file);

% sampling rate 
Fs=10000; 
% name plate information  
Motor_params = struct(...
    'Power_rated', 15000, ...
    'P', 2, ...
    'fs', 60, ...
    'V_rated', 220, ...
    'I_rated', 47.7, ...
    'Efficiency', 91.7 ...
);
Rs=0.1

% plot the raw data 
figure(); title('\alpha \beta  Volateg ');
hold on;
plot(sample_data.VoltageAlpha{1});
plot(sample_data.VoltageBeta{1});

figure(); title('\alpha \beta  Curretn ');
hold on;
plot(sample_data.CurrentAlpha{1});
plot(sample_data.CurrentBeta{1});


%% FAST estimation 

%% rotation speed estimation 

% === DFLL 轉速估測器 ===
% 假設輸入：ia, ib 是 alpha/beta 軸電流，f_sample 為取樣率
ia=sample_data.CurrentAlpha{1}; 
ib=sample_data.CurrentBeta{1};
% 範例參數
f_sample = 10000;      % 取樣頻率 [Hz]
N=length(sample_data.CurrentBeta{1});
t = (0:1/f_sample:1-1/f_sample);

% 假設電流信號（若已存在 ia、ib 請忽略）
% ia = ...  % alpha 軸電流
% ib = ...  % beta 軸電流

% 定義目標頻率範圍（側頻搜尋）
f_target = 57:0.001:61;   % 可調整解析度與範圍
window = hann(length(t)); % 使用窗函數降噪

% 建立掃描器
Amp = zeros(size(f_target));
for k = 1:length(f_target)
    freq = f_target(k);
    w = 2*pi*freq;
    ref = exp(1i*w*t') .* window; % 複數正弦掃描
    sig = (ia + 1i*ib).*window;  % 複數表示法（可用 dq0 結果）
    Amp(k) = abs(sum(sig .* conj(ref))); % 內積作為頻率能量估測
end

% 找出最大對應頻率
[~, idx] = max(Amp);
f_peak = f_target(idx);

% 轉速估算（以同步轉速 fs = 60Hz, 極數 = 4 為例）
fs = 60;  % 主頻
p = 2;
slip = abs(f_peak - fs)/fs;
Ns = 120*fs/p;
Nr = (1 - slip)*Ns;

fprintf("推估轉子頻率：%.4f Hz\n", f_peak);
fprintf("推估 slip：%.4f\n", slip);
fprintf("推估轉速：%.2f RPM\n", Nr);


%% power factor analysis 
% 模擬 α-β 軸電壓與電流訊號 (請以你的資料取代)
% 假設電壓與電流均為正弦波，且電流相對電壓延遲30度
v_alpha = sample_data.VoltageAlpha{1};
v_beta  = sample_data.VoltageBeta{1};
i_alpha = sample_data.CurrentAlpha{1};
i_beta  = sample_data.CurrentBeta{1};

f0=Motor_params.fs;

get_signal_phase_delay(v_alpha, v_beta, Fs, f0);
get_stator_power_factor(v_alpha, v_beta, i_alpha, i_beta);

%% functions 
% double channel 
function get_stator_power_factor(v_alpha, v_beta, i_alpha, i_beta)
% 計算每個時間點的電壓與電流相位角
theta_v_all = atan2(v_beta, v_alpha);
theta_i_all = atan2(i_beta, i_alpha);

% 相位差（每個時間點）
phase_diff_all = theta_v_all - theta_i_all;

% 將相位差包進 [-pi, pi] 區間
phase_diff_all = mod(phase_diff_all + pi, 2*pi) - pi;

% 計算平均相位差
mean_phase_diff = mean(phase_diff_all);

% 計算功率因數（以平均相位差為基礎）
PF_phase_avg = cos(mean_phase_diff);

fprintf('利用平均相位差法計算:\n');
fprintf('平均相位差：%.2f 度\n', rad2deg(mean_phase_diff));
fprintf('功率因數：%.4f\n', PF_phase_avg);
end 

function get_signal_phase_delay(signal1, signal2, Fs, f0)
% 模擬兩個訊號
% signal1 為參考訊號，signal2 延遲 30 度
% 方法一：利用 FFT 分析基波相位
N = length(signal1);
FFT1 = fft(signal1);
FFT2 = fft(signal2);
freqAxis = (0:N-1) * Fs / N;
t=(0:N-1)/Fs;

% 找出與基波頻率 f0 最接近的頻率索引
[~, idx] = min(abs(freqAxis - f0));

% 取得基波相位 (弧度)
phase1_fft = angle(FFT1(idx));
phase2_fft = angle(FFT2(idx));
phase_diff_fft = phase1_fft - phase2_fft;
phase_diff_fft_deg = rad2deg(phase_diff_fft);

% 方法二：利用 Hilbert 轉換計算瞬時相位
phase1_inst = angle(hilbert(signal1));
phase2_inst = angle(hilbert(signal2));
phase_diff_inst = unwrap(phase1_inst - phase2_inst);  % unwrap 處理跳變

% 可視化
figure;

% (1) 時域訊號
subplot(3,1,1);
plot(t, signal1, 'b', t, signal2, 'r');
xlabel('時間 (秒)');
ylabel('振幅');
title('時域訊號');
legend('Signal 1', 'Signal 2');

% (2) 基波相位 (FFT 方法)
subplot(3,1,2);
bar([rad2deg(phase1_fft), rad2deg(phase2_fft)]);
set(gca, 'XTickLabel', {'Signal 1', 'Signal 2'});
ylabel('相位 (度)');
title(['FFT 取得基波相位: Phase Difference = ' num2str(phase_diff_fft_deg, '%.2f') '°']);

% (3) 瞬時相位與相位差 (Hilbert 方法)
subplot(3,1,3);
plot(t, unwrap(phase1_inst), 'b', t, unwrap(phase2_inst), 'r', t, phase_diff_inst, 'k--', 'LineWidth',1.5);
xlabel('時間 (秒)');
ylabel('相位 (弧度)');
title('Hilbert 轉換瞬時相位及相位差');
legend('Phase of Signal 1', 'Phase of Signal 2', 'Phase Difference');

% 輸出計算結果
fprintf('FFT 方法計算的基波相位差：%.2f 度\n', phase_diff_fft_deg);
fprintf('Hilbert 方法計算的瞬時相位差 (平均值)：%.2f 度\n', rad2deg(mean(phase_diff_inst)));

end 
