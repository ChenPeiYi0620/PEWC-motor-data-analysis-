% This script is used to reralize the proposed nameplate IM estimation
% method ofor Flux, Torque and Speed 
% Paper name is "Measurement Analysis and Efficiency Estimation of Three Phase Induction Machines Using Instantaneous Electrical Quantities"
clc; close all; clear; 
%% import and plot  sample data and initialize parameters 
% smaple_file='../test_data/RUL_Data_2_2752.parquet';
smaple_file='../test_data/RUL_Data_5_2754.parquet';
% smaple_file='../test_data/RUL_Data_2_4831.parquet';
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


%% FFT
current_alpha=sample_data.CurrentAlpha{1};
current_beta=sample_data.CurrentBeta{1};

% 假設 current_alpha, current_beta 已定義，Fs 為採樣率
current_alpha_beta = current_alpha + 1i * current_beta;
N = length(current_alpha_beta);  % 取樣點數

% 雙邊 FFT（未做移位）
Y = fft(current_alpha_beta);

% 頻率軸（雙邊）
f = (-N/2:N/2-1) * (Fs/N);  % 雙邊頻率軸（含負頻率）

% 雙邊頻譜圖（中心對齊）
Y_shifted = fftshift(Y);            % 頻譜移位
magnitude_spectrum = abs(Y_shifted)/N;  % 幅度

% 畫圖
figure;
plot(f, magnitude_spectrum);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Double-sided FFT Spectrum of current\_alpha\_beta');
grid on;

