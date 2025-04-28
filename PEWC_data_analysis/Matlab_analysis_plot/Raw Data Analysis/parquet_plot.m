clc; clear; 
warning('off','MATLAB:table:ModifiedVarnames');
% PEWC_data_0208_0217_2_2 = parquetread('RUL_Data_5_30.parquet');
% PEWC_data_0208_0217_2_2 = parquetread('../test_data/RUL_Data_2_315.parquet');
data = parquetread('../test_data/RUL_Data_2_2752.parquet');
warning('on','MATLAB:table:ModifiedVarnames');

figure(); hold on;
% subplot(4,1,1)
plot(data.CurrentAlpha{1})

%% FFt analysis 
current_alpha=data.CurrentAlpha{1};
current_beta=data.CurrentBeta{1};



