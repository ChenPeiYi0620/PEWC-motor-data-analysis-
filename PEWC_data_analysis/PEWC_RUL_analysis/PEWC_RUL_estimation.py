'''This script is used to estimate the RUL result of PEWC dataset.'''
'''The data is use the PEWC dataset of '''

# pip tools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from datetime import datetime
import pandas as pd
import os
import sys 

# 設定當前工作目錄為腳本所在的目錄
# 這樣可以確保相對路徑正確
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 將專案根目錄加入 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 現在就可以正常 import
from PEWC_analysis_helpler.rul_helplers import time_data_preprocess as t_process



def get_datetime_from_unix(unix_time):
    # 將 Unix 時間戳轉換為可讀的日期時間格式
    return pd.to_datetime(unix_time, unit='s').strftime('%Y-%m-%d')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

""" 0208to0217PEWC
    0217to0226PEWC
    0226to0310PEWC
    0310to0318PEWC
    0318to0325PEWC""" 
    
# 讀取資料檔案
device_number = 2
record_info = '0208to0217PEWC'

output_file_parquet = os.path.join("..", "time_list_extraction", f"RUL_{device_number}",
                                   "timelist_data", "parquet", f"motor_time_list{device_number}_{record_info}.parquet")
motor_time_list = pd.read_parquet(output_file_parquet).to_dict(orient='list')

# Get first and last data 

start_date= get_datetime_from_unix(motor_time_list["Time stamps"][0])
end_date= get_datetime_from_unix(motor_time_list["Time stamps"][-1])
print("start date: ", start_date)
print("end date: ", end_date)

#%% test plot region 

torque_timelist = np.array(motor_time_list["torque_time_list"])
time =np.array(motor_time_list["Elapsed time"])


# Use moving window method to detect outliers
filtered_data, kept_idx, filtered_idx=t_process.remove_outliers_moving_window(torque_timelist, window_size=10, n_std=2.5, plot=False)
filtered_time=time[kept_idx]

ema_data=t_process.ewma(filtered_data, alpha=0.1)
health_indicator=ema_data



# %% plot sections 

#  plot outliers 
plt.figure(figsize=(12, 8))
plt.plot(time, torque_timelist, label='Original Data', alpha=0.5)
plt.plot(time[filtered_idx], torque_timelist[filtered_idx], 'o', label='Outlier Data', color='red', markersize=4)
plt.plot(filtered_time, filtered_data, label='Filtered Data', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Elapsed time [min]', fontsize=14)
plt.ylabel('Torque [Nm]', fontsize=14)
plt.tick_params(axis='both', labelsize=14)  # Sets font size for tick labels on both axes
plt.title(f'Estimated torque of motor2 versus Time \n {start_date} to {end_date}', fontsize=14)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show(block=False)


#  plot ema data  
plt.figure(figsize=(12, 8))
plt.plot(filtered_time, filtered_data, label='Filtered Data', alpha=0.5)
plt.plot(filtered_time, ema_data, label='EWMA Data', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Elapsed time [min]', fontsize=14)
plt.ylabel('Torque [Nm]', fontsize=14)
plt.tick_params(axis='both', labelsize=14)  # Sets font size for tick labels on both axes
plt.title(f'EWMA torque of motor2 versus Time \n {start_date} to {end_date}', fontsize=14)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show(block=False)



#%% 

