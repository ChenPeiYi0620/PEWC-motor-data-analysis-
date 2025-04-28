import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from datetime import datetime
import pandas as pd
import Data_handle_in_IPC
import os

# 設定當前工作目錄為腳本所在的目錄
# 這樣可以確保相對路徑正確
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
record_info = '0310to0318PEWC'
essemble_data_file = f"../PEWC dataset/PEWC_essembled_data/RUL_{device_number}/{record_info}_essemble.h5"
essemble_data = pd.read_hdf(essemble_data_file)


output_file_parquet = os.path.join(f"time_list_extraction\RUL_{device_number}", 
                                   "timelist_data", "parquet", f"motor_time_list{device_number}_{record_info}.parquet")
motor_time_list = pd.read_parquet(output_file_parquet).to_dict(orient='list')
torque_timelist=motor_time_list["torque_time_list"]

# parameters
num_samples = len(essemble_data)
Fs = 10000  # 取樣頻率 (Hz)
torque_min = 20
torque_max = 35

# 建立圖形和四個子圖（垂直排列）
fig, axs = plt.subplots(4, 1, figsize=(12, 16))
plt.subplots_adjust(bottom=0.25, hspace=0.4)

# 定義一個函式用來計算時間軸（單位：秒），給定信號長度
def get_time_axis(length):
    return np.arange(0, length) / Fs

# 初始樣本索引
initial_index = 0
sample = essemble_data.iloc[initial_index]

# 子圖1：扭矩歷程資料 torque_time_list
time_axis_torque = np.arange(1, len(torque_timelist) + 1)
line_torque, = axs[0].plot(time_axis_torque, torque_timelist, lw=2, label="Torque")
# 標記當前扭矩點：假設取最後一個點
marker_torque, = axs[0].plot(time_axis_torque[initial_index], torque_timelist[initial_index], 'ro', label="Current Torque")
axs[0].set_xlabel("編號")
axs[0].set_ylabel("扭矩")
axs[0].set_title(f"扭矩歷程 (樣本編號: {initial_index})")
axs[0].legend()
axs[0].set_ylim(torque_min, torque_max )  # Add small padding

# 子圖2：兩軸電壓資料
time_axis_voltage = get_time_axis(len(sample['Voltage alpha']))
line_voltage_alpha, = axs[1].plot(time_axis_voltage, sample['Voltage alpha'], lw=2, label="Voltage Alpha")
line_voltage_beta, = axs[1].plot(time_axis_voltage, sample['Voltage beta'], lw=2, label="Voltage Beta")

# 顯適用於計算力矩的有效範圍
line_voltage_alpha_vlaid, = axs[1].plot(time_axis_voltage[2250:2500], sample['Voltage alpha'][2250:2500], 'r-', lw=2, label="Voltage Alpha")
line_voltage_beta_vlaid, = axs[1].plot(time_axis_voltage[2250:2500], sample['Voltage beta'][2250:2500], 'r-', lw=2, label="Voltage Beta")
axs[1].set_xlabel("時間 (秒)")
axs[1].set_ylabel("電壓")
axs[1].set_title(f"兩軸電壓 (樣本編號: {initial_index})")
axs[1].legend()

# 子圖3：兩軸電流資料
time_axis_current = get_time_axis(len(sample['Current alpha']))
line_current_alpha, = axs[2].plot(time_axis_current, sample['Current alpha'], lw=2, label="Current Alpha")
line_current_beta, = axs[2].plot(time_axis_current, sample['Current beta'], lw=2, label="Current Beta")
# 顯適用於計算力矩的有效範圍
line_current_alpha_vlaid, = axs[2].plot(time_axis_current[2250:2500], sample['Current alpha'][2250:2500], 'r-', lw=2, label="Current Alpha")
line_current_beta_vlaid, = axs[2].plot(time_axis_current[2250:2500], sample['Current beta'][2250:2500], 'r-', lw=2, label="Current Beta")
axs[2].set_xlabel("時間 (秒)")
axs[2].set_ylabel("電流")
axs[2].set_title(f"兩軸電流 (樣本編號: {initial_index})")
axs[2].legend()

# 子圖4：計算出的訊號 temp_torque
time_axis_temp = time_axis_current
temp_torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(sample, debug=False)
line_temp_torque, = axs[3].plot(time_axis_temp, temp_torque, lw=2, label="Temp Torque")# 顯適用於計算力矩的有效範圍
line_temp_torque_vlaid, = axs[3].plot(time_axis_temp[2250:2500], temp_torque[2250:2500], 'r-', lw=2, label="Current Alpha")
axs[3].set_xlabel("時間 (秒)")
axs[3].set_ylabel("Temp Torque")
axs[3].set_title(f"計算訊號 (temp_torque) (樣本編號: {initial_index})")
axs[3].legend()

# 在子圖1中加入文字區塊，顯示該筆資料的收集時間
time_text = axs[0].text(0.05, 0.95, '', transform=axs[0].transAxes,
                        fontsize=10, verticalalignment='top')

def update_time_text(index):
    sample_unix_time = essemble_data.iloc[index]['Unix Time']
    dt_str = datetime.fromtimestamp(int(sample_unix_time)).strftime('%Y-%m-%d %H:%M:%S')
    time_text.set_text(f"樣本 {index} 收集時間: {dt_str}")

update_time_text(initial_index)

# 建立滑桿，範圍為 0 到 num_samples - 1 (整數步進)
ax_slider = plt.axes([0.25, 0.05, 0.65, 0.015])
slider = Slider(ax_slider, '樣本編號', 0, num_samples - 1, valinit=initial_index, valstep=1)

def update(val):
    index = int(slider.val)
    sample = essemble_data.iloc[index]
    
    # 更新子圖1：扭矩歷程
    time_axis_torque = np.arange(1, len(torque_timelist) + 1)
    # 用列表包裝數值，確保 set_data 接受序列
    marker_torque.set_data([time_axis_torque[index]], [torque_timelist[index]])
    axs[0].set_xlim(time_axis_torque[0], time_axis_torque[-1])
    axs[0].relim()
    axs[0].set_ylim(torque_min, torque_max )  # Add small padding
    
    # 更新子圖2：兩軸電壓
    time_axis_voltage = get_time_axis(len(sample['Voltage alpha']))
    line_voltage_alpha.set_data(time_axis_voltage, sample['Voltage alpha'])
    line_voltage_beta.set_data(time_axis_voltage, sample['Voltage beta'])
    # 顯適用於計算力矩的有效範圍
    line_voltage_alpha_vlaid.set_data(time_axis_voltage[2250:2500], sample['Voltage alpha'][2250:2500])
    line_voltage_beta_vlaid.set_data(time_axis_voltage[2250:2500], sample['Voltage beta'][2250:2500])
    axs[1].set_xlim(time_axis_voltage[0], time_axis_voltage[-1])
    axs[1].relim()
    axs[1].autoscale_view()
    
    # 更新子圖3：兩軸電流
    time_axis_current = get_time_axis(len(sample['Current alpha']))
    line_current_alpha.set_data(time_axis_current, sample['Current alpha'])
    line_current_beta.set_data(time_axis_current, sample['Current beta'])
     # 顯適用於計算力矩的有效範圍
    line_current_alpha_vlaid.set_data(time_axis_current[2250:2500], sample['Current alpha'][2250:2500])
    line_current_beta_vlaid.set_data(time_axis_current[2250:2500], sample['Current beta'][2250:2500])
    axs[2].set_xlim(time_axis_current[0], time_axis_current[-1])
    axs[2].relim()
    axs[2].autoscale_view()
    
    # 更新子圖4：計算訊號 temp_torque
    # 正確 unpack 估算結果（根據原先使用方式）

    # is_after_march_10 = int(sample["Unix Time"]) >= datetime(2025, 3, 10).date()
    # if file_num > 1559 or is_after_march_10:
    #         # turn the voltage beta into phase c voltage 
    #         data_read["Voltage beta"] = data_read["Voltage alpha"] / 2 - data_read["Voltage beta"] * np.sqrt(3) / 2 
    #         torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(data_read, debug=False)
    #     else:
    #         torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(data_read, debug=False)
    
    temp_torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(sample, debug=False)
    # 根據 temp_torque 的長度重新建立時間軸
    time_axis_temp = get_time_axis(len(temp_torque))
    line_temp_torque.set_data(time_axis_temp, temp_torque)
    line_temp_torque_vlaid.set_data(time_axis_temp[2250:2500], temp_torque[2250:2500])
    axs[3].set_xlim(time_axis_temp[0], time_axis_temp[-1])
    axs[3].relim()
    axs[3].autoscale_view()
    
    # 更新各子圖標題
    axs[0].set_title(f"扭矩歷程 (樣本編號: {index}, 扭矩: {torque_timelist[index]})")
    axs[1].set_title(f"兩軸電壓 (樣本編號: {index})")
    axs[2].set_title(f"兩軸電流 (樣本編號: {index})")
    axs[3].set_title(f"計算訊號 (temp_torque) (樣本編號: {index})")
    
    # 更新收集時間文字
    update_time_text(index)
    
    fig.canvas.draw_idle()

slider.on_changed(update)
manager = plt.get_current_fig_manager()
if hasattr(manager.window, 'showMaximized'):
    manager.window.showMaximized()
else:
    try:
        manager.window.state('zoomed')
    except Exception as e:
        print("無法自動最大化視窗，使用預設大小：", e)
plt.show()
