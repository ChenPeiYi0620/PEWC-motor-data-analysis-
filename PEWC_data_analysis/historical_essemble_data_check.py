# This profgram is to extract the time list data from essembled data import os

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import rul_data_read
import rul_data_read as rul_rd
import Data_handle_in_IPC
import time
import os
import Analysis_helpler


# 設定當前工作目錄為腳本所在的目錄
# 這樣可以確保相對路徑正確
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def parse_timestamp(timestamp):
    """將 Unix 時間轉換為 datetime 物件"""
    return datetime.fromtimestamp(timestamp)
# plot time list result (dict version)

def plot_basic_time_list_df(basic_extract_result: pd.DataFrame, keys: list, plot_file=""):
    import math
    """
    根據 basic_extract_result 和指定的 keys 清單，自動產生圖表並佈局子圖。
    
    參數:
      basic_extract_result: 包含時間戳記 (Time stamps) 與各數值清單的 DataFrame
      keys: 要用來繪圖的 DataFrame 欄位名稱清單 (例如 ['torque_time_list', 'speed_time_list', ...])
      plot_file: (可選) 如果提供檔案路徑，則儲存圖形到該路徑
    """
    # 取得起始與結束時間（格式化成 YYYY-MM-DD）
    start_timestamp = basic_extract_result["Time stamps"].iloc[0]
    end_timestamp = basic_extract_result["Time stamps"].iloc[-1]
    start_date = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
    end_date = datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d')

    # 計算每筆資料相對於第一筆時間的經過分鐘數
    elapsed_time = pd.Series([
        (datetime.fromtimestamp(ts) - datetime.fromtimestamp(start_timestamp)).total_seconds() / 60
        for ts in basic_extract_result["Time stamps"]
    ])

    # 依照經過時間設定 x 軸的 tick 標記，這裡以 2000 分鐘為間隔 (可依需求調整)
    max_elapsed = int(elapsed_time.iloc[-1])
    # Dynamically calculate interval based on max elapsed time
    # Aim for roughly 8-12 ticks on x-axis
    interval = max(2000, int(max_elapsed / 10))  # At least 2000, or 1/10th of total range
    interval = round(interval / 1000) * 1000  # Round to nearest thousand
    
    x_ticks = np.arange(0, max_elapsed + 1, interval)
    x_labels = [f"{int(tick)} min" for tick in x_ticks]
    if len(x_labels) > 0:
        x_labels[0] = f"{start_date}\n{int(x_ticks[0])} min"
        x_labels[-1] = f"{end_date}\n{int(x_ticks[-1])} min"

    # 根據 keys 長度自動決定子圖排列，預設採用 2 欄排版
    n_plots = len(keys)
    ncols = 2 if n_plots > 1 else 1
    nrows = math.ceil(n_plots / ncols)

    # 建立整體圖表及子圖
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
    
    # 如果只有一個子圖，轉換為 list 以便統一後續處理
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 根據每個 key 畫出對應的曲線
    for i, key in enumerate(keys):
        ax = axes[i]
        ax.plot(elapsed_time, basic_extract_result[key])
        ax.set_xlabel("Elapsed Time (minutes)")
        ax.set_ylabel(key)  # 這裡直接使用 key 作為 Y 軸標籤，可以依需要進行字串處理
        ax.set_title(f"{key} vs Elapsed Time")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.grid(True)

    # 將多餘的子圖刪除（例如：網格超過實際繪圖數量時）
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()

    # 儲存圖檔（如果有提供檔案路徑）
    if plot_file:
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")

    plt.show(block=False)

def plot_time_list_by_section(basic_extract_result_list, basic_extract_result : pd.DataFrame, plot_keys_info: dict, plot_file=""):
    
    
    keys = list(plot_keys_info.keys())
    import math
    # 取得起始與結束時間
    start_timestamp = basic_extract_result["Time stamps"].iloc[0]
    end_timestamp = basic_extract_result["Time stamps"].iloc[-1]
    start_date = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
    end_date = datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d')

    # 計算經過分鐘數
    elapsed_time = pd.Series([
        (datetime.fromtimestamp(ts) - datetime.fromtimestamp(start_timestamp)).total_seconds() / 60
        for ts in basic_extract_result["Time stamps"]
    ])

    # 依照經過時間設定 x 軸的 tick 標記，這裡以 2000 分鐘為間隔 (可依需求調整)
    max_elapsed = int(elapsed_time.iloc[-1])
    # Dynamically calculate interval based on max elapsed time
    # Aim for roughly 8-12 ticks on x-axis
    interval = max(2000, int(max_elapsed / 10))  # At least 2000, or 1/10th of total range
    interval = round(interval / 1000) * 1000  # Round to nearest thousand
    
    x_ticks = np.arange(0, max_elapsed + 1, interval)
    x_labels = [f"{int(tick)} min" for tick in x_ticks]
    if len(x_labels) > 0:
        x_labels[0] = f"{start_date}\n{int(x_ticks[0])} min"
        x_labels[-1] = f"{end_date}\n{int(x_ticks[-1])} min"

    # 設定子圖佈局
    n_plots = len(keys)
    ncols = 2 if n_plots > 1 else 1
    nrows = math.ceil(n_plots / ncols)

    # 建立圖表
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
    
    # 處理axes格式
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 產生不同顏色
    colors = plt.cm.rainbow(np.linspace(0, 1, len(basic_extract_result_list)))

    # 繪製每個key的圖
    for i, key in enumerate(keys):
        ax = axes[i]
        
        # 為每個section畫一條線
        for section_idx, section_data in enumerate(basic_extract_result_list):
            # 計算此section的相對時間
            section_elapsed_time = pd.Series([
                (datetime.fromtimestamp(ts) - datetime.fromtimestamp(start_timestamp)).total_seconds() / 60
                for ts in section_data["Time stamps"]
            ])
            
            start_date = datetime.fromtimestamp(section_data["Time stamps"][0]).strftime('%m-%d')
            end_date = datetime.fromtimestamp(section_data["Time stamps"][-1]).strftime('%m-%d')
    
            # 繪製此section的數據
            ax.plot(section_elapsed_time, section_data[key], 
                    color=colors[section_idx], 
                    label=f'data from {start_date} to {end_date}')
        
        
        # Set y-axis limits based on plot_keys_info
        if key in plot_keys_info:
            y_min, y_max = plot_keys_info[key]
            ax.set_ylim(y_min, y_max)
            
        ax.legend(loc='lower right')
        ax.set_xlabel("Elapsed Time (minutes)")
        ax.set_ylabel(key)
        ax.set_title(f"{key} vs Elapsed Time")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.grid(True)

    # 移除多餘的子圖
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()

    # 儲存圖檔
    if plot_file:
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")

    plt.show(block=False)

def process_motor_time_list(output_dir, device_number, record_info, essemble_data_file,overwrite=False, ref_time=0):
    if not os.path.exists(output_dir):
        print("Directory does not exist, exiting the program.")
        return None

    # Set output file path
    # Create subdirectories for csv and parquet files
    csv_dir = os.path.join(output_dir, "timelist_data", "csv")
    parquet_dir = os.path.join(output_dir, "timelist_data", "parquet")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(parquet_dir, exist_ok=True)

    # Set output file paths
    output_file_parquet = os.path.join(parquet_dir, f"motor_time_list{device_number}_{record_info}.parquet")
    output_file_csv = os.path.join(csv_dir, f"motor_time_list{device_number}_{record_info}.csv")

    # Check if the data has been extracted before
    if os.path.exists(output_file_parquet) and overwrite==False:
        motor_time_list = pd.read_parquet(output_file_parquet).to_dict(orient='list')
    else:
        print(f"time list extraction for device {device_number} record {record_info} start...")

        # Read the ensemble data
        essemble_data = pd.read_hdf(essemble_data_file)

        if not essemble_data.empty:
            start_time = time.time()

            # Extract motor time list data
            motor_time_list = extract_time_list_data_essemble(essemble_data, ref_time)

            # Save as both CSV and Parquet
            df = pd.DataFrame(motor_time_list)
            df.to_csv(output_file_csv, index=False, float_format='%.6f')
            df.to_parquet(output_file_parquet)

            print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
        else:
            print("No data found in essmble file, exiting the program.")
            return None

    return motor_time_list

def extract_time_list_data_essemble(essemble_data, ref_time=0):

    # initialize time stamp list
    data_time_list, torq_time_list, spd_time_list, pwr_time_list, eff_time_list, acc_time_list,elapsed_time = [], [], [], [], [], [], [],
    
    # get the first time of the essemble data
    # Check if first_time is after March 10th
    first_time_date = datetime.fromtimestamp(int(ref_time))
    is_after_march_10 = first_time_date.date() >= datetime(2025, 3, 10).date()
    
    # initialize the first time from essemble data
    data_time_list, torq_time_list, spd_time_list, pwr_time_list, eff_time_list, acc_time_list, elapsed_time = [], [], [], [], [], [], []
    
    # iterate through each row in essemble_data DataFrame
    for index, data_read in essemble_data.iterrows():
        
      
        # deal the data of different batch (data after 1559 or after 0310 is different from the previous data)
        file_num = int(data_read['filename'].split('_')[-1].split('.')[0])
        if file_num > 1559 or is_after_march_10:
            # turn the voltage beta into phase c voltage 
            data_read["Voltage beta"] = data_read["Voltage alpha"] / 2 - data_read["Voltage beta"] * np.sqrt(3) / 2 
            torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(data_read, debug=False)
        else:
            torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(data_read, debug=False)
    
        # Calculate average torque based on array length
        
        if len(torque) > 2500:
            if len(torque) == 4000 and is_after_march_10:
                torque_avg = np.mean(np.abs(torque[-2000:]))
            else :
                torque_avg = np.mean(np.abs(torque[2250:2500]))
        else:
            torque_avg = np.mean(np.abs(torque[-500:]))

        # add data to timelist
        data_time_list.append(int(data_read["Unix Time"]))
        elapsed_time.append((parse_timestamp(int(data_read["Unix Time"])) - parse_timestamp(int(ref_time))).total_seconds() / 60)
        torq_time_list.append(torque_avg)
        spd_time_list.append(data_read["Speed"][0])
        pwr_time_list.append(power_sts["Power_E"])
        eff_time_list.append(power_sts["Efficiency"])
        acc_time_list.append(data_read["vibration rms"][0] if data_read["vibration rms"] else 0)

    basic_extract_result={
        "Time stamps":data_time_list,
        "Elapsed time" :elapsed_time, # in minute 
        "torque_time_list":torq_time_list,
        "speed_time_list": spd_time_list,
        "power_time_list": pwr_time_list,
        "efficiency_time_list": eff_time_list,
        "vibration_time_list": acc_time_list,
    }


    return basic_extract_result

def update_timelist(timelist_name, timelist_dataframe, essembel_dataframe, timelistfunction): 
    if timelist_name not in timelist_dataframe.columns:
        # 呼叫傳入的函式處理 essembel_dataframe
        new_column = timelistfunction(essembel_dataframe)
        
        # 把結果存進 timelist_dataframe
        timelist_dataframe[timelist_name] = new_column
    
    return timelist_dataframe


if __name__ == "__main__":

#%% sample data check

    #Motor device number                      
    device_number = 5
    # output file name
    output_dir = f"time_list_extraction\RUL_{device_number}"
    motor_time_list_total = pd.DataFrame()

    # data_read = rul_rd.read_rul_data('../PEWC dataset/PEWC_raw_data/0208to0217PEWC/RUL_2/RUL_Data_2_10_60.03Hz_THD-2.72%.csv')
    # data_read = rul_rd.read_rul_data('../PEWC dataset/PEWC_raw_data/0226to0310PEWC/RUL_2/RUL_Data_2_1558.parquet')
    data_read = rul_rd.read_rul_data('../PEWC dataset/PEWC_raw_data/0310to0318PEWC/RUL_2/RUL_Data_2_2754.parquet')
    # data_read = rul_rd.read_rul_data('../PEWC dataset/PEWC_raw_data/0318to0325PEWC/RUL_2/RUL_Data_2_4838.parquet')
    # data_read = rul_rd.read_rul_data('../PEWC dataset/PEWC_raw_data/0217to0226PEWC/RUL_5/RUL_Data_5_910.parquet')
    # data_read = rul_rd.read_rul_data('../PEWC dataset/PEWC_raw_data/0408to0427PEWC/RUL_2/RUL_Data_2_6441.parquet')
    
    if datetime.fromtimestamp(int(data_read["Unix Time"])).date() >= datetime(2025, 3, 1).date():
        # turn the voltage beta into phase c voltage 
        data_read["Voltage beta"] = data_read["Voltage alpha"] / 2 - data_read["Voltage beta"] * np.sqrt(3) / 2 
            
    torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(data_read, speed_v=3530, debug=True)
    #  show the short circuit analysis result
    Data_handle_in_IPC. get_cn_sts_list(data_read["Current alpha"]*193.6, data_read["Current beta"]*193.6, debug=True)
    
    # %% plot cn data time distribution
    
    record_info='0318to0325PEWC'
    
    # read the motor time list data
    parquet_dir = os.path.join(output_dir, "timelist_data", "parquet")
    output_file_parquet = os.path.join(parquet_dir, f"motor_time_list{device_number}_{record_info}.parquet")
    motor_time_list = pd.read_parquet(output_file_parquet).to_dict(orient='list')
    
    I_cn_x_timelist = motor_time_list['CNfault_time_list']
    I_cn_y_timelist = motor_time_list['Icn_y']
    
    cn_threshold=0.05
    
    # plot the CN xy plot result
    plt.figure(figsize=(10, 5))
    # 畫出半徑為 cn_threshold 的圓
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = cn_threshold * np.cos(theta)
    circle_y = cn_threshold * np.sin(theta)
    plt.plot(circle_x, circle_y, 'k--', alpha=0.5, label=f'CN threshold range = {cn_threshold}')

    # 使用 scatter 並以時間順序變色（這裡以 index 作為時間指標）
    time_index = np.arange(len(I_cn_x_timelist))
    scatter = plt.scatter(I_cn_x_timelist, I_cn_y_timelist, c=time_index, cmap='viridis', marker='x', label='CN index points')

    # 加上 colorbar 表示時間對應色帶
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time Index')

    plt.xlim(-(cn_threshold + 0.02), (cn_threshold + 0.02))
    plt.ylim(-(cn_threshold + 0.02), (cn_threshold + 0.02))
    # Add three phase lines at 0, 120, and 240 degrees
    angles = [0, 120, 240]  # degrees
    line_length = cn_threshold + 0.02
    for angle in angles:
        rad = np.deg2rad(angle)
        x = [0, line_length * np.cos(rad)]
        y = [0, line_length * np.sin(rad)]
        plt.plot(x, y, 'r--', alpha=0.5)
    
    plt.gca().set_aspect('equal')
    
    plt.xlabel('unbalanced current in x-axis (p.u.)')
    plt.ylabel('unbalanced current in y-axis (p.u.)')

    start_date = datetime.fromtimestamp(motor_time_list["Time stamps"][0]).strftime('%Y-%m-%d')
    end_date = datetime.fromtimestamp(motor_time_list["Time stamps"][-1]).strftime('%Y-%m-%d')
    plt.title(f'Time Distribution of winding fault index of motor {device_number}\nFrom {start_date} to {end_date}')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

     
    #%% time list extraction
    
    # collection date reference 
    record_info = '0208to0217PEWC'  
    # import essemble data name 
    essemble_data_file = f"../PEWC dataset/PEWC_essembled_data/RUL_{device_number}/{record_info}_essemble.h5"
    essemble_data = pd.read_hdf(essemble_data_file)
    time_reference= essemble_data.iloc[0]["Unix Time"]
     
    record_infos = [
        '0208to0217PEWC',
        '0217to0226PEWC', 
        '0226to0310PEWC',
        '0310to0318PEWC',
        '0318to0325PEWC',
        '0325to0408PEWC',
        '0408to0427PEWC'
    ]
    # time_list_overwrite = False
    time_list_overwrite = True

    # Collection dates in chronological order
   
    # record_info = '0120to0208PEWC'  
    # # import essemble data name 
    # essemble_data_file = f"../PEWC dataset/PEWC_essembled_data/RUL_{device_number}/{record_info}_essemble.h5"
    # motor_time_list = process_motor_time_list(output_dir, device_number, record_info, essemble_data_file, overwrite=time_list_overwrite, ref_time=time_reference)
    # motor_time_list_total = pd.concat([motor_time_list_total, pd.DataFrame(motor_time_list)], ignore_index=True)
    # plot_file_name = os.path.join(output_dir, f"timelist_figures/time_list_plot{device_number}_{record_info}.png")
    # plot_basic_time_list_df(pd.DataFrame(motor_time_list), plot_file_name)

    # the timelist to be ploted
    plot_keys_list = ['torque_time_list', 'power_time_list', 'efficiency_time_list', 'CNfault_time_list']
    # collection of the time list data for ploting
    motor_timelist_section_inline=[]
    # Process each record_info
    for record_info in record_infos:
        essemble_data_file = f"../PEWC dataset/PEWC_essembled_data/RUL_{device_number}/{record_info}_essemble.h5"
        # basic time list extraction
        motor_time_list = process_motor_time_list(output_dir, device_number, record_info, essemble_data_file, 
                                                overwrite=time_list_overwrite, ref_time=time_reference)
        # specific time list extraction
        motor_time_list= Analysis_helpler.extract_cn_timelist(motor_time_list, "CNfault_time_list",output_dir, device_number, record_info, essemble_data_file, 
                                                update=time_list_overwrite, ref_time=time_reference)
        # concatenate the time list data
        # motor_time_list_total = pd.concat([motor_time_list_total, pd.DataFrame(motor_time_list)], ignore_index=True)
        
        motor_timelist_section_inline.append(motor_time_list)
        
        # plot the time list result
        plot_file_name = os.path.join(output_dir, f"timelist_figures/time_list_plot{device_number}_{record_info}.png")
        plot_basic_time_list_df(pd.DataFrame(motor_time_list), plot_keys_list, plot_file_name)

    
    
    
    # Combine all time list sections into motor_time_list_total
    motor_time_list_total = pd.concat([pd.DataFrame(section) for section in motor_timelist_section_inline], ignore_index=True)
    
    # save the time list summary 
    output_file = os.path.join(output_dir, f"motor_time_list{device_number}_Summary.parquet")
    csv_output_file = output_file.replace('.parquet', '.csv')
    motor_time_list_total.to_csv(csv_output_file, index=False, float_format='%.6f')
    motor_time_list_total.to_parquet(output_file)
    
    #plot the time list result
    plot_file_name = os.path.join(output_dir, f"timelist_figures/time_list_plot{device_number}_Summary.png")
    plot_basic_time_list_df(motor_time_list_total, plot_keys_list, plot_file_name)
    
    plot_keys_info = {
        'torque_time_list': [0 , 40],
        'power_time_list': [ 5000, 15000], 
        'efficiency_time_list':[0 , 100],
        'CNfault_time_list': [0 , 0.05]
    }
    plot_time_list_by_section(motor_timelist_section_inline, motor_time_list_total, plot_keys_info, plot_file_name)


# %%
