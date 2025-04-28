"""
This program is used to extract the time list data of motor state from the raw data of PEWC.
The data is extracted file by file and the time list data is saved as a parquet file.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import rul_data_read
import rul_data_read as rul_rd
import Data_handle_in_IPC
import time
import os

def parse_timestamp(timestamp):
    """將 Unix 時間轉換為 datetime 物件"""
    return datetime.fromtimestamp(timestamp)


def extract_time_list_data(files):

    # initialize time stamp list
    data_time_list, torq_time_list, spd_time_list, pwr_time_list, eff_time_list, acc_time_list,elapsed_time = [], [], [], [], [], [], [],
    
    data_read = rul_rd.read_rul_data(files[0])
    first_time = data_read["Unix Time"]
  
    for file in files:
    
        data_read = rul_rd.read_rul_data(file)  # 這裡假設檔案是 CSV 格式

        if first_time is None:
            first_time = data_read["Unix Time"]  # 記錄第一個檔案的起始時間

        if file.endswith('.csv'):
            torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(data_read, debug=False)
        else:
            # deal the data of different batch
            file_num = int(os.path.splitext(os.path.basename(file))[0].split('_')[-1])
            if file_num > 1559 :
                data_read["Voltage beta"] = data_read["Voltage alpha"] / 2 - data_read["Voltage beta"] * np.sqrt(3) / 2
                torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(data_read, debug=False)
            else:
                torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(data_read, debug=False)
        
        # Calculate average torque based on array length
        if len(torque) > 2500:
            torque_avg = np.mean(np.abs(torque[2250:2500]))
        else:
            torque_avg = np.mean(np.abs(torque[-500:]))

        # add data to timelist
        data_time_list.append(int(data_read["Unix Time"]))
        elapsed_time.append((parse_timestamp(int(data_read["Unix Time"])) - parse_timestamp(int(first_time))).total_seconds() / 60)
        torq_time_list.append(torque_avg)
        spd_time_list.append(data_read["Speed"][0])
        pwr_time_list.append(power_sts["Power_E"])
        eff_time_list.append(torque_avg/power_sts["Power_E"])
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


def plot_basic_time_list(basic_extract_result):
    tt=basic_extract_result["Time stamps"][-2]
    plt.figure(figsize=(12, 10))
    start_date = datetime.fromtimestamp(basic_extract_result["Time stamps"][0]).strftime('%Y-%m-%d')
    end_date = datetime.fromtimestamp(basic_extract_result["Time stamps"][-1]).strftime('%Y-%m-%d')

    # Generate x-ticks for elapsed time in integer minutes
    x_ticks = np.arange(0, int(basic_extract_result["Elapsed time"][-1]) + 1, 2000)
    x_labels = [f"{int(tick)} min" for tick in x_ticks]
    x_labels[0] = f"{start_date}\n{int(x_ticks[0])} min"
    x_labels[-1] = f"{end_date}\n{int(x_ticks[-1])} min"

    # Subplot for Torque
    plt.subplot(3, 2, 1)
    plt.plot(basic_extract_result["Elapsed time"], basic_extract_result["torque_time_list"], linestyle='-')
    plt.xlabel("Elapsed Time (minutes)")
    plt.ylabel("Torque")
    plt.title("Torque vs Elapsed Time")
    plt.xticks(x_ticks, x_labels, rotation=45)  
    plt.grid()

    # Subplot for Speed
    plt.subplot(3, 2, 2)
    plt.plot(basic_extract_result["Elapsed time"], basic_extract_result["speed_time_list"], linestyle='-')
    plt.xlabel("Elapsed Time (minutes)")
    plt.ylabel("Speed")
    plt.title("Speed vs Elapsed Time")
    plt.xticks(x_ticks, x_labels, rotation=45)
    plt.grid()

    # Subplot for Power
    plt.subplot(3, 2, 3)
    plt.plot(basic_extract_result["Elapsed time"], basic_extract_result["power_time_list"], linestyle='-')
    plt.xlabel("Elapsed Time (minutes)")
    plt.ylabel("Power")
    plt.title("Power vs Elapsed Time")
    plt.xticks(x_ticks, x_labels, rotation=45)
    plt.grid()

    # Subplot for Efficiency
    plt.subplot(3, 2, 4)
    plt.plot(basic_extract_result["Elapsed time"], basic_extract_result["efficiency_time_list"], linestyle='-')
    plt.xlabel("Elapsed Time (minutes)")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Elapsed Time")
    plt.xticks(x_ticks, x_labels, rotation=45)
    plt.grid()

    # Subplot for Vibration
    plt.subplot(3, 2, 5)
    plt.plot(basic_extract_result["Elapsed time"], basic_extract_result["vibration_time_list"], linestyle='-')
    plt.xlabel("Elapsed Time (minutes)")
    plt.ylabel("Vibration")
    plt.title("Vibration vs Elapsed Time")
    plt.xticks(x_ticks, x_labels, rotation=45)
    plt.grid()
    plt.tight_layout()

    # Save the plot
    plot_output_folder = "time_list_extraction/"
    if not os.path.exists(plot_output_folder):
        os.makedirs(plot_output_folder)
    plot_file = os.path.join(plot_output_folder, f"time_list_plot{device_number}_{record_info}.png")
    plt.savefig(plot_file)  
    print(f"Plot saved to {plot_file}")

    # show  the plot resut 
    plt.show()



if __name__ == "__main__":

#%% data check
    # data_read = rul_rd.read_rul_data('../PEWC dataset/read test data/RUL_Data_2_10_60.03Hz_THD-2.72%.csv')
    # data_read = rul_rd.read_rul_data('../PEWC dataset/PEWC_raw_data/0226to0310PEWC/RUL_5/RUL_Data_5_1711.parquet')
    # data_read = rul_rd.read_rul_data('../PEWC dataset/PEWC_raw_data/0217to0226PEWC/RUL_5/RUL_Data_5_902.parquet')

    # torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(data_read, debug=True)
#%%      

    device_number = 2
    record_info = '0208to0217PEWC_test'  # the number of collect times
    
    # output file name
    # Check and create output directory if it doesn't exist
    output_dir = "time_list_extraction"
    if not os.path.exists(output_dir):
        print(f"NO directory : {output_dir}")
        exit()    
    output_file = f"{output_dir}/motor_time_list{device_number}_{record_info}.parquet"


    print(f"time list extraction for device {device_number} record {record_info} start...")

    # for unstable data collect before 0217
    folder_path = f"../PEWC dataset/PEWC_processed_data/PEWC_collect2/RUL_{device_number}_60Hz_thd5"
    # for stable data after 0217
    
    # # folder_path = f"../PEWC dataset/PEWC_raw_data/0217to0226PEWC/RUL_{device_number}"
    # folder_path = f"../PEWC dataset/PEWC_raw_data/{record_info}/RUL_{device_number}"

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv") or f.endswith(".parquet")]
    if files[0].endswith('.parquet'):
        files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
    else:
        files.sort(key=lambda f: os.path.getctime(f))
    # only for test
    # file_read_number=50 ; files = files[:file_read_number]
    

    if files:
        # record the start time for time cost calculation
        start_time = time.time()
        #get the time list data of motor state 
        motor_time_list = extract_time_list_data(files)

        # Convert to dataframe and save as both parquet and csv
        df = pd.DataFrame(motor_time_list)
        csv_output_file = output_file.replace('.parquet', '.csv')
        df.to_csv(csv_output_file, index=False, float_format='%.6f')
        df.to_parquet(output_file)

        print(f"Time elapsed: {time.time() - start_time} seconds")
    else:
        print("No CSV files found, exiting the program.")
        exit()

    # plot the time list result
    plot_basic_time_list(motor_time_list)


# %%
