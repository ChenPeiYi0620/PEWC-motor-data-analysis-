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
    first_time = None

    for file in files:
        data_read = rul_rd.read_rul_data(file)  # 這裡假設檔案是 CSV 格式

        if first_time is None:
            first_time = data_read["Unix Time"]  # 記錄第一個檔案的起始時間
        # print(data_read["Unix Time"])
        torque, _, _, v_alpha, v_beta, power_sts = Data_handle_in_IPC.estimate_torque(data_read["Voltage alpha"], data_read["Voltage beta"],
                                                                                 data_read["Current alpha"], data_read["Current beta"], debug=False)
        torque_avg=np.mean(torque[:500])     
        
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
    plot_file = os.path.join(plot_output_folder, f"time_list_plot{device_number}_{record_number}.png")
    plt.savefig(plot_file)  
    print(f"Plot saved to {plot_file}")

    # show  the plot resut 
    plt.show()



if __name__ == "__main__":
    device_number = 5
    record_number = 2  # the number of collect times

    output_file = f"time_list_extraction/motor_time_list{device_number}_{record_number}.parquet"
    if os.path.exists(output_file):
        motor_time_list = pd.read_parquet(output_file).to_dict(orient='list')
    else:
        folder_path = f"../PEWC dataset/PEWC_processed_data/PEWC_collect2/RUL_{device_number}_60Hz_thd5"
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
        files.sort(key=lambda f: os.path.getmtime(f))

        if files:
            start_time = time.time()

            motor_time_list = extract_time_list_data(files)
            df = pd.DataFrame(motor_time_list)
            df.to_parquet(output_file)

            print(f"Time elapsed: {time.time() - start_time} seconds")
        else:
            print("No CSV files found, exiting the program.")
            exit()

    # plot the time list result
    plot_basic_time_list(motor_time_list)

