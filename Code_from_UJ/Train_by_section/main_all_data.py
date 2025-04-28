import CCAE_function_1024_point_all_data
import os 
import time 
import pandas as pd
import matplotlib.pyplot as plt
from  CCAE_function_1024_point_all_data import get_datetime_from_unix as get_date
from CCAE_function_1024_point_all_data import read_rul_data as read_rul
import random
import numpy as np
#--------------------------------------------------------------------------
# venv: CCAE_function_1024_point_all_data
# python: 3.10.0
# C:\Users\MotorTech\PycharmProjects\Tecom_AQbox_Master\venv\Scripts\python.exe


# 設定當前工作目錄為腳本所在的目錄
# 這樣可以確保相對路徑正確
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 設定要讀取的資料夾
motor_folder = '../../PEWC dataset/PEWC_raw_data/0310to0318PEWC/RUL_2'
motor_abnormal_folder = '../../PEWC dataset/PEWC_raw_data/0310to0318PEWC/RUL_5'



# Get list of parquet files
parquet_files = sorted([f for f in os.listdir(motor_folder) if f.endswith('.parquet')])

# Get first and last data 
first_data = read_rul(os.path.join(motor_folder, parquet_files[0]))
last_data = read_rul(os.path.join(motor_folder, parquet_files[-1]))

start_date= get_date(first_data["Unix Time"])
end_date= get_date(last_data["Unix Time"])
print("start date: ", start_date)
print("end date: ", end_date)

# 設定讀取資料夾中資料的範圍
select_range = 10
data_range =  len([f for f in os.listdir(motor_folder) if f.endswith('.parquet')])

Is_predict = 1 # 決定是還是訓練的旗標

model_name = "model.keras"
figure_name = "figure_1"

seg_size=10

RUL_2 = []
RUL_5 = []
# 使用訓練好的模型做預測
if Is_predict == 1:

   
    
    
    # 確認 "CCAE_MSE_results.csv" 是否存在
    if not os.path.exists("CCAE_MSE_results.csv"):          
        #  run the model to get the MSE
        # load the data 
        motor_data_list = CCAE_function_1024_point_all_data.load_range(motor_folder, start=1, end=data_range)  
        motor_abnormal_data_list = CCAE_function_1024_point_all_data.load_range(motor_abnormal_folder, start=1, end=data_range)

        # 取得所有重建MSE誤差
        RUL2_CCAE_Mse=CCAE_function_1024_point_all_data.get_CCAE_MSE(raw_data= motor_data_list, model_name = model_name , label = 1, window_size=seg_size )
        RUL5_CCAE_Mse=CCAE_function_1024_point_all_data.get_CCAE_MSE(raw_data= motor_abnormal_data_list, model_name = model_name , label = 1, window_size=seg_size )

        # Save RUL2_CCAE_Mse and RUL5_CCAE_Mse to CSV
        mse_data = {
            "RUL2_CCAE_Mse": RUL2_CCAE_Mse,
            "RUL5_CCAE_Mse": RUL5_CCAE_Mse
        }

        mse_df = pd.DataFrame(mse_data)
        mse_df.to_csv("CCAE_MSE_results.csv", index=False)
    else:
        mse_data= pd.read_csv("CCAE_MSE_results.csv")

    # get the segment number per batch 
    segment_number_perfile=(2000-1024)//seg_size+1
    
    RUL2_CCAE_Mse = mse_data["RUL2_CCAE_Mse"].values
    RUL5_CCAE_Mse = mse_data["RUL5_CCAE_Mse"].values
    RUL2_BDs, RUL2_segmented_mse= CCAE_function_1024_point_all_data. get_CCAE_MSE_BD(RUL2_CCAE_Mse, RUL2_CCAE_Mse, segment_number_perfile, file_batch=10, plot=False)
    RUL5_BDs, RUL5_segmented_mse= CCAE_function_1024_point_all_data. get_CCAE_MSE_BD(RUL2_CCAE_Mse, RUL5_CCAE_Mse, segment_number_perfile, file_batch=10, plot=False)


    #%% plot CCAE results 
    
    # Plot RUL2_CCAE_Mse
    # Scatter plot for RUL2_segmented_mse
    ref_mse = RUL2_segmented_mse[0]
    down_sample_size=min(200, len(ref_mse))
    maker_size=10
    plt.figure(figsize=(10, 6))
    for idx, mse_values in enumerate(RUL2_segmented_mse):
        if idx == 0:
            random.seed(42)  # Set a fixed seed for reproducibility
            sample_size = min(down_sample_size, len(mse_values))
            sampled_values = random.sample(list(ref_mse), sample_size)
            plt.scatter([idx] * sample_size, sampled_values, alpha=0.7, s=maker_size, marker='x', color='red', label='reference')
        else:
            random.seed(42)  # Set a fixed seed for reproducibility
            sample_size = min(down_sample_size, len(mse_values))
            sampled_values = random.sample(list(mse_values), sample_size)
            plt.scatter([idx] * sample_size, sampled_values, alpha=0.7, s=maker_size, marker='.', color='blue')
    plt.title(f'RUL2 Segmented MSE Distribution\n{start_date} to {end_date}')
    plt.xlabel('Time Index [every 10 samples]')
    plt.ylabel('MSE Value')
    plt.legend(['reference', 'RUL2_segmented_mse'])
    plt.grid(True)
    plt.savefig('RUL2_Segmented_MSE_Distribution.png')
    plt.ylim([0, 0.005])
    plt.show()

    # Scatter plot for RUL5_segmented_mse
    plt.figure(figsize=(10, 6))
    for idx, mse_values in enumerate(RUL5_segmented_mse):
        if idx == 0:
            random.seed(42)  # Set a fixed seed for reproducibility
            sample_size = min(down_sample_size, len(mse_values))
            sampled_values = random.sample(list(ref_mse), sample_size)
            plt.scatter([idx] * sample_size, sampled_values, alpha=0.7, s=maker_size, marker='x', color='red', label='reference')
        else:
            random.seed(42)  # Set a fixed seed for reproducibility
            sample_size = min(down_sample_size, len(mse_values))
            sampled_values = random.sample(list(mse_values), sample_size)
            plt.scatter([idx] * sample_size, sampled_values, alpha=0.7, s=maker_size, marker='.', color='blue')
    
    plt.title(f'RUL5 Segmented MSE Distribution\n{start_date} to {end_date}')
    plt.xlabel('Time Index [every 10 samples]')
    plt.ylabel('MSE Value')
    plt.legend(['reference', 'RUL5_segmented_mse'])
    plt.grid(True)
    plt.savefig('RUL5_Segmented_MSE_Distribution.png')
    plt.ylim([0, 0.005])
    plt.show()
    #%%
    plt.figure(figsize=(10, 6))
    plt.plot(RUL2_CCAE_Mse,label='RUL2_CCAE_Mse', linewidth=1)
    plt.plot(RUL5_CCAE_Mse, label='RUL5_CCAE_Mse', linewidth=1)
    plt.title(f'RUL2 CCAE MSE\n{start_date} to {end_date}')
    plt.xlabel('Sample')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('RUL2_CCAE_MSE.png')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(RUL2_BDs[1:],label=f'RUL2_BDs, mean BD={np.array(RUL2_BDs).mean():.4f}')
    plt.plot(RUL5_BDs[1:],label=f'RUL5_BDs, mean BD={np.array(RUL5_BDs).mean():.4f}')
    plt.title(f'BD of reconstruction error\n{start_date} to {end_date}')
    plt.xlabel('Time Index [every 10 samples]')
    plt.ylabel('BD')
    plt.legend()
    plt.grid(True)
    plt.savefig('RUL2_CCAE_BD.png')
    plt.show()


     #%% 
# 訓練新模型
if Is_predict == 0:

    # load the data 
    motor_data_list = CCAE_function_1024_point_all_data.load_range(motor_folder, start=1, end=data_range)  
    motor_abnormal_data_list = CCAE_function_1024_point_all_data.load_range(motor_abnormal_folder, start=1, end=data_range)
    
    CCAE_function_1024_point_all_data.CCAE_train(Motor_data = motor_data_list,
                            model_name = model_name)

# CCAE_function_1024_point_all_data.plot_current(motor_data_list[1], motor_abnormal_data_list[1])
# %%
