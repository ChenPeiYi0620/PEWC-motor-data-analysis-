from sklearn.model_selection import train_test_split
from keras.src.models import Model
from keras.src.saving.saving_api import load_model
from keras.src.layers import Input, Conv1D, Add, Activation, UpSampling1D, Dense, Flatten, Reshape, Concatenate
from keras.src.layers import Input, Conv1D, Dense, concatenate, RepeatVector, MaxPooling1D, Activation ,UpSampling1D, Conv1DTranspose
from keras.src.utils import plot_model
import os
import numpy as np 


import pandas as pd
from openpyxl import Workbook
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import rul_data_read as rul_rd
import csv
import io 

def read_mse_results(csv_path):
    
    # Read MSE results from CSV file
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        mse_data = list(reader)
    # Convert string values to float and return as numpy array
    golden_sample_mses = np.array([float(x) for x in mse_data[0]])
    return golden_sample_mses

df = pd.read_csv('D:\Pycharm_conda_projects\PEWC data analysis\PEWC_data_analysis\PEWC_CCAE_develpoment\Golden_sample_Mses.csv')
golden_sample_Mses = df.iloc[:, 0]

def single_signal_reconstruction(signal, model, label):
    # 將信號數據轉換為符合模型輸入格式的形狀 (1, 1024, 1)
    signal_segments = data_augmentation(signal, time_steps=1024, window_size=1, cols=[0], random_seed=42)
    label = np.array([label]).reshape(1, 1)  # 將標籤轉換為正確的形狀 (1, 1)

    # 使用模型進行重建
    # Add channel dimension and repeat the label for each signal
    signal_segments = signal_segments.reshape(signal_segments.shape[0], signal_segments.shape[1], 1)  # reshape to (n, 1024, 1)
    label = np.repeat(label, signal_segments.shape[0], axis=0)  # repeat label n times
    # Predict reconstructed signals
    reconstructed_segments = model.predict([signal_segments, label])

    def reconstruct_from_segments(segments, step_size=1):
        reconstructed_full = np.zeros((segments.shape[0]-1)*step_size + segments.shape[1])
        count = np.zeros_like(reconstructed_full)

        for i in range(segments.shape[0]):
            start = i * step_size
            reconstructed_full[start:start+segments.shape[1]] += segments[i, :, 0]
            count[start:start+segments.shape[1]] += 1

        return reconstructed_full / np.maximum(count, 1)
    reconstructed_signal=reconstruct_from_segments(reconstructed_segments, step_size=1)

    # 將重建的信號轉換為一維數組
    reconstructed_signal = np.squeeze(reconstructed_signal)

    return reconstructed_signal

class plot_hlpler:
    def plot_current(data1, data2 = np.array([])):
    
        plt.figure(figsize=(12, 8))  # Set the figure size

        plt.plot(data1, label="Current alpha")
        if data2.any():
            plt.plot(data2, label="Current alpha")
        plt.xlabel("Sample Index")
        plt.ylabel("Current")
        plt.title("Current alpha and beta")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()  # Adjust subplots to fit into the figure area.
        plt.show()  # Display the figure)
    
    def plot_CCAE_reconstrcut_signal(raw_data, model, label):
        reconstrcut_signal= single_signal_reconstruction(raw_data, model, label)
        plt.figure(figsize=(12, 8))
        plt.plot(raw_data.flatten(), label='Raw Signal', alpha=0.7)
        plt.plot(reconstrcut_signal, label='Reconstructed Signal', alpha=0.7)
        plt.plot(raw_data.flatten()-reconstrcut_signal, label='Reconstructed Signal', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title('Raw vs Reconstructed Signal')
        plt.legend()
        plt.grid(True)
        plt.show()

def data_augmentation(df, time_steps, window_size, cols=None, random_seed=None):
    # 如果未指定 cols 參數，則預設使用資料框的所有欄位
    if cols is None:
        cols = df.columns

    # 初始化一個空的列表來存放提取出的樣本數據
    samples_list = []

    # 對指定的每一列進行滑動窗口操作
    for col in cols:
        # 根據窗口大小和時間步長，從每列中提取子序列樣本
        for i in range(0, len(df) - time_steps + 1, window_size):
            # 使用 iloc 根據索引提取從 i 到 i + time_steps 的時間段的數據
            # 並將其轉換為 NumPy 陣列，方便進行後續的數據處理
            # samples_list.append(df.iloc[i:i + time_steps].to_numpy())
            samples_list.append(df[i:i + time_steps])

    # 將收集到的所有樣本轉換成 NumPy 多維陣列
    final_data = np.array(samples_list)

    # 如果指定了 random_seed，則設置隨機種子，確保數據打亂時的隨機性是可重現的
    if random_seed is not None:
        np.random.seed(random_seed)

    # 返回增強後的數據集，這是一個 NumPy 陣列
    return final_data

def Bhattacharyya_Distance(Normal_data_mse_errors, Abnormal_data_mse_errors):
    # 計算兩組數據的均值和標準差
    mu_normal, sigma_normal = np.mean(Normal_data_mse_errors), np.std(Normal_data_mse_errors)
    mu_abnormal, sigma_abnormal = np.mean(Abnormal_data_mse_errors), np.std(Abnormal_data_mse_errors)

    # 計算 Bhattacharyya 距離
    def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
        term1 = (mu1 - mu2)**2 / (sigma1**2 + sigma2**2)
        term2 = np.log((sigma1 + sigma2) / (2 * np.sqrt(sigma1 * sigma2)))
        return 0.25 * (term1 + term2)

    # 計算兩組分布的 Bhattacharyya 距離
    distance = bhattacharyya_distance(mu_normal, sigma_normal, mu_abnormal, sigma_abnormal)

    return distance

def CCAE_hist_compare(target_data, model_name, label, figure_name, debug=False):

    # 預防模型名稱未打.keras
    if '.' not in model_name:
        model_name += '.keras'
    loaded_model = load_model(model_name)
    

    # 將資料做帶重疊的切片
    augmented_target=(data_augmentation(target_data, time_steps=1024, window_size=5, cols=[0], random_seed=42))
   
    # 製作標籤
    labels_Target = np.full(augmented_target.shape[0], label)
    # 模型預測 (samples, signal_length, num_features)
    reconstructed_Target = loaded_model.predict([augmented_target, labels_Target]) 
    # 將預測後之為度改為(samples, signal_length)，移除所有大小为為 1 的维度
    reconstructed_Target_squeezed = np.squeeze(reconstructed_Target)
    augmented_target = np.squeeze(augmented_target)  # 變成 (98,1024)
    # Calculate MSE
    Target_data_mse_errors = np.mean(np.square(augmented_target - reconstructed_Target_squeezed), axis=1)
    
    #根據目標資料長度 重新採樣golden sample 資料點
    # Set random seed for reproducibility
    np.random.seed(42)
    resize_gs_mses = np.random.choice(golden_sample_Mses, size=len(Target_data_mse_errors), replace=False)

    # 計算誤差相似度
    BD = Bhattacharyya_Distance(resize_gs_mses, Target_data_mse_errors)
    # 將浮點數格式化為保留三位小數
    BD = f"{BD:.3f}"
    # 替換小數點為 'P'
    BD = BD.replace('.', 'P')


    if debug:
        # 繪製正常和異常樣本的MSE誤差分布
        plt.figure(figsize=(10, 6))
        counts_golden, bin_edges_golden, _=plt.hist(resize_gs_mses , bins=20, alpha=0.7, label='Golden Sample')
        counts_traget, bin_edges_target, _=plt.hist(Target_data_mse_errors, bins=20, alpha=0.7, label='Target Data')
        plt.xlabel('MSE Error', fontsize=14)
        plt.tick_params(axis='x', labelsize=14)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.ylabel('Number of Samples', fontsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.legend(['Golden Sample', 'Target Data'])
        plt.show(block=False)
        plt.close()
    else: 
        counts_golden, bin_edges_golden = np.histogram(resize_gs_mses, bins=15)
        counts_traget, bin_edges_target = np.histogram(Target_data_mse_errors, bins=15)
        
    #若需要檢視結果則可印出
    # for i in range(len(counts_traget)):
    #     print(f'Bin {i+1}: [{bin_edges_target[i]:.3f}, {bin_edges_target[i+1]:.3f}) -> Count: {int(counts_traget[i])}')
    
    return BD, [counts_golden, bin_edges_golden, counts_traget, bin_edges_target] 



# %%
if __name__ == '__main__':
    print("Running PEWC_CCAE_standalone.py directly")
    
    # 設定當前工作目錄為腳本所在的目錄
# 這樣可以確保相對路徑正確
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 設定要讀取的資料夾
test_file="../../PEWC dataset/PEWC_raw_data/0310to0318PEWC/RUL_2"
mormal_data_read = rul_rd.read_rul_data('../../PEWC dataset/PEWC_raw_data/0310to0318PEWC/RUL_2/RUL_Data_2_2754.parquet')
# mormal_data_read = rul_rd.read_rul_data('../../PEWC dataset/PEWC_raw_data/0318to0325PEWC/RUL_2/RUL_Data_2_4837.parquet')
test_current_normal=mormal_data_read["Current alpha"]
test_current_normal=( test_current_normal - np.mean(test_current_normal) ) / np.std(test_current_normal) # 標準化

abnormal_data_read = rul_rd.read_rul_data('../../PEWC dataset/PEWC_raw_data/0310to0318PEWC/RUL_5/RUL_Data_5_2754.parquet')
test_current_abnormal=abnormal_data_read["Current alpha"]
test_current_abnormal=( test_current_abnormal - np.mean(test_current_abnormal) ) / np.std(test_current_abnormal) # 標準化


motor_folder = '../../PEWC dataset/PEWC_raw_data/0310to0318PEWC/RUL_2'
motor_abnormal_folder = '../../PEWC dataset/PEWC_raw_data/0310to0318PEWC/RUL_5'

Is_predict = 1 # 決定是預測還是訓練的旗標

model_name = "model_pei.keras"
figure_name = "figure_1"

# 預防模型名稱未打.keras
if '.' not in model_name:
    model_name += '.keras'
loaded_model = load_model(model_name)

plot_hlpler.plot_CCAE_reconstrcut_signal(test_current_normal, loaded_model, label=1) # 畫出單一電流的重建圖
plot_hlpler.plot_CCAE_reconstrcut_signal(test_current_abnormal, loaded_model, label=1) # 畫出單一電流的重建圖
CCAE_hist_compare(test_current_normal, model_name, label=1, figure_name=figure_name, debug=True) # 畫出正常電流的重建圖
CCAE_hist_compare(test_current_abnormal, model_name, label=1, figure_name=figure_name, debug=True) # 畫出正常電流的重建圖

#%%