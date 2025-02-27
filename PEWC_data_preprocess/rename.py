import os
import re
import shutil
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

def calculate_thd(signal, sampling_rate=20000):
    """
    計算總諧波失真 (THD)
    """
    N = len(signal)
    if N == 0:
        return "NaN"
    
    T = 1 / sampling_rate
    freq_values = fftfreq(N, T)[:N // 2]
    fft_magnitudes = np.abs(fft(signal)[:N // 2])
    
    # 找到基頻的索引 (最大頻率成分)
    fundamental_index = np.argmax(fft_magnitudes)
    fundamental_magnitude = fft_magnitudes[fundamental_index]
    
    if fundamental_index == 0 or fundamental_magnitude == 0:
        return "NaN"  # 避免 slice step 為 0 的錯誤
    
    # 計算諧波能量，選取基頻的整數倍頻率
    harmonic_indices = np.arange(2 * fundamental_index, len(fft_magnitudes), fundamental_index)
    harmonic_magnitudes = fft_magnitudes[harmonic_indices]
    
    thd = np.sqrt(np.sum(harmonic_magnitudes**2)) / fundamental_magnitude * 100  # 百分比表示
    
    return f"{thd:.2f}%"

def get_dominant_frequency_and_thd(csv_file):
    df = pd.read_csv(csv_file, skiprows=2)  # 跳過前兩行標題
    if df.empty or df.shape[1] < 3:  # 確認至少有三欄資料
        return "NaN", "NaN"
    
    signal = df.iloc[:, 2].dropna().values  # 假設第三欄為信號數據
    N = len(signal)
    if N == 0:
        return "NaN", "NaN"
    
    T = 1 / 10000  # 採樣率 20kHz
    freq_values = fftfreq(N, T)[:N // 2]
    fft_magnitudes = np.abs(fft(signal)[:N // 2])
    
    dominant_freq = freq_values[np.argmax(fft_magnitudes)]  # 找到最大振幅對應的頻率
    thd = calculate_thd(signal)
    
    return f"{dominant_freq:.2f}Hz", thd

def rename_csv_files(folder_path, old_name, new_name, start_index, target_folder):
    # 讀取CSV檔，並按照順序排列
    csv_files = sorted(
        [f for f in os.listdir(folder_path) if f.startswith(old_name) and f.endswith(".csv")],
        key=lambda x: int(re.search(rf"{old_name}_(\d+)", x).group(1))
    )
    
    for index, file in enumerate(csv_files, start=start_index):
        old_file_path = os.path.join(folder_path, file)
        dominant_freq, thd = get_dominant_frequency_and_thd(old_file_path)
        new_file_name = f"{new_name}_{index}_{dominant_freq}_THD-{thd}.csv"
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)
        print(f"{file} 已更名為 {new_file_name}")
        
        # 若基頻為 60.03Hz 且 THD 小於 5%，則複製到目標資料夾
        thd_value = float(thd.replace('%', '')) if thd != "NaN" else 100
        # thd_value < 5 代表 5% 以內
        if dominant_freq == "60.03Hz" and thd_value < 5:
            shutil.copy(new_file_path, os.path.join(target_folder, new_file_name))
            print(f"{new_file_name} 已複製到 {target_folder}")


# 此程式會將目標資料夾的檔案重新命名，並取出符合條件的檔案複製一份至新的資料夾

# 設定 CSV 檔案所在的資料夾與新名稱
folder_path = "../PEWC dataset/PEWC_raw_data/0217PEWC/RUL_5"  # 請更改為你的資料夾路徑
target_folder = "../PEWC dataset/PEWC_processed_data/PEWC_collect2/RUL_5_60Hz_thd5"  # 另存新檔的目標資料夾
os.makedirs(target_folder, exist_ok=True)  # 確保目標資料夾存在

old_name = "RUL_Data_5"  # 舊的命名前綴
new_name = "RUL_Data_5"  # 新的命名前綴
start_index = 1  # 起始編號
rename_csv_files(folder_path, old_name, new_name, start_index, target_folder)

