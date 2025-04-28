from sklearn.model_selection import train_test_split
from keras.src.models import Model
from keras.src.saving.saving_api import load_model
from keras.src.layers import Input, Conv1D, Dense, concatenate, RepeatVector, MaxPooling1D, Activation ,UpSampling1D, Conv1DTranspose
from keras.src.utils import plot_model
import numpy as np 
import pandas as pd
from openpyxl import Workbook
import matplotlib.pyplot as plt
import os
import csv


def get_datetime_from_unix(unix_time):
    # 將 Unix 時間戳轉換為可讀的日期時間格式
    return pd.to_datetime(unix_time, unit='s').strftime('%Y-%m-%d')

def read_rul_data(filepath, default_spd=0, default_trq=0, default_pwr=0, default_eff=0):

    data_read = None
    # 檢查檔案是否存在
    if os.path.exists(filepath):
        if filepath.endswith('.parquet'):
            df_loaded = pd.read_parquet(filepath)
            tt=np.array(df_loaded["Voltage alpha"].iloc[0])
            data_read = {
                "Unix Time": df_loaded["Unix Time"].iloc[0],
                "Speed": [df_loaded["Speed"].iloc[0]],
                "Torque": [df_loaded["Torque"].iloc[0]],
                "Power": [df_loaded["Power"].iloc[0]],
                "Efficiency": [df_loaded["Efficiency"].iloc[0]],
                "vibration rms": [df_loaded["vibration rms"].iloc[0]] if "vibration rms" in df_loaded else [],
                "Voltage alpha": np.array([df_loaded["Voltage alpha"].iloc[0]]).T,
                "Voltage beta": np.array([df_loaded["Voltage beta"].iloc[0]]).T,
                "Current alpha": np.array([df_loaded["Current alpha"].iloc[0]]).T,  # 轉為 List
                "Current beta": np.array([df_loaded["Current beta"].iloc[0]]).T,
                "vibration data": np.array([df_loaded["vibration data"].iloc[0]]).T if "vibration rms" in df_loaded else [],
            }
        elif filepath.endswith('.csv'):
            # csv read code version
            # read time stamp from first line
            with open(filepath, "r") as file:
                first_line = file.readline().strip()  # 讀取第一行並去掉換行符
            unix_time = first_line.split(",")[1]  # 取第二個欄位 (1736773960)

            # read rest of the data
            # df_loaded=pd.read_csv(filepath, skiprows=1, names=["V_alpha", "V_beta", "I_alpha", "I_beta"])
            df_loaded = pd.read_csv(filepath, skiprows=1)

            data_read = {
                "Unix Time": unix_time,
                "Speed":    [default_spd],
                "Torque":   [default_trq],
                "Power":    [default_pwr],
                "Efficiency": [default_eff],
                "Voltage alpha": df_loaded["V_alpha"].to_numpy(),
                "Voltage beta": df_loaded["V_beta"].to_numpy(),
                "Current alpha":df_loaded["I_alpha"].to_numpy(),
                "Current beta": df_loaded["I_beta"].to_numpy(),
                "vibration rms": [0],
            }
        else:
            print(f"Unsupported file format: {filepath}")
            return data_read


    else:
        print(f"檔案 {filepath} 不存在，請確認檔案路徑。")
    return data_read

def load_range(folder_path, start, end):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".parquet")])  # 取得所有 parquet 檔案並排序
    files = files[start-1:end]  # 取出 start 到 end 範圍內的檔案
    
    data_list = []  # 存放讀取的資料

    for file in files:
        file_path = os.path.join(folder_path, file)  # 取得完整路徑
        df = read_rul_data(file_path)  
        
        mean = np.mean(df["Current alpha"])

        # standard = 1
        standard = np.std(df["Current alpha"])
        # standard = np.max(np.abs(df["Current alpha"]))

        data_list.append((df["Current alpha"] - mean)/ standard)  # 加入清單
    
    return data_list

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

def CCAE_model(Normal_motor_data, Abnormal_motor_data, model_name, label, figure_name):

    # 預防模型名稱未打.keras
    if '.' not in model_name:
        model_name += '.keras'
    loaded_model = load_model(model_name)

    # Normal
    # 將資料做帶重疊的切片
    all_Data = []
    for i in range(len(Normal_motor_data)):
        # 將資料做帶重疊的切片
        all_Data.append(data_augmentation(Normal_motor_data[i], time_steps=1024, window_size=10, cols=[0], random_seed=42))
    Normal_final_data =  np.concatenate(all_Data)

    # 製作標籤
    labels_Normal = np.full(Normal_final_data.shape[0], label)
    # 模型預測 (samples, signal_length, num_features)
    reconstructed_Normal_data = loaded_model.predict([Normal_final_data, labels_Normal],batch_size=32) 
    # 將預測後之為度改為(samples, signal_length)，移除所有大小为為 1 的维度
    reconstructed_Normal_data_squeezed = np.squeeze(reconstructed_Normal_data)
    Normal_final_data = np.squeeze(Normal_final_data)  # 變成 (98,1024)
    # Calculate MSE
    Normal_data_mse_errors = np.mean(np.square(Normal_final_data - reconstructed_Normal_data_squeezed), axis=1)
    
    # Abnormal
    all_Data = []
    for i in range(len(Abnormal_motor_data)):
        # 將資料做帶重疊的切片
        all_Data.append(data_augmentation(Abnormal_motor_data[i], time_steps=1024, window_size=10, cols=[0], random_seed=42))
    Abnormal_final_data =  np.concatenate(all_Data)
    
    labels_Abnormal = np.full(Abnormal_final_data.shape[0], label)
    reconstructed_Abnormal_data = loaded_model.predict([Abnormal_final_data, labels_Abnormal],batch_size=32)
    reconstructed_Abnormal_data_squeezed = np.squeeze(reconstructed_Abnormal_data)
    Abnormal_final_data = np.squeeze(Abnormal_final_data)  # 變成 (98,1024)
    Abnormal_data_mse_errors = np.mean(np.square(Abnormal_final_data - reconstructed_Abnormal_data_squeezed), axis=1)


    # 計算誤差相似度
    BD = Bhattacharyya_Distance(Normal_data_mse_errors, Abnormal_data_mse_errors)
    # # 將浮點數格式化為保留三位小數
    # BD = f"{BD:.3f}"
    # # 替換小數點為 'P'
    # BD = BD.replace('.', 'P')

    # # 繪製正常和異常樣本的MSE誤差分布
    # plt.figure(figsize=(10, 6))
    # plt.hist(Normal_data_mse_errors, bins=20, alpha=0.7)
    # plt.hist(Abnormal_data_mse_errors, bins=20, alpha=0.7)
    # plt.xlabel('MSE Error', fontsize=20)
    # plt.tick_params(axis='x', labelsize=20)
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    # plt.ylabel('Number of Samples', fontsize=20)
    # plt.tick_params(axis='y', labelsize=20)
    # plt.savefig(figure_name+"_BD"+str(BD))
    # plt.close()

    # Save_CSV(Normal_data_mse_errors, Abnormal_data_mse_errors)

    return BD

def get_CCAE_MSE(raw_data, model_name, label, window_size=10):

    # 預防模型名稱未打.keras
    if '.' not in model_name:
        model_name += '.keras'
    loaded_model = load_model(model_name)

    # Normal
    # 將資料做帶重疊的切片
    all_Data = []
    for i in range(len(raw_data)):
        # 將資料做帶重疊的切片
        all_Data.append(data_augmentation(raw_data[i], time_steps=1024, window_size=window_size, cols=[0], random_seed=42))
    Normal_final_data =  np.concatenate(all_Data)

    # 製作標籤
    labels_Normal = np.full(Normal_final_data.shape[0], label)
    # 模型預測 (samples, signal_length, num_features)
    reconstructed_Normal_data = loaded_model.predict([Normal_final_data, labels_Normal],batch_size=32) 
    # 將預測後之為度改為(samples, signal_length)，移除所有大小为為 1 的维度
    reconstructed_Normal_data_squeezed = np.squeeze(reconstructed_Normal_data)
    Normal_final_data = np.squeeze(Normal_final_data)  # 變成 (98,1024)
    # Calculate MSE
    Normal_data_mse_errors = np.mean(np.square(Normal_final_data - reconstructed_Normal_data_squeezed), axis=1)
    
    return Normal_data_mse_errors


def get_CCAE_MSE_BD(ref_mse, target_mse, segment_perfile, file_batch=10, plot=False):
    
    segment_per_mse=segment_perfile*file_batch
    
    # 取ref_mse第一個片段作為參考樣本分布
    ref_mse= ref_mse[0:segment_per_mse]

    # 分段目標樣本，每段樣本數量與參考樣本相同
    target_mse_segments = []
    for i in range(0, len(target_mse) // segment_per_mse * segment_per_mse, segment_per_mse):
        target_segment = target_mse[i:i + segment_per_mse]
        target_mse_segments.append(target_segment)
    
    BD_all=[]
    # 計算每目標樣本每段 MSE 與參考樣本之 BD
    for i in range(len(target_mse_segments)):
        BD = Bhattacharyya_Distance(ref_mse, target_mse_segments[i])
        BD_all.append(BD)
    np.array(BD_all)

    return BD_all, target_mse_segments

def CCAE_train(Motor_data,model_name):
    
    all_Data = []
    for i in range(len(Motor_data)):
        # 將資料做帶重疊的切片
        all_Data.append(data_augmentation(Motor_data[i], time_steps=1024, window_size=10, cols=[0], random_seed=42))

    Data =  np.concatenate(all_Data)

    # 製作標籤
    Label = np.full(Data.shape[0], 1)

    # 先分割出80%的訓練數據和20%的驗證
    test_data_size = len(Label) * 2 // 10 # 測試數據是20%，額外算以確保各個label數量相同
    train_data, val_data, train_labels, val_labels = train_test_split(
        Data, Label, test_size=test_data_size, random_state=38, shuffle=True, stratify=Label)
    
    # 時間序列和條件數據的輸入
    time_series_input = Input(shape=(1024, 1), name='series') 
    condition_input = Input(shape=(1,), name='condition')        
    condition_layer_repeated = RepeatVector(1024)(condition_input)
    merged_encoder_input = concatenate([time_series_input, condition_layer_repeated]) 

    # encoded
    encoded_start = Conv1D(filters=64, kernel_size=64, strides=16, padding='same')(merged_encoder_input) 
    x = MaxPooling1D(pool_size=2, strides=2)(encoded_start)
    x = Activation('relu')(x)

    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=16, kernel_size=3, strides=1, padding='same')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    encoded = Activation('relu')(x)

    encoder_model = Model(inputs=[time_series_input, condition_input], outputs = encoded)

    # decoded
    decoder_input = Input(shape=(encoder_model.output_shape[1], encoder_model.output_shape[2]))
    decoder_condition_input_new = Input(shape=(1,), name='decoder_condition') 
    decoder_condition_input_begin = RepeatVector(encoder_model.output_shape[1])(decoder_condition_input_new)
    merged_decoder_input = concatenate([decoder_input, decoder_condition_input_begin])

    x = Conv1DTranspose(filters=16, kernel_size=3, strides=1, padding='same')(merged_decoder_input)
    x = UpSampling1D(size=2)(x)
    x = Activation('relu')(x)

    x = Conv1DTranspose(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = UpSampling1D(size=2)(x)
    x = Activation('relu')(x)

    x = Conv1DTranspose(filters=64, kernel_size=64, strides=16, padding='same')(x)
    x = UpSampling1D(size=2)(x)
    x = Activation('tanh')(x)


    decoded = Dense(1,activation='linear')(x)
    decoder_model = Model(inputs=[decoder_input, decoder_condition_input_new], outputs=decoded)

    # Full Model
    encoder_outputs = encoder_model([time_series_input, condition_input])
    decoder_outputs = decoder_model([encoder_outputs, condition_input])
 
    model = Model(inputs=[time_series_input, condition_input], outputs=decoder_outputs)
    model.compile(optimizer='Adam', loss='mse')
    # 輸出模型結構
    model.summary()
    
    def plot_model_architecture(model, file_name):     
        plot_model(model, to_file=file_name, show_shapes=True, show_layer_names=True, rankdir='TB')

    # 做模型訓練
    history = model.fit([train_data, train_labels], train_data, 
                epochs= 20,
                batch_size=10,
                validation_data=([val_data, val_labels], val_data))
    
    # 確認檔名包含.keras
    if '.' not in model_name:
        model_name += '.keras'
    model.save(model_name)
    
def Save_CSV(Normal_data_mse_errors, Abnormal_data_mse_errors):
    # 將資料寫入excel
    wb = Workbook()

    # 繪製正常和異常樣本的MSE誤差分布
    ws = wb.active
    ws.title = "data1"  # 設置工作表名稱
    ws.append(['normal', '', '', '', 'abnormal', '', ''])
    ws.append(['count','left','right','','count','left','right'])

    n_normal, bins_normal, patches_normal = plt.hist(Normal_data_mse_errors, bins=20, alpha=0.7)
    n_abnormal, bins_abnormal, patches_abnormal = plt.hist(Abnormal_data_mse_errors, bins=20, alpha=0.7)
    for j in range(len(n_normal)):
        new_raw = []
        new_raw.append(n_normal[j])
        new_raw.append(bins_normal[j])
        new_raw.append(bins_normal[j+1])
        new_raw.append('')
        new_raw.append(n_abnormal[j])
        new_raw.append(bins_abnormal[j])
        new_raw.append(bins_abnormal[j+1])
        ws.append(new_raw)

    # 儲存 Excel 文件
    file_path = 'output.xlsx'
    success_flag = 0
    while success_flag == 0:
        try:
            wb.save(file_path)
            success_flag = 1
        except IOError as e:
            print(f"存檔案時發生錯誤: {e}")
            input("按Enter繼續")
    print("資料已存入"+file_path)

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
    plt.show()  # Display the figure

def write_csv(data1, data2):
    csv_output = [['RUL_2', 'RUL_5']] + list(zip(data1, data2))

    # 指定檔案名稱
    file_name = 'all_data_BD.csv'

    # 開啟檔案寫入模式
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 寫入所有數據
        writer.writerows(csv_output)