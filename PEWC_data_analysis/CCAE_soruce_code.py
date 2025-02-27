from series_make import data_augmentation
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Conv1D, Dense, concatenate, RepeatVector, MaxPooling1D, Activation ,UpSampling1D, Conv1DTranspose
# from tensorflow.keras.utils import plot_model

# from keras.src.models.model import Model
# from keras.src.saving.saving_api import load_model

from keras.src.models import Model
from keras.src.saving.saving_api import load_model
from keras.src.layers import Input, Conv1D, Dense, concatenate, RepeatVector, MaxPooling1D, Activation ,UpSampling1D, Conv1DTranspose
from keras.src.utils import plot_model

import numpy as np 
import pandas as pd
from openpyxl import Workbook
import matplotlib.pyplot as plt

import time #for time consumption check

def main_CCAE():
    # data_dir_path  = 'D:\test\軸承資料'  #data_path
    #load data
   
    motor_600 = pd.read_csv('normal_600_All_Load_raw.csv', header=None)
    motor_1800 = pd.read_csv('normal_1800_All_Load_raw.csv', header=None)
    motor_3000 = pd.read_csv('normal_3000_All_Load_raw.csv', header=None)
    
    motor_Abnormal_600 = pd.read_csv('abnormal_600_All_Load_raw.csv', header=None)
    motor_Abnormal_1800 = pd.read_csv('abnormal_600_All_Load_raw.csv', header=None)
    motor_Abnormal_3000 = pd.read_csv('abnormal_600_All_Load_raw.csv', header=None)
 
    #sliding window
    #####
    
    Normal_600_L00_final_data = data_augmentation(motor_600, time_steps=2048, window_size=10, cols=[0], random_seed=42)
    Normal_600_L50_final_data = data_augmentation(motor_600, time_steps=2048, window_size=10, cols=[1], random_seed=42)
    Normal_600_L100_final_data = data_augmentation(motor_600, time_steps=2048, window_size=10, cols=[2], random_seed=42)

    Normal_1800_L00_final_data = data_augmentation(motor_1800, time_steps=2048, window_size=10, cols=[0], random_seed=42)
    Normal_1800_L50_final_data = data_augmentation(motor_1800, time_steps=2048, window_size=10, cols=[1], random_seed=42)
    Normal_1800_L100_final_data = data_augmentation(motor_1800, time_steps=2048, window_size=10, cols=[2], random_seed=42)

    Normal_3000_L00_final_data = data_augmentation(motor_3000, time_steps=2048, window_size=10, cols=[0], random_seed=42)
    Normal_3000_L50_final_data = data_augmentation(motor_3000, time_steps=2048, window_size=10, cols=[1], random_seed=42)
    Normal_3000_L100_final_data = data_augmentation(motor_3000, time_steps=2048, window_size=10, cols=[2], random_seed=42)
  
    #####
    Abnormal_600_L00_final_data = data_augmentation(motor_Abnormal_600, time_steps=2048, window_size=10, cols=[0], random_seed=42)
    Abnormal_600_L50_final_data = data_augmentation(motor_Abnormal_600, time_steps=2048, window_size=10, cols=[1], random_seed=42)
    Abnormal_600_L100_final_data = data_augmentation(motor_Abnormal_600, time_steps=2048, window_size=10, cols=[2], random_seed=42)
    
    Abnormal_1800_L00_final_data = data_augmentation(motor_Abnormal_1800, time_steps=2048, window_size=10, cols=[0], random_seed=42)
    Abnormal_1800_L50_final_data = data_augmentation(motor_Abnormal_1800, time_steps=2048, window_size=10, cols=[1], random_seed=42)
    Abnormal_1800_L100_final_data = data_augmentation(motor_Abnormal_1800, time_steps=2048, window_size=10, cols=[2], random_seed=42)

    Abnormal_3000_L00_final_data = data_augmentation(motor_Abnormal_3000, time_steps=2048, window_size=10, cols=[0], random_seed=42)
    Abnormal_3000_L50_final_data = data_augmentation(motor_Abnormal_3000, time_steps=2048, window_size=10, cols=[1], random_seed=42)
    Abnormal_3000_L100_final_data = data_augmentation(motor_Abnormal_3000, time_steps=2048, window_size=10, cols=[2], random_seed=42)
    
 

    #label
    #####
    labels_Normal_600_L00 = np.full(Normal_600_L00_final_data.shape[0], 1)
    labels_Normal_600_L50 = np.full(Normal_600_L50_final_data.shape[0], 2)
    labels_Normal_600_L100 = np.full(Normal_600_L100_final_data.shape[0], 3)

    labels_Normal_1800_L00 = np.full(Normal_1800_L00_final_data.shape[0], 4)
    labels_Normal_1800_L50 = np.full(Normal_1800_L50_final_data.shape[0], 5)
    labels_Normal_1800_L100 = np.full(Normal_1800_L100_final_data.shape[0], 6)

    labels_Normal_3000_L00 = np.full(Normal_3000_L00_final_data.shape[0], 7)
    labels_Normal_3000_L50 = np.full(Normal_3000_L50_final_data.shape[0], 8)
    labels_Normal_3000_L100 = np.full(Normal_3000_L100_final_data.shape[0], 9)

    #####
    labels_Abnormal_600_L00 = np.full(Abnormal_600_L00_final_data.shape[0], 1)
    labels_Abnormal_600_L50 = np.full(Abnormal_600_L50_final_data.shape[0], 2)
    labels_Abnormal_600_L100 = np.full(Abnormal_600_L100_final_data.shape[0], 3)

    labels_Abnormal_1800_L00 = np.full(Abnormal_1800_L00_final_data.shape[0], 4)
    labels_Abnormal_1800_L50 = np.full(Abnormal_1800_L50_final_data.shape[0], 5)
    labels_Abnormal_1800_L100 = np.full(Abnormal_1800_L100_final_data.shape[0], 6)
   
    labels_Abnormal_3000_L00 = np.full(Abnormal_3000_L00_final_data.shape[0], 7)
    labels_Abnormal_3000_L50 = np.full(Abnormal_3000_L50_final_data.shape[0], 8)
    labels_Abnormal_3000_L100 = np.full(Abnormal_3000_L100_final_data.shape[0], 9)
  
   
    all_data = np.concatenate([Normal_600_L00_final_data, Normal_600_L50_final_data, Normal_600_L100_final_data,
                               Normal_1800_L00_final_data, Normal_1800_L50_final_data, Normal_1800_L100_final_data, 
                               Normal_3000_L00_final_data, Normal_3000_L50_final_data, Normal_3000_L100_final_data], axis=0) 
    all_labels = np.concatenate([labels_Normal_600_L00, labels_Normal_600_L50, labels_Normal_600_L100,
                                 labels_Normal_1800_L00, labels_Normal_1800_L50, labels_Normal_1800_L100, 
                                 labels_Normal_3000_L00, labels_Normal_3000_L50, labels_Normal_3000_L100 ], axis=0)

    # 先分割出80%的訓練數據和20%的驗證
    train_data, val_data, train_labels, val_labels = train_test_split(
        all_data, all_labels, test_size=6831, random_state=38, shuffle=True, stratify=all_labels)

    # 時間序列和條件數據的輸入
    time_series_input = Input(shape=(2048, 1), name='series') 
    condition_input = Input(shape=(1,), name='condition')        
    condition_layer_repeated = RepeatVector(2048)(condition_input)
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



    # should_train = False
    start_time=time.time()
    should_train = True
    if should_train:  
        history = model.fit([train_data, train_labels], train_data, 
                  epochs= 20,
                  batch_size=10,
                #   callbacks=stop_at_threshold, 
                  validation_data=([val_data, val_labels], val_data))
        model.save("model_20.keras")
        # plot_model_architecture(model,file_name='model_name.png')
        # Save the encoder model
        encoder_model.save("encoder_model_20.keras")
        # plot_model_architecture(encoder_model, file_name='encoder_model_name.png')
        # Save the decoder model
        decoder_model.save("decoder_model_20.keras")
        # plot_model_architecture(decoder_model, file_name='decoder_model_name.png')

    end_time = time.time()
    print(f"程式執行時間：{(end_time-start_time):.6f} 秒")
        # Plot the training and validation loss
        # plt.figure(figsize=(12, 6))
        # plt.plot(history.history['loss'], label='Training Loss')
        # plt.plot(history.history['val_loss'], label='Validation Loss')
        # plt.title('Training and Validation Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

    loaded_model = load_model("model_20.keras")
    #正常訓練
    reconstructed_train_data = loaded_model.predict([train_data, train_labels])  # 使用模型進行預測
    #計算每個樣本的MSE
    reconstructed_train_data_squeezed = np.squeeze(reconstructed_train_data)
    train_data_mse_errors = np.mean(np.square(train_data - reconstructed_train_data_squeezed), axis=1)



    #模型預測 (samples, signal_length, num_features)
    reconstructed_Normal_600_L00_data = loaded_model.predict([Normal_600_L00_final_data, labels_Normal_600_L00]) #1
    reconstructed_Normal_600_L50_data = loaded_model.predict([Normal_600_L50_final_data, labels_Normal_600_L50]) #2
    reconstructed_Normal_600_L100_data = loaded_model.predict([Normal_600_L100_final_data, labels_Normal_600_L100]) #3

    reconstructed_Normal_1800_L00_data = loaded_model.predict([Normal_1800_L00_final_data, labels_Normal_1800_L00]) #4
    reconstructed_Normal_1800_L50_data = loaded_model.predict([Normal_1800_L50_final_data, labels_Normal_1800_L50]) #5
    reconstructed_Normal_1800_L100_data = loaded_model.predict([Normal_1800_L100_final_data, labels_Normal_1800_L100]) #6

    reconstructed_Normal_3000_L00_data = loaded_model.predict([Normal_3000_L00_final_data, labels_Normal_3000_L00]) #7
    reconstructed_Normal_3000_L50_data = loaded_model.predict([Normal_3000_L50_final_data, labels_Normal_3000_L50]) #8
    reconstructed_Normal_3000_L100_data = loaded_model.predict([Normal_3000_L100_final_data, labels_Normal_3000_L100]) #9

    #將預測後之為度改為(samples, signal_length)
    reconstructed_Normal_600_L00_data_squeezed = np.squeeze(reconstructed_Normal_600_L00_data)
    reconstructed_Normal_600_L50_data_squeezed = np.squeeze(reconstructed_Normal_600_L50_data)
    reconstructed_Normal_600_L100_data_squeezed = np.squeeze(reconstructed_Normal_600_L100_data)

    reconstructed_Normal_1800_L00_data_squeezed = np.squeeze(reconstructed_Normal_1800_L00_data)
    reconstructed_Normal_1800_L50_data_squeezed = np.squeeze(reconstructed_Normal_1800_L50_data)
    reconstructed_Normal_1800_L100_data_squeezed = np.squeeze(reconstructed_Normal_1800_L100_data)

    reconstructed_Normal_3000_L00_data_squeezed = np.squeeze(reconstructed_Normal_3000_L00_data)
    reconstructed_Normal_3000_L50_data_squeezed = np.squeeze(reconstructed_Normal_3000_L50_data)
    reconstructed_Normal_3000_L100_data_squeezed = np.squeeze(reconstructed_Normal_3000_L100_data)


    #Calculate MSE
    Normal_600_L00_data_mse_errors = np.mean(np.square(Normal_600_L00_final_data - reconstructed_Normal_600_L00_data_squeezed), axis=1)
    Normal_600_L50_data_mse_errors = np.mean(np.square(Normal_600_L50_final_data - reconstructed_Normal_600_L50_data_squeezed), axis=1)
    Normal_600_L100_data_mse_errors = np.mean(np.square(Normal_600_L100_final_data - reconstructed_Normal_600_L100_data_squeezed), axis=1)

    Normal_1800_L00_data_mse_errors = np.mean(np.square(Normal_1800_L00_final_data - reconstructed_Normal_1800_L00_data_squeezed), axis=1)
    Normal_1800_L50_data_mse_errors = np.mean(np.square(Normal_1800_L50_final_data - reconstructed_Normal_1800_L50_data_squeezed), axis=1)
    Normal_1800_L100_data_mse_errors = np.mean(np.square(Normal_1800_L100_final_data - reconstructed_Normal_1800_L100_data_squeezed), axis=1)

    Normal_3000_L00_data_mse_errors = np.mean(np.square(Normal_3000_L00_final_data - reconstructed_Normal_3000_L00_data_squeezed), axis=1)
    Normal_3000_L50_data_mse_errors = np.mean(np.square(Normal_3000_L50_final_data - reconstructed_Normal_3000_L50_data_squeezed), axis=1)
    Normal_3000_L100_data_mse_errors = np.mean(np.square(Normal_3000_L100_final_data - reconstructed_Normal_3000_L100_data_squeezed), axis=1)

    #Abnormal
    reconstructed_Abnormal_600_L00_data = loaded_model.predict([Abnormal_600_L00_final_data, labels_Abnormal_600_L00])
    reconstructed_Abnormal_600_L50_data = loaded_model.predict([Abnormal_600_L50_final_data, labels_Abnormal_600_L50])
    reconstructed_Abnormal_600_L100_data = loaded_model.predict([Abnormal_600_L100_final_data, labels_Abnormal_600_L100])

    reconstructed_Abnormal_1800_L00_data = loaded_model.predict([Abnormal_1800_L00_final_data, labels_Abnormal_1800_L00])
    reconstructed_Abnormal_1800_L50_data = loaded_model.predict([Abnormal_1800_L50_final_data, labels_Abnormal_1800_L50])
    reconstructed_Abnormal_1800_L100_data = loaded_model.predict([Abnormal_1800_L100_final_data, labels_Abnormal_1800_L100])

    reconstructed_Abnormal_3000_L00_data = loaded_model.predict([Abnormal_3000_L00_final_data, labels_Abnormal_3000_L00])
    reconstructed_Abnormal_3000_L50_data = loaded_model.predict([Abnormal_3000_L50_final_data, labels_Abnormal_3000_L50])
    reconstructed_Abnormal_3000_L100_data = loaded_model.predict([Abnormal_3000_L100_final_data, labels_Abnormal_3000_L100])

    reconstructed_Abnormal_600_L00_data_squeezed = np.squeeze(reconstructed_Abnormal_600_L00_data)
    reconstructed_Abnormal_600_L50_data_squeezed = np.squeeze(reconstructed_Abnormal_600_L50_data)
    reconstructed_Abnormal_600_L100_data_squeezed = np.squeeze(reconstructed_Abnormal_600_L100_data)

    reconstructed_Abnormal_1800_L00_data_squeezed = np.squeeze(reconstructed_Abnormal_1800_L00_data)
    reconstructed_Abnormal_1800_L50_data_squeezed = np.squeeze(reconstructed_Abnormal_1800_L50_data)
    reconstructed_Abnormal_1800_L100_data_squeezed = np.squeeze(reconstructed_Abnormal_1800_L100_data)

    reconstructed_Abnormal_3000_L00_data_squeezed = np.squeeze(reconstructed_Abnormal_3000_L00_data)
    reconstructed_Abnormal_3000_L50_data_squeezed = np.squeeze(reconstructed_Abnormal_3000_L50_data)
    reconstructed_Abnormal_3000_L100_data_squeezed = np.squeeze(reconstructed_Abnormal_3000_L100_data)

    Abnormal_600_L00_data_mse_errors = np.mean(np.square(Abnormal_600_L00_final_data - reconstructed_Abnormal_600_L00_data_squeezed), axis=1)
    Abnormal_600_L50_data_mse_errors = np.mean(np.square(Abnormal_600_L50_final_data - reconstructed_Abnormal_600_L50_data_squeezed), axis=1)
    Abnormal_600_L100_data_mse_errors = np.mean(np.square(Abnormal_600_L100_final_data - reconstructed_Abnormal_600_L100_data_squeezed), axis=1)

    Abnormal_1800_L00_data_mse_errors = np.mean(np.square(Abnormal_1800_L00_final_data - reconstructed_Abnormal_1800_L00_data_squeezed), axis=1)
    Abnormal_1800_L50_data_mse_errors = np.mean(np.square(Abnormal_1800_L50_final_data - reconstructed_Abnormal_1800_L50_data_squeezed), axis=1)
    Abnormal_1800_L100_data_mse_errors = np.mean(np.square(Abnormal_1800_L100_final_data - reconstructed_Abnormal_1800_L100_data_squeezed), axis=1)

    Abnormal_3000_L00_data_mse_errors = np.mean(np.square(Abnormal_3000_L00_final_data - reconstructed_Abnormal_3000_L00_data_squeezed), axis=1)
    Abnormal_3000_L50_data_mse_errors = np.mean(np.square(Abnormal_3000_L50_final_data - reconstructed_Abnormal_3000_L50_data_squeezed), axis=1)
    Abnormal_3000_L100_data_mse_errors = np.mean(np.square(Abnormal_3000_L100_final_data - reconstructed_Abnormal_3000_L100_data_squeezed), axis=1)

    # 選取訓練樣本裡的對應標籤已用來比對測試樣本之分布
    normal_600_L00_indices = np.where(train_labels == 1)[0]
    normal_600_L00_mse_errors = train_data_mse_errors[normal_600_L00_indices]

    normal_600_L50_indices = np.where(train_labels == 2)[0]
    normal_600_L50_mse_errors = train_data_mse_errors[normal_600_L50_indices]

    normal_600_L100_indices = np.where(train_labels == 3)[0]
    normal_600_L100_mse_errors = train_data_mse_errors[normal_600_L100_indices]

    normal_1800_L00_indices = np.where(train_labels == 4)[0]
    normal_1800_L00_mse_errors = train_data_mse_errors[normal_1800_L00_indices]

    normal_1800_L50_indices = np.where(train_labels == 5)[0]
    normal_1800_L50_mse_errors = train_data_mse_errors[normal_1800_L50_indices]

    normal_1800_L100_indices = np.where(train_labels == 6)[0]
    normal_1800_L100_mse_errors = train_data_mse_errors[normal_1800_L100_indices]

    normal_3000_L00_indices = np.where(train_labels == 7)[0]
    normal_3000_L00_mse_errors = train_data_mse_errors[normal_3000_L00_indices]

    normal_3000_L50_indices = np.where(train_labels == 8)[0]
    normal_3000_L50_mse_errors = train_data_mse_errors[normal_3000_L50_indices]

    normal_3000_L100_indices = np.where(train_labels == 9)[0]
    normal_3000_L100_mse_errors = train_data_mse_errors[normal_3000_L100_indices]

    normal_mse_errors = [normal_600_L00_mse_errors, normal_600_L50_mse_errors, normal_600_L100_mse_errors,
                         normal_1800_L00_mse_errors, normal_1800_L50_mse_errors, normal_1800_L100_mse_errors,
                         normal_3000_L00_mse_errors, normal_3000_L50_mse_errors, normal_3000_L100_mse_errors]

    Abnormal_mse_errors = [Abnormal_600_L00_data_mse_errors, Abnormal_600_L50_data_mse_errors, Abnormal_600_L100_data_mse_errors,
                         Abnormal_1800_L00_data_mse_errors, Abnormal_1800_L50_data_mse_errors, Abnormal_1800_L100_data_mse_errors,
                         Abnormal_3000_L00_data_mse_errors, Abnormal_3000_L50_data_mse_errors, Abnormal_3000_L100_data_mse_errors]
    png_name = ["600_00", "600_50", "600_100",
                "1800_00", "1800_50", "1800_100",
                "3000_00", "3000_50", "3000_100",]


        
    # 將資料寫入excel
    wb = Workbook()
    '''
    '''
    # 繪製正常和異常樣本的MSE誤差分布
    for i in range(9):
        if i == 0:
            ws = wb.active
            ws.title = "data1"  # 設置工作表名稱
        else :
            ws = wb.create_sheet(title="data"+str(i+1))
        # ws1.title = "data"+str(i)  # 設置工作表名稱
        ws.append(['normal', '', '', '', 'abnormal', '', ''])
        ws.append(['count','left','right','','count','left','right'])

        plt.figure(i+1)
        n_normal, bins_normal, patches_normal = plt.hist(normal_mse_errors[i], bins=20, alpha=0.7)
        n_abnormal, bins_abnormal, patches_abnormal = plt.hist(Abnormal_mse_errors[i], bins=20, alpha=0.7)
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

        plt.xlabel('MSE Error', fontsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.ylabel('Number of Samples', fontsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.savefig(png_name[i])
    plt.show()
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
    plt.show()

if __name__ == '__main__':    
    main_CCAE()