import CCAE_function_1024_point
import os
import keras.src.models as models
from keras.src.saving.saving_api import load_model
from CCAE_function_1024_point import plot_hlpler

# 設定當前工作目錄為腳本所在的目錄
# 這樣可以確保相對路徑正確
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 設定要讀取的資料夾
motor_folder = '../../PEWC dataset/PEWC_raw_data/0310to0318PEWC/RUL_2'
motor_abnormal_folder = '../../PEWC dataset/PEWC_raw_data/0310to0318PEWC/RUL_5'

# 讀取資料夾中start到end的檔案
motor_data_list = CCAE_function_1024_point.load_range(motor_folder, start=1, end=60) 
motor_abnormal_data_list = CCAE_function_1024_point.load_range(motor_abnormal_folder, start=61, end=120)

Is_predict = 1 # 決定是預測還是訓練的旗標

model_name = "model_pei.keras"
figure_name = "figure_1"

# 預防模型名稱未打.keras
if '.' not in model_name:
    model_name += '.keras'
loaded_model = load_model(model_name)

#%% 使用訓練好的模型做預測
if Is_predict == 1:
    # plot_hlpler.plot_CCAE_reconstrcut_signal(motor_data_list[0], loaded_model, label=1) # 畫出單一電流的重建圖
    # plot_hlpler.plot_CCAE_reconstrcut_signal(motor_abnormal_data_list[0], loaded_model, label=1) # 畫出單一電流的重建圖
    # plot_hlpler.plot_CCAE_reconstrcut_signal(motor_abnormal_data_list[1], loaded_model, label=1) # 畫出單一電流的重建圖

    CCAE_function_1024_point.CCAE_model(Normal_motor_data = motor_data_list , # 健康的資料，用來畫圖
                            Abnormal_motor_data = motor_abnormal_data_list , # 損壞的資料
                            model_name = model_name , # 模型
                            label = 1 , # 標籤編號
                            figure_name = figure_name) # 圖檔名稱
    # plot_hlpler.plot_CCAE_reconstrcut_signal(motor_data_list[0], loaded_model, label=1) # 畫出單一電流的重建圖
   
    
# 訓練新模型
if Is_predict == 0:
    CCAE_function_1024_point.CCAE_train(Motor_data = motor_data_list,
                            model_name = model_name)
    # CCAE_function_1024_point.CCAE_train_resnet(Motor_data = motor_data_list,
    #                         model_name = "model_res.keras")

# 印出其中各一筆電流
# CCAE_function_1024_point_all_data.plot_current(motor_data_list[1], motor_abnormal_data_list[1])
# %%
