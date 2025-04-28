import CCAE_function_1024_point_all_data
import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# 讀取資料夾中start到end的檔案
select_range = 10
data_range = 2276
# 1-2276
motor_data_list = CCAE_function_1024_point_all_data.load_range(motor_folder, start=1, end=select_range)  

Is_predict = 1 # 決定是還是訓練的旗標

model_name = "model.keras"
figure_name = "figure_1"

RUL_2 = []
RUL_5 = []
# 使用訓練好的模型做預測
if Is_predict == 1:
    for i in range (data_range // select_range):
        motor_abnormal_data_list = CCAE_function_1024_point_all_data.load_range(motor_folder, start=i*select_range+1, end=(i+1)*select_range)

        BD = CCAE_function_1024_point_all_data.CCAE_model(Normal_motor_data = motor_data_list , # 健康的資料，用來畫圖
                                Abnormal_motor_data = motor_abnormal_data_list , # 損壞的資料
                                model_name = model_name , # 模型
                                label = 1 , # 標籤編號
                                figure_name = figure_name) # 圖檔名稱
        RUL_2.append(BD)

    for i in range (data_range // select_range):
        motor_abnormal_data_list = CCAE_function_1024_point_all_data.load_range(motor_abnormal_folder, start=i*select_range+1, end=(i+1)*select_range)

        BD = CCAE_function_1024_point_all_data.CCAE_model(Normal_motor_data = motor_data_list , # 健康的資料，用來畫圖
                                Abnormal_motor_data = motor_abnormal_data_list , # 損壞的資料
                                model_name = model_name , # 模型
                                label = 1 , # 標籤編號
                                figure_name = figure_name) # 圖檔名稱
        RUL_5.append(BD)

CCAE_function_1024_point_all_data.write_csv(RUL_2,RUL_5)


# 訓練新模型
if Is_predict == 0:
    CCAE_function_1024_point_all_data.CCAE_train(Motor_data = motor_data_list,
                            model_name = model_name)

# CCAE_function_1024_point_all_data.plot_current(motor_data_list[1], motor_abnormal_data_list[1])