"""函式說明： 此函式為PEWC 資料RUL 預測分析的獨立運行版本
用途為獨立執行在本地端，並將結果儲存至指定資料夾中
輸入： 前五十筆觀測資料(MK窗大小) 輸出： RUL 預測結果
會動態存入歷史資料以協助預測"""


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
from datetime import datetime
import pandas as pd
import os
import sys
from IPython.display import HTML, display, IFrame
import pickle

# 將專案根目錄加入 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
 # 匯入自定義函式
from PEWC_analysis_helpler.rul_helplers import (
    time_data_preprocess as t_process,
    plot_helplers as plt_helper,  
    get_ema,
    get_smape_and_csmape,
    rolling_mk,
    estimate_rul_exp
)

# 匯入RUL分析所需的函式庫
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from IPython.display import HTML
from base64 import b64encode


MK_window = 50  # MK窗大小
def create_rul_prediction_animation(X_est, ema_data, y_hat, RUL_thres, filtered_time, RUL_start_idx, MK_window, gamma, fps=20):
    """
    Create and display an HTML animation of RUL prediction model evolution.
    Returns HTML animation, GIF file path and MP4 file path.
    """
    import matplotlib as mpl
    mpl.rcParams['animation.embed_limit'] = 50 * 1024 * 1024  
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_domain = np.linspace(0, 10000, 1000)

    line_model, = ax.plot([], [], label='Predicted Model', color='blue')
    line_model_pred, = ax.plot([], [], label='Model Prediction', color='skyblue', linestyle='--', alpha=0.5)
    ax.axhline(RUL_thres, color='black', linestyle='--', label='Threshold')
    x_fail = filtered_time[Fail_idx_MA] - filtered_time[RUL_start_idx - MK_window]
    ax.axvline(x_fail, color='black', linestyle='--', label='Threshold time ')
    point, = ax.plot([], [], 'ko', label='Intersection Point')
    intersec_line = ax.axvline(x=np.nan, color='skyblue', linestyle='--', alpha=0.7)
    line_hi_all, = ax.plot([], [], color='gray', alpha=0.3, label='Full HI History')
    line_hi_partial, = ax.plot([], [], color='green', linestyle='-', linewidth=2, label='Observed HI')
    line_hi_track, = ax.plot([], [], color='red', linestyle='--', linewidth=1, label='Tracked HI')
    
    ax.set_xlim(0, 10000)
    ax.set_ylim(RUL_thres-1, np.max(ema_data) * 1.05)
    ax.set_xlabel('Relative Time from estimation start[min]')
    ax.set_ylabel('HI')
    ax.grid(True)
    ax.legend(loc='upper right')

    # Calculate intersections
    intersections_x = []
    intersections_y = []
    for x_hat in X_est:
        phi0_i, alpha_i, beta_i = x_hat
        def f_eq(x): return phi0_i + alpha_i * x + beta_i / (x + gamma) - RUL_thres
        try:
            sol = fsolve(f_eq, x0=10000)[0]
            intersections_x.append(sol if sol > 0 else np.nan)
            intersections_y.append(RUL_thres if sol > 0 else np.nan)
        except:
            intersections_x.append(np.nan)
            intersections_y.append(np.nan)

    x_hi_relative = filtered_time - filtered_time[RUL_start_idx - MK_window]

    def update(frame):
        current_time = t[frame]  
        
        phi0_i, alpha_i, beta_i = X_est[frame]
        y_model = phi0_i + alpha_i * x_domain + beta_i / (x_domain + gamma)
        
        x_int = intersections_x[frame]
        y_int = intersections_y[frame]
        point.set_data([x_int], [y_int]) if not np.isnan(x_int) else point.set_data([], [])
        
        line_model.set_data(x_domain[ x_domain <= current_time], y_model[ x_domain <= current_time])
        line_model_pred.set_data(x_domain[(x_domain > current_time) & (x_domain < x_int)], y_model[(x_domain > current_time) & (x_domain < x_int)])
        line_hi_all.set_data(x_hi_relative, ema_data)
        valid_mask = x_hi_relative <= current_time
        line_hi_partial.set_data(x_hi_relative[valid_mask], ema_data[valid_mask])
        line_hi_track.set_data(t[1:frame], y_hat[1:frame])
        intersec_line.set_xdata([x_int])
        
        ax.set_title(f'Inspection time {t[frame ]:.0f} min| Intersection at x={x_int:.0f} min' if not np.isnan(x_int) else f'Model step {frame}')
        return line_model, point, line_hi_partial, line_hi_all

    # Create HTML animation
    ani_html = animation.FuncAnimation(fig, update, frames=len(X_est), blit=True, interval=1000//fps)
    html_animation = HTML(ani_html.to_jshtml())
    html_path = os.path.join("..", "animations", "rul_prediction_animation.html")
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, 'w') as f:
        f.write(html_animation.data)
    display(html_animation)  # Add this line to display in Jupyter

    # Create GIF animation
    writer_gif = PillowWriter(fps=fps)
    gif_path = os.path.join("..", "animations", "rul_prediction_animation.gif")
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    ani_gif = animation.FuncAnimation(fig, update, frames=len(X_est), blit=True)
    ani_gif.save(gif_path, writer=writer_gif)

    # Create MP4 animation
    mp4_path = os.path.join("..", "animations", "rul_prediction_animation.mp4")
    writer_mp4 = animation.FFMpegWriter(fps=fps, bitrate=2000)
    ani_mp4 = animation.FuncAnimation(fig, update, frames=len(X_est), blit=True)
    ani_mp4.save(mp4_path, writer=writer_mp4)

    plt.close(fig)
    return html_animation, gif_path, mp4_path

def display_mp4(file_path):
    """Display MP4 video in Jupyter notebook."""
    video_file = open(file_path, "rb").read()
    video_encoded = b64encode(video_file).decode('utf-8')
    video_tag = f'<video controls><source src="data:video/mp4;base64,{video_encoded}" type="video/mp4"></video>'
    return HTML(video_tag)

def get_datetime_from_unix(unix_time):
    return pd.to_datetime(unix_time, unit='s').strftime('%Y-%m-%d')

def model_func(x, phi0, alpha, beta, gamma):
    return phi0 + alpha * x + beta / (x + gamma)

def convert_unix_to_minutes(unix_str_list):
    """
    將Unix時間的字串list轉換成從起始點起算的分鐘數值nparray。

    Args:
        unix_str_list (list of str): 例如 ["1714891600", "1714891660", ...]

    Returns:
        np.ndarray: 每個時間點與首個時間點的差值（單位：分鐘）
    """
    # 將字串轉成 float 秒
    unix_seconds = np.array([float(t) for t in unix_str_list])

    # 減去起點時間並換算成分鐘
    delta_minutes = (unix_seconds) / 60.0

    return delta_minutes   

class DynamicRULDetector:
    def __init__(self,
                 mk_window_size: int = 50,
                 ema_coeff: float = 0.1,
                 filt_window: int = 10,
                 filt_n_std: float = 3.0,
                 mk_epsilon: float = 0.0,
                 start_thres_ratio: float = 0.95,
                 rls_beta: float = 0.99,
                 rls_p0: float = 5.0,
                 fail_thres_ratio: float = 0.82):
        """
        動態 RUL 啟動檢測器。

        Args:
            mk_window_size: MK 檢定用滑動窗大小（筆數）。
            ema_coeff: EMA 平滑係數 α。
            filt_window: 離群值濾波所用 local window 大小。
            filt_n_std: 濾波閾值 n_std 倍數。
            mk_epsilon: MK 檢定 epsilon 敏感度參數。
            start_thres_ratio: RUL 啟動門檻相對健康指標初值的比例（如 0.95）。
        """
        self.mk_w = mk_window_size
        self.ema_coeff = ema_coeff
        self.filt_w = filt_window
        self.n_std = filt_n_std
        self.mk_eps = mk_epsilon
        self.start_ratio = start_thres_ratio
        self.fail_ratio = fail_thres_ratio

        # 狀態變數
        self.raw_hist = np.zeros(mk_window_size)
        self.ema_hist = np.zeros(mk_window_size)
        self.ts_hist  = [None] * mk_window_size
        self.count = 0
        self.ema_last = None
        self.RUL_started = False
        self.RUL_start_stamp = ''
        self.RUL_start_idx = None
        self.initial_health = 0
        
        #初始化相關
        self.rul_enable = 0
        self.NLR_initial =[self.initial_health, -1, 1, 1] # NLR初始化參數
        self.NLR_bound = ([0, -np.inf, -np.inf, 0], [np.inf, 0, np.inf, np.inf]) # NLR邊界
        
        #預測相關
        self.beta = rls_beta # RLS預測係數
        self.rls_p0=rls_p0 * np.eye(3) # RLS初始P矩陣
        self.RUL_thres =0 # 失效門檻
        self.RLS_initial_params=None # RLS初始化參數
        self.rls_params_upd = None # RLS預測參數更新
        
        #預測結果 
        self.rul_pred=None
        self.rul_curve=[]
            
    
    @staticmethod
    def simple_plot(x1,x2, y1, y2, title='Plot', xlabel='X', ylabel='Y'):
        x1 = np.array(x1)
        x2 = np.array(x2)
        fig = plt.figure(figsize=(10, 6))
        plt.plot(x1, y1, label='Data1')
        plt.plot(x2, y2, label='Data2')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.show(block=False)  
        return fig
    
    def update(self, value: float, timestamp, is_test=False) -> int:
        """
        推入新的觀測值並檢測是否達成 RUL 啟動條件。

        Args:
            value: 當前健康指標（HI）觀測值。
            timestamp: 當前時間戳記，可為任意可比較物件（如 UNIX time）。

        Returns:
            RUL_start_idx: 若剛觸發啟動，回傳啟動時的全域觀測次數索引；否則回傳 None。
            RUL_curve_points: 若剛觸發啟動，回傳 RUL 到達終點前曲線的點
            RUL_intersection: 若剛觸發啟動，回傳 RUL時間點。
        """
        # 第一次呼叫時設定初始健康指標
        if self.count == 0:
            self.initial_health = value
            self.ema_last = value
            self.NLR_initial =[self.initial_health, -1, 1, 1] # NLR初始化參數
            self.RUL_thres=self.initial_health * self.fail_ratio   # 失效門檻

        self.count += 1
        
        # 更新時間戳記
        self.ts_hist = self.ts_hist[1:] + [timestamp]

        # 推入 raw 與 ema 串列
        # --- 滑動更新 raw_hist
        self.raw_hist = np.roll(self.raw_hist, -1)
        self.raw_hist[-1] = value

        # 計算離群值 local filter（以 raw_hist 的最後 filt_window 筆）
        if self.count < self.filt_w:
            # 若尚未達到 mk window 大小，尚不做檢定
            # 先填入原始值
            self.ema_hist = np.roll(self.ema_hist, -1)
            self.ema_hist[-1] = value
            return None
        
        local = self.raw_hist[-self.filt_w:]
        mean_local = np.mean(local)
        std_local  = np.std(local)
        if abs(value - mean_local) > self.n_std * std_local:
            filt_val = mean_local
        else:
            filt_val = value

        # 推入 ema_hist
        self.ema_last = self.ema_coeff * filt_val + (1 - self.ema_coeff) * self.ema_last
        self.ema_hist = np.roll(self.ema_hist, -1)
        self.ema_hist[-1] = self.ema_last

        # 若尚未達到 mk window 大小，尚不做檢定
        if self.count < self.mk_w:
            return None

        # 若尚未啟動，做 MK test + 門檻判定
        if not self.RUL_started:
            # MK檢定：最後一點的 Z value
            _, Z = rolling_mk(self.ema_hist, self.mk_w, self.mk_eps)
            z_last = Z[-1]
            # 啟動條件：|Z| > 3 且 EMA 下降到初始值 start_ratio 以下
            if abs(z_last) > 3 and self.ema_last < self.initial_health * self.start_ratio:
                # 啟動條件成立，記錄啟動時間戳記
                self.RUL_started = True
                self.RUL_start_stamp = self.ts_hist[0]
                
                # 用NLR初始化預測參數
                y_fit = self.ema_hist   
                x_start = convert_unix_to_minutes([self.ts_hist[0]])
                x_fit_relative =convert_unix_to_minutes(self.ts_hist)-x_start
            
                initial_guess =self.NLR_initial
                bounds = self.NLR_bound

                params, _ = curve_fit(model_func, x_fit_relative, y_fit, p0=initial_guess, bounds=bounds, maxfev=2000)
                phi0, alpha, beta, gamma = params
                self.RLS_initial_params = {
                    'phi0': phi0,
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma
                }
                
                self.rls_params_upd= self.RLS_initial_params.copy()
                
                print("Initial model parameters:")
                print(f"phi0 = {phi0:.4f}, alpha = {alpha:.4f}, beta = {beta:.4f}, gamma = {gamma:.4f}")
                # plt_helper.plot_curve_fitting(x_fit_relative, y_fit, model_func, params)
                
                # 記錄全域啟動位置 = 現在已處理總筆數 - mk_window_size
                self.RUL_start_idx = self.count - self.mk_w
                print(f"RUL 開始於全域觀測第 {self.RUL_start_idx} 筆 (資料點索引 i={self.count - 1})")
                return self.RUL_start_idx
            else :
                return None
            
        # RUL 啟動後，動態更新PRLS預測 
        # 如果 指標達到失效點則跳出
        if self.ema_last < self.RUL_thres:
            print(f"RUL 觸發於全域觀測第 {self.count} 筆 (資料點索引 i={self.count - 1})")
            return 0
        
        
        P0 = self.rls_p0.copy() # 取得當前P矩陣
        
        x_temp = np.array([self.rls_params_upd['phi0'], 
                              self.rls_params_upd['alpha'],
                              self.rls_params_upd['beta']])
        
        # 取得當前資料相對時間點
        t = convert_unix_to_minutes([self.RUL_start_stamp])[0]
        current_t = convert_unix_to_minutes([timestamp])[0] 
        t_relative = current_t - t
        

        # 建立迴歸矩陣 h  
        h = np.array([1, t_relative, 1/(t_relative + self.RLS_initial_params['gamma'])]).reshape(-1,1)

        # 當前觀察值
        y = self.ema_last

        # 預測當前值
        y_pred = np.dot(h.T, x_temp.reshape(-1,1))[0,0] 

        # 計算預測誤差
        e = y - y_pred

        # 更新增益矩陣K
        K = P0 @ h / (self.beta + h.T @ P0 @ h)

        # 更新參數向量 x
        x_temp = x_temp + (K.flatten() * e)

        # 更新協方差矩陣 P
        P0 = (np.eye(3) - K @ h.T) @ P0 / self.beta
        self.rls_p0= P0
        
        # 更新參數
        self.rls_params_upd['phi0'] = x_temp[0]
        self.rls_params_upd['alpha'] = x_temp[1]
        self.rls_params_upd['beta'] = x_temp[2]

        # 預測失效時間
        phi0_i, alpha_i, beta_i = x_temp
        def f_eq(x):
            return phi0_i + alpha_i * x + beta_i / (x + self.RLS_initial_params['gamma']) - self.RUL_thres
        try:
            sol = fsolve(f_eq, x0=10000)[0]
            if sol > 0:
                x_curve = np.linspace(t_relative, sol, 100)
                y_curve = phi0_i + alpha_i * x_curve + beta_i / (x_curve + self.RLS_initial_params['gamma'])

                if is_test:
                    #   測試模式印出並畫出結果
                    print(f'Elapsed time : {int(t_relative)} min ,phi0={x_temp[0]:.4f}, alpha={x_temp[0]:.4f},beta={beta_i:.4f}', end="")
                    print(f' e={e:.4f}, k={K[0,0]:.6f}, P0={P0[0,0]:.6f}')
                    # 取出RUL曲線點
                    t_window=convert_unix_to_minutes(self.ts_hist)-t
                    # Clear previous plot if it exists
                    plt.clf()
                    # Create new plot
                    # fig = self.simple_plot(t_window, x_curve, self.ema_hist, y_curve, title='RUL EMA', xlabel='Time [min]', ylabel='EMA HI')
                    
                    # Add additional plot elements for debugging
                    x1 = np.array(t_window)
                    x2 = np.array(x_curve)
                    plt.plot(x1, self.ema_hist, label='Data1')
                    plt.plot(x2, y_curve, label='Data2')
                    plt.grid(True)
                    plt.legend()
                    plt.xlim(0, 10000)
                    plt.axhline(y=self.RUL_thres, color='r', linestyle='--', label='RUL Threshold')
                    plt.text(0.02, 0.98, f'Time: {t_relative:.1f} min', transform=plt.gca().transAxes)
                    plt.text(0.02, 0.94, f'RUL: {sol-t_relative:.1f} min', transform=plt.gca().transAxes)
                    plt.legend()
                    plt.pause(0.05)
                    # Draw and pause briefly to show animation
                    plt.draw()
            
                # 回傳剩餘壽命時間(分鐘)
                # print(f"RUL 預測時間: {sol- t_relative:.2f} min")
                self.rul_pred = sol - t_relative
                # Create numpy array with both x and y curve data
                self.rul_curve = np.vstack((x_curve, y_curve))
                return sol - t_relative
        except:
            return None

        return None

if __name__ == "__main__":
   
    #%% 動態RUL檢測器初始化
    
    # Change working directory to script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    

    # 將專案根目錄加入 sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)


    # 顯示中文設定
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    #%% 讀取所有歷程資料與前處理

    device_number = 2
    record_info = '0208to0217PEWC'
    output_file_parquet = os.path.join("..", "time_list_extraction", f"RUL_{device_number}",
                                    "timelist_data", "parquet", f"motor_time_list{device_number}_{record_info}.parquet")
    motor_time_list = pd.read_parquet(output_file_parquet).to_dict(orient='list')

    start_date = get_datetime_from_unix(motor_time_list["Time stamps"][0])
    end_date = get_datetime_from_unix(motor_time_list["Time stamps"][-1])
    print("start date:", start_date)
    print("end date:", end_date)

    raw_time = np.array(motor_time_list["Elapsed time"])
    torque_timelist = np.array(motor_time_list["torque_time_list"])

    filtered_data, kept_idx, filtered_idx = t_process.detect_outliers_window_tail(torque_timelist, window_size=20, n_std=3, plot=False)
    filtered_time = raw_time[kept_idx]
    initial_health = np.mean(filtered_data[:10])
    ema_data = get_ema(filtered_data, alpha=0.1, initial_value=initial_health)
    health_indicator = ema_data
     
   #%% 開始動態預測RUL
   
    # 建立新的 detector
    print("Creating new detector...")
    detector = DynamicRULDetector(
            mk_window_size=50,
            ema_coeff=0.1, 
            filt_window=10,
            filt_n_std=2.5,
            mk_epsilon=0.0,
            start_thres_ratio=0.95
        )   
   

    # 測試用 從中間某點開始進行預測
    test_idx=355
    
    for i, (hi, ts) in enumerate(zip(torque_timelist[:test_idx], motor_time_list["Time stamps"][:test_idx])):
        rul = detector.update(hi, ts, is_test=True)
        # 儲存 class 實例
        with open('dynamic_rul_detector.pkl', 'wb') as f:
            pickle.dump(detector, f)
    
    # 讀取斷點資料
    print("Loading existing detector...")
    with open('dynamic_rul_detector.pkl', 'rb') as f:
        detector = pickle.load(f)
    
    # 從斷點資料開始更新    
    for i, (hi, ts) in enumerate(zip(torque_timelist[test_idx:], motor_time_list["Time stamps"][test_idx:])):
        rul = detector.update(hi, ts, is_test=True)
        # 儲存 class 實例
        with open('dynamic_rul_detector.pkl', 'wb') as f:
            pickle.dump(detector, f)
        if rul==0:
            print(f"RUL 結束於全域觀測第 {i} 筆")
            break


    # # 檢查 dynamic_rul_detector.pkl 檔案是否存在
    # if os.path.exists('dynamic_rul_detector.pkl') and test_idx not 0:
    #     # 如果存在則讀取已有的 detector
    #     print("Loading existing detector...")
    #     with open('dynamic_rul_detector.pkl', 'rb') as f:
    #         detector = pickle.load(f)
    # else:
    #     # 如果不存在則建立新的 detector
    #     print("Creating new detector...")
    #     detector = DynamicRULDetector(
    #         mk_window_size=50,
    #         ema_coeff=0.1, 
    #         filt_window=10,
    #         filt_n_std=2.5,
    #         mk_epsilon=0.0,
    #         start_thres_ratio=0.95
    #     )
        
    
    
    
 