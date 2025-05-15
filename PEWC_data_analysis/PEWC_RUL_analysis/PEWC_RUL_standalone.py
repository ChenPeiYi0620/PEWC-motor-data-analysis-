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

def get_rul_realtime(dynamic_data_path, rul_package, device_number, record_info):
    
    def simple_plot(x, y1, y2, title='Plot', xlabel='X', ylabel='Y'):
        x = np.array(x)
        fig = plt.figure(figsize=(10, 6))
        plt.plot(x, y1, label='Data1')
        plt.plot(x, y2, label='Data2')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.show(block=False)
        return fig
    
    rul_package['observed_count']=rul_package['observed_count']+1
    
    # 若資料量不足則更新資料並跳出
    if rul_package['observed_count'] <= rul_package['MK_window_size']:
        rul_package['MK_window_raw'][rul_package['observed_count']-1]=rul_package['current_observe']
        rul_package['MK_window_filt'][rul_package['observed_count']-1]=rul_package['current_observe']
        rul_package['EMA_HI']=(1-rul_package['EMA_coeff'])*rul_package['EMA_HI_last']+rul_package['EMA_coeff']*rul_package['current_observe']
        rul_package['EMA_HI_last']= rul_package['EMA_HI']
        rul_package['MK_window_EMA'][rul_package['observed_count']-1]=rul_package['EMA_HI']
        rul_package['MK_window_stamp'][rul_package['observed_count']-1]=rul_package['current_stamp']
        return[]
    else:
        
        # 更新慮波值
        n_std=rul_package['filt_n_std'] # 三倍標準差
        last_obsereves=rul_package['MK_window_raw'][-rul_package['filt_window']:]
        local_mean = np.mean(last_obsereves)
        local_std = np.std(last_obsereves)
        
        # 若為離群值則用平均值代替
        if abs(rul_package['current_observe'] - local_mean) >= n_std * local_std:
            # return[]
            rul_package['MK_window_filt'] = np.delete(rul_package['MK_window_filt'], 0)
            rul_package['MK_window_raw'] = np.delete(rul_package['MK_window_raw'], 0)
            rul_package['MK_window_filt'] = np.append(rul_package['MK_window_filt'], rul_package['MK_window_raw'][-1]) 
            rul_package['MK_window_raw'] = np.append(rul_package['MK_window_raw'], rul_package['current_observe'])
            # print(f"count= {rul_package['observed_count']}, local_mean={local_mean:.4f}, current_observe={rul_package['current_observe']:.4f}")        
            # fig=simple_plot(range(len(rul_package['MK_window_raw'])), rul_package['MK_window_raw'], rul_package['MK_window_filt'], title='RUL EMA', xlabel='Time', ylabel='EMA HI')
            # plt.close(fig)
        else:
            rul_package['MK_window_filt'] = np.delete(rul_package['MK_window_filt'], 0)
            rul_package['MK_window_raw'] = np.delete(rul_package['MK_window_raw'], 0)
            rul_package['MK_window_filt'] = np.append(rul_package['MK_window_filt'], rul_package['current_observe'])
            rul_package['MK_window_raw'] = np.append(rul_package['MK_window_raw'], rul_package['current_observe'])
            # fig=simple_plot(range(len(rul_package['MK_window_raw'])), rul_package['MK_window_raw'], rul_package['MK_window_filt'], title='RUL EMA', xlabel='Time', ylabel='EMA HI')
            # plt.close(fig)
        # 更新原始資料
        
        # 更新EMA資料
        rul_package['EMA_HI']=(1-rul_package['EMA_coeff'])*rul_package['EMA_HI_last']+rul_package['EMA_coeff']*rul_package['current_observe']
        rul_package['EMA_HI_last']=rul_package['EMA_HI']
        rul_package['MK_window_EMA'] = np.delete(rul_package['MK_window_EMA'], 0)
        rul_package['MK_window_EMA'] = np.append(rul_package['MK_window_EMA'], rul_package['EMA_HI'])
        rul_package['MK_window_stamp'] = np.delete(rul_package['MK_window_stamp'], 0)
        rul_package['MK_window_stamp'] = np.append(rul_package['MK_window_stamp'], rul_package['current_stamp'])
        
        # 除錯用
        # fig=simple_plot(range(len(rul_package['MK_window_raw'])), rul_package['MK_window_raw'], rul_package['MK_window_filt'], title='RUL EMA', xlabel='Time', ylabel='EMA HI')
        # plt.close(fig)
        # fig=simple_plot(range(len(rul_package['MK_window_raw'])), rul_package['MK_window_raw'], rul_package['MK_window_EMA'], title='RUL EMA', xlabel='Time', ylabel='EMA HI')
        # plt.close(fig)
        
        
    # RUL啟動判定
    if rul_package['RUL_enable']== 0 and rul_package['observed_count']>=rul_package['MK_window_size']:
        
        # 進行MK檢定
        _, Z_window = rolling_mk(
            data=rul_package['MK_window_EMA'], 
            roll_length=rul_package['MK_window_size'],
            epsilon=rul_package['MK_epsilon'])
        z_i = Z_window[-1]
        
        if np.abs(z_i) < 3 or rul_package['EMA_HI'] > rul_package['RUL_start_thres']:
            return []
        
        #若啟動則進行RUL初始化
        else:
            rul_package['RUL_start_idx'] = rul_package['observed_count'] - rul_package['MK_window_size']
            fig=simple_plot(range(len(rul_package['MK_window_filt'])), rul_package['MK_window_filt'], rul_package['MK_window_EMA'], title='RUL EMA', xlabel='Time', ylabel='EMA HI')
            plt.close(fig)
            # 進行RUL預測
            rul_package['RUL_enable'] = 1
            # 用NLR初始化預測參數
            y_fit = rul_package['MK_window_EMA']   
            x_start = convert_unix_to_minutes([rul_package['MK_window_stamp'][0]])     
            x_fit_relative = convert_unix_to_minutes(rul_package['MK_window_stamp'])-x_start
            
            fig=simple_plot(x_fit_relative, rul_package['MK_window_filt'], y_fit, title='RUL EMA', xlabel='Time', ylabel='EMA HI')
            plt.close(fig)

            initial_guess =rul_package['NLR_initial']
            bounds = rul_package['Proj_bound']

            params, _ = curve_fit(model_func, x_fit_relative, y_fit, p0=initial_guess, bounds=bounds, maxfev=2000)
            phi0, alpha, beta, gamma = params
            print("Initial model parameters:")
            print(f"phi0 = {phi0:.4f}, alpha = {alpha:.4f}, beta = {beta:.4f}, gamma = {gamma:.4f}")
            print(f"Fail_idx_MA = {Fail_idx_MA}, RUL_start_idx = {RUL_start_idx}")
            
            
    # 若RUL已啟動則進行RUL預測
    
    return []
    
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
    
    def update(self, value: float, timestamp) -> int:
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
                
                print(f'Elapsed time : {int(t_relative)} min ,phi0={x_temp[0]:.4f}, alpha={x_temp[0]:.4f},beta={beta_i:.4f}', end="")
                print(f' e={e:.4f}, k={K[0,0]:.6f}, P0={P0[0,0]:.6f}')
                # 取出RUL曲線點
                x_curve = np.linspace(0, sol, 100)
                y_curve = phi0_i + alpha_i * x_curve + beta_i / (x_curve + self.RLS_initial_params['gamma'])
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
                return sol - t_relative
        except:
            return None

        return None

if __name__ == "__main__":
   

    rul_dynamic_package={
        # 資料前處理相關
        'Initial_health': 0, # 初始健康值
        'current_observe': 0, # 當前觀測資料
        'current_stamp': '', # 當前時間戳記
        'EMA_HI_last': 0, # 上一步 EMA健康指標
        'EMA_coeff': 0.1, # EMA 係數
        'EMA_HI': 0, # EMA後之健康指標
        'observed_count': 0, # 已觀測資料數量
        'filt_window': 10, # 濾波窗大小
        'filt_n_std': 2.5, # 濾波標準差倍數
        
        # MK 檢測相關
        'MK_window_raw': np.zeros(MK_window), # 初始化長度為 MK_window 的 nparray
        'MK_window_filt': np.zeros(MK_window), # 過濾後的窗內資料
        'MK_window_EMA': np.zeros(MK_window), # EMA 處理後的窗內資料
        'MK_window_size': MK_window, # MK窗大小
        'MK_epsilon': 0, # MK 檢測的 epsilon 值
        'RUL_enable':0,      # RUL 啟旗標
        
        # 預測相關
        'RUL_thres':[],
        'RUL_start_thres':0,
        'RUL_start_time': [], # RUL 啟動時間 ('Unix')
        'RUL_start_idx': 0, # RUL 啟動位置(debug)
        'RUL_parameter_last':[],   # RUL 上一步模型參數
        'MK_window_stamp':[None]*MK_window,       # RUL 時間戳記
        'NLR_initial':[], # NLR 初始化參數
        
        # RLS相關
        'Proj_bound':[], # 預測邊界
        
    }
    
    detector = DynamicRULDetector(
    mk_window_size=50,
    ema_coeff=0.1,
    filt_window=10,
    filt_n_std=2.5,
    mk_epsilon=0.0,
    start_thres_ratio=0.95
    )
    
    # 設定當前工作目錄為腳本所在的目錄
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
   # GPT 版本
   

    for i, (hi, ts) in enumerate(zip(torque_timelist, motor_time_list["Time stamps"])):
        idx = detector.update(hi, ts)
            
   
    window_observed_data=[]
   # 初始化RUL預測處理器
    rul_dynamic_package['Initial_health']=initial_health
    rul_dynamic_package['MK_window_data']=window_observed_data
    rul_dynamic_package['RUL_thres']=initial_health*0.82
    rul_dynamic_package['RUL_start_thres']=initial_health * 0.95
    rul_dynamic_package['EMA_HI_last']=initial_health
    rul_dynamic_package['NLR_initial'] = [initial_health, -1, 1, 1]
    rul_dynamic_package['Proj_bound']  = ([0, -np.inf, -np.inf, 0], [np.inf, 0, np.inf, np.inf])
    # rul_dynamic_package[]=
    # rul_dynamic_package[]=
    # rul_dynamic_package[]=
    
    for i in range(len(torque_timelist)):
        rul_dynamic_package['current_observe']=torque_timelist[i]
        rul_dynamic_package['current_stamp']=motor_time_list["Time stamps"][i]
        get_rul_realtime('', rul_dynamic_package, device_number, record_info)
   
    

    #%% MK 檢定與 RUL 啟動判定

    mk_S, mk_Z = rolling_mk(ema_data, roll_length=MK_window, epsilon=0)

    RUL_thres = initial_health * 0.82
    if np.max(ema_data) < RUL_thres:
        RUL_thres = np.max(ema_data)
    fail_candidates = np.where(ema_data <= RUL_thres)[0]
    if len(fail_candidates) == 0:
        raise ValueError("資料未達到失效門檢，請確認數據與門檢設定。")
    Fail_idx_MA = fail_candidates[0]

    rul_start_thres = initial_health * 0.95
    trigger_candidates = np.where((np.abs(mk_Z) > 3) & (ema_data < rul_start_thres))[0]
    if len(trigger_candidates) == 0:
        raise ValueError("找不到符合條件的 RUL 啟動點")
    RUL_start_idx = trigger_candidates[0]

    #%% 非線性參數擬合

    #  lin_mod = 'phi0+alpha*x+beta/(x+gamma)';

    x_fit = filtered_time[RUL_start_idx - MK_window + 1:RUL_start_idx + 1]
    y_fit = ema_data[RUL_start_idx - MK_window + 1:RUL_start_idx + 1]
    x_fit_relative = x_fit - x_fit[0]

    initial_guess = [initial_health, -1, 1, 1]
    bounds = ([0, -np.inf, -np.inf, 0], [np.inf, 0, 0, np.inf])

    params, _ = curve_fit(model_func, x_fit_relative, y_fit, p0=initial_guess, bounds=bounds)
    
    phi0, alpha, beta, gamma = params
    print("Initial model parameters:")
    print(f"phi0 = {phi0:.4f}, alpha = {alpha:.4f}, beta = {beta:.4f}, gamma = {gamma:.4f}")
    print(f"Fail_idx_MA = {Fail_idx_MA}, RUL_start_idx = {RUL_start_idx}")


    # Call the function
    plt_helper.plot_curve_fitting(x_fit_relative, y_fit, model_func, params)
    plt_helper.plot_rul_validation(filtered_time, ema_data, x_fit, x_fit_relative, params, model_func,
                    RUL_start_idx, Fail_idx_MA, rul_start_thres, RUL_thres, mk_Z)

    #%% RLS Tracking 建立迴歸矩陣並遞迴擬合

    Beta = 0.99
    P0 = 5 * np.eye(3)
    x_initial = np.array([phi0, alpha, beta])


    track_range = np.arange(RUL_start_idx + 1, Fail_idx_MA + 1)
    t = filtered_time[track_range] - filtered_time[RUL_start_idx - MK_window]
    h_exp = np.stack([np.ones_like(t), t, 1 / (t + gamma)], axis=1)

    y_target = ema_data[track_range]

    X_est = []
    y_hat = []
    P = P0.copy()
    x = x_initial.copy()

    for i in range(len(t)):
        h = h_exp[i].reshape(-1, 1)
        y = y_target[i]
        y_pred = np.dot(h.T, x.reshape(-1, 1))[0, 0]
        e = y - y_pred
        K = P @ h / (Beta + h.T @ P @ h)
        x = x + (K.flatten() * e)
        P = (np.eye(3) - K @ h.T) @ P / Beta
        y_hat.append(y_pred)
        X_est.append(x.copy())

    X_est = np.array(X_est)
    y_hat = np.array(y_hat)
    print("RLS completed.")

    plt_helper.plot_rls_tracking(t, y_target, y_hat)

    #%% 預測失效時間 (RUL)

    fail_time_pred = []
    for i, x_hat in enumerate(X_est):
        phi0_i, alpha_i, beta_i = x_hat
        def f_eq(x):
            return phi0_i + alpha_i * x + beta_i / (x + gamma) - RUL_thres
        try:
            sol = fsolve(f_eq, x0=10000)[0]
            if sol > 0:
                fail_time_pred.append(sol + filtered_time[RUL_start_idx - MK_window])
            else:
                fail_time_pred.append(np.nan)
        except:
            fail_time_pred.append(np.nan)

    fail_time_pred = np.array(fail_time_pred)


    # %% # === 圖形繪製 ===
    # Call the functions
    plt_helper.plot_original_data(raw_time, torque_timelist, filtered_idx, filtered_time, filtered_data, device_number, start_date, end_date)
    plt_helper.plot_ewma_data(filtered_time, filtered_data, ema_data, device_number, start_date, end_date)
    plt_helper.plot_mk_values(filtered_time, mk_Z)
    plt_helper.plot_curve_tracking(t, y_target,y_hat)

    #%% RLS 預測失效時間圖

    time_to_failure = fail_time_pred - (t + filtered_time[RUL_start_idx - MK_window])
    actual_failure_time = filtered_time[Fail_idx_MA] - (t + filtered_time[RUL_start_idx - MK_window])

    # Calculate SMAPE for time_to_failure predictions
    valid_predictions = ~np.isnan(time_to_failure)
    smape, csmape = get_smape_and_csmape(time_to_failure[valid_predictions], actual_failure_time[valid_predictions], t-t[0])
    print(f"SMAPE/SMAPE  for RUL predictions: {smape:.2f}/{csmape:.2f}%")

    plt_helper.plot_failure_prediction(t, time_to_failure, alpha=0.2)


    #%% 預測曲線動畫儲存並播放，含交會點與 HI 歷程顯示 (模擬 plotIntersectionAnimation 功能)

    # Check if animation files exist
    gif_path = os.path.join("..", "animations", "rul_prediction_animation.gif")
    mp4_path = os.path.join("..", "animations", "rul_prediction_animation.mp4")
    html_path = os.path.join("..", "animations", "rul_prediction_animation.html")

    if os.path.exists(html_path) and os.path.exists(mp4_path):
        print("Loading existing animation files...")
        # Display existing MP4 file
        # display(display_mp4(mp4_path))
        # Display existing HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        display(HTML(html_content))
    else:
        print("Creating new animation files...")
        # Create new animations
        animation_html, rul_gif, rul_mp4 = create_rul_prediction_animation(
            X_est, ema_data, y_hat, RUL_thres, filtered_time, RUL_start_idx, MK_window, gamma
        )
        # Display the newly created MP4
        # display(display_mp4(rul_mp4))
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        display(HTML(html_content))


    # %% 