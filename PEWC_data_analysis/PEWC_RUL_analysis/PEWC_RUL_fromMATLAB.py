import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
from datetime import datetime
import pandas as pd
import os
import sys
from IPython.display import HTML, display, IFrame

# 設定當前工作目錄為腳本所在的目錄
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from IPython.display import HTML
from base64 import b64encode


def get_datetime_from_unix(unix_time):
    return pd.to_datetime(unix_time, unit='s').strftime('%Y-%m-%d')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

#%% 資料讀取與前處理

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

filtered_data, kept_idx, filtered_idx = t_process.remove_outliers_moving_window(torque_timelist, window_size=10, n_std=2.5, plot=False)
filtered_time = raw_time[kept_idx]
initial_health = np.mean(filtered_data[:10])
ema_data = get_ema(filtered_data, alpha=0.1, initial_value=initial_health)
health_indicator = ema_data

#%% MK 檢定與 RUL 啟動判定

MK_window = 50
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

def model_func(x, phi0, alpha, beta, gamma):
    return phi0 + alpha * x + beta / (x + gamma)

x_fit = filtered_time[RUL_start_idx - MK_window + 1:RUL_start_idx + 1]
y_fit = ema_data[RUL_start_idx - MK_window + 1:RUL_start_idx + 1]
x_fit_relative = x_fit - x_fit[0]

initial_guess = [initial_health, -1, 1, 1]
bounds = ([0, -np.inf, -np.inf, 0], [np.inf, 0, np.inf, np.inf])

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

#%% Failure Time Prediction (RUL)

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