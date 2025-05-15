import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.optimize import fsolve
from IPython.display import Image, display


class time_data_preprocess:
    def remove_outliers_moving_window(data, window_size=20, n_std=3, plot=False):
        """
        Remove outliers using moving window method.
        
        Args:
            data (np.array): Input data array
            window_size (int): Size of the moving window
            n_std (float): Number of standard deviations for outlier threshold
            
        Returns:
            tuple: (filtered_data, removed_indices) where filtered_data is the data without outliers
                  and removed_indices are the indices of points that were removed
        """
        data=np.array(data)
        filtered_data = []
        removed_indices = []
        kept_indices = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(data), i + window_size//2)
            window = data[start_idx:end_idx]
            
            local_mean = np.mean(window)
            local_std = np.std(window)
            
            if abs(data[i] - local_mean) <= n_std * local_std:
                filtered_data.append(data[i])
                kept_indices.append(i)
            else: 
                removed_indices.append(i)
        if plot:
            plt.figure(figsize=(12, 8))
            plt.plot(data, label='Original Data')
            plt.plot(removed_indices, data[removed_indices], 'o', label='outlier Data', color='red')
            plt.show(block=False)
        
        return filtered_data, kept_indices, removed_indices
    
    def detect_outliers_window_tail(data, window_size=20, n_std=3, plot=False):
        """
        Detect outliers using a moving window where each window's last point is evaluated
        based on the preceding points within the window.
        
        Args:
            data (np.array): Input data array
            window_size (int): Size of the moving window
            n_std (float): Number of standard deviations for outlier threshold
            plot (bool): Whether to plot the results
        
        Returns:
            tuple: (filtered_data, kept_indices, removed_indices)
        """
        data = np.array(data)
        filtered_data = []
        kept_indices = []
        removed_indices = []

        for i in range(window_size, len(data)):
            window = data[i - window_size:i]  # 前 window_size 筆
            tail = data[i]  # 待判斷點

            local_mean = np.mean(window)
            local_std = np.std(window)

            if abs(tail - local_mean) <= n_std * local_std:
                filtered_data.append(tail)
                kept_indices.append(i)
            else:
                removed_indices.append(i)

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(data, label='Original Data')
            plt.plot(removed_indices, data[removed_indices], 'o', label='Outliers', color='red')
            plt.legend()
            plt.title("Outlier Detection using Moving Window Tail Method")
            plt.show(block=False)

        return np.array(filtered_data), kept_indices, removed_indices
    
    def ewma(data, alpha=0.3):
        """
        Exponentially Weighted Moving Average (EWMA) filter.
        
        Args:
            data (np.array): Input data array
            alpha (float): Smoothing factor (0 < alpha < 1)
            
        Returns:
            np.array: Filtered data using EWMA
        """
        ewma_data = np.zeros_like(data)
        ewma_data[0] = data[0]
        
        for i in range(1, len(data)):
            ewma_data[i] = alpha * data[i] + (1 - alpha) * ewma_data[i - 1]
        
        return ewma_data


def get_ema(data, alpha, initial_value):
    """Exponential Moving Average"""
    data = np.array(data)
    St = np.zeros_like(data)
    St[0] = initial_value
    for i in range(1, len(data)):
        St[i] = alpha * data[i] + (1 - alpha) * St[i - 1]
    return St

def get_smape_and_csmape(actual, predicted, relative_time):
    """Symmetric Mean Absolute Percentage Error"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    smape = np.where(denominator == 0, 0, numerator / denominator)
    
    # Calculate time-weighted cumulative SMAPE using numpy operations
    weights = relative_time / np.sum(relative_time)
    csmape = np.sum(smape * weights)

    return np.mean(smape) * 100, csmape * 100


def rolling_mk(data, roll_length, epsilon, downsample_n=None):
    """Mann-Kendall trend test with optional downsampling"""
    data = np.array(data)
    if downsample_n:
        data_ds = data[::downsample_n]
        long_mk_s = np.zeros_like(data)
        long_mk_z = np.zeros_like(data)
    else:
        data_ds = data

    S = np.zeros(len(data_ds))
    Z = np.zeros(len(data_ds))

    for i in range(len(data_ds) - roll_length + 1):
        segment = data_ds[i:i + roll_length]
        s_i = 0
        for k in range(len(segment) - 1):
            for j in range(k + 1, len(segment)):
                diff = segment[j] - segment[k]
                if diff > epsilon:
                    s_i += 1
                elif abs(diff) <= epsilon:
                    s_i += 0
                else:
                    s_i -= 1
        var_s = (len(segment) * (len(segment) - 1) * (2 * len(segment) + 5)) / 18
        z_i = s_i / np.sqrt(var_s) if abs(s_i) > epsilon else 0
        S[i + roll_length - 1] = s_i
        Z[i + roll_length - 1] = z_i

    if downsample_n:
        for i in range(len(data)):
            if i % downsample_n == 0:
                long_mk_s[i] = S[i // downsample_n]
                long_mk_z[i] = Z[i // downsample_n]
            else:
                long_mk_s[i] = long_mk_s[i - 1]
                long_mk_z[i] = long_mk_z[i - 1]
        return long_mk_s, long_mk_z
    else:
        return S, Z

def estimate_rul_exp(estimate_x, valve, t0, phi1):
    """Calculate RUL based on exponential estimation model"""
    RUL = []
    failure_time = []
    for i in range(len(estimate_x[0])):
        est = ((np.log(abs(valve - phi1))) - estimate_x[0][i] + estimate_x[1][i] * t0) / estimate_x[1][i]
        RUL.append(est - i)
        failure_time.append(est)
    return np.array(RUL), np.array(failure_time)

class plot_helplers():
    def plot_original_data(raw_time, torque_timelist, filtered_idx, filtered_time, filtered_data, device_number, start_date, end_date):
        plt.figure(figsize=(12, 8))
        plt.plot(raw_time, torque_timelist, label='Original Data', alpha=0.5)
        plt.plot(raw_time[filtered_idx], torque_timelist[filtered_idx], 'o', label='Outlier Data', color='red', markersize=4)
        plt.plot(filtered_time, filtered_data, label='Filtered Data', linewidth=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Elapsed time [min]', fontsize=14)
        plt.ylabel('Torque [Nm]', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.title(f'Estimated torque of motor{device_number} versus Time\n{start_date} to {end_date}', fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show(block=False)

    def plot_ewma_data(filtered_time, filtered_data, ema_data, device_number, start_date, end_date):
        plt.figure(figsize=(12, 8))
        plt.plot(filtered_time, filtered_data, label='Filtered Data', alpha=0.5)
        plt.plot(filtered_time, ema_data, label='EWMA Data', linewidth=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Elapsed time [min]', fontsize=14)
        plt.ylabel('Torque [Nm]', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.title(f'EWMA torque of motor{device_number} versus Time\n{start_date} to {end_date}', fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show(block=False)

    def plot_mk_values(filtered_time, mk_Z):
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_time, mk_Z, label='Z value (MK)', color='tab:orange')
        plt.axhline(y=3, color='red', linestyle='--', label='Threshold (+3)')
        plt.axhline(y=-3, color='red', linestyle='--', label='Threshold (-3)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Elapsed time [min]', fontsize=14)
        plt.ylabel('MK Z value', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.title('Mann-Kendall Z values of EWMA Torque', fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show(block=False)

    def plot_rul_validation(filtered_time, ema_data, x_fit, x_fit_relative, params, model_func,
                        RUL_start_idx, Fail_idx_MA, rul_start_thres, RUL_thres, mk_Z):
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        plt.plot(filtered_time, ema_data, label='EMA HI', linewidth=2)
        x_fit_full = np.linspace(0, x_fit_relative[-1], 200)
        y_fit_curve = model_func(x_fit_full, *params)
        plt.plot(x_fit_full + x_fit[0], y_fit_curve, 'r--', label='NLR Fit')

        plt.axvline(x=filtered_time[RUL_start_idx], color='red', linestyle='--', label='RUL Start')
        plt.axvline(x=filtered_time[Fail_idx_MA], color='black', linestyle='--', label='Failure Time')
        plt.axhline(y=rul_start_thres, color='gray', linestyle='--', label='RUL start Threshold')
        plt.axhline(y=RUL_thres, color='gray', linestyle='--', label='Failure Threshold')
        plt.xlabel('Elapsed Time [min]', fontsize=14)
        plt.ylabel('Health Indicator [Nm]', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.grid(True)
        plt.legend()
        plt.title('Health Indicator of test region', fontsize=14)

        plt.subplot(2, 1, 2)
        plt.plot(filtered_time, mk_Z, label='MK Z', color='orange')
        plt.axhline(y=3, color='red', linestyle='--', label='Z = ±3')
        plt.axhline(y=-3, color='red', linestyle='--')
        plt.axvline(x=filtered_time[RUL_start_idx], color='red', linestyle='--', label='RUL Start')
        plt.axvline(x=filtered_time[Fail_idx_MA], color='black', linestyle='--', label='Failure Time')
        plt.xlabel('Elapsed Time [min]', fontsize=14)
        plt.ylabel('MK Z Value', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.grid(True)
        plt.legend()
        plt.title('Mann-Kendall Z values and RUL Trigger', fontsize=14)

        plt.tight_layout()
        plt.show(block=False)

    def plot_rls_tracking(t, y_target, y_hat):
        plt.figure(figsize=(12, 6))
        plt.plot(t, y_target, label='True HI')
        plt.plot(t, y_hat, label='RLS Prediction', linestyle='--')
        plt.xlabel('Time from RUL start [min]')
        plt.ylabel('HI [p.u.]')
        plt.grid(True)
        plt.legend()
        plt.title('RLS Tracking Result')
        plt.tight_layout()
        plt.show(block=False)
    
    # Plot initial curve fitting result
    def plot_curve_fitting(x_fit_relative, y_fit, model_func, params):
        """
        Plot the initial curve fitting result.
        
        Parameters:
        -----------
        x_fit_relative : array-like
            Relative time data for fitting
        y_fit : array-like
            Health indicator data for fitting
        x_plot : array-like
            Time points for plotting fitted curve
        y_plot : array-like 
            Model values for plotting fitted curve
        """
        
        x_fit_full = np.linspace(0, x_fit_relative[-1], 200)
        y_fit_curve = model_func(x_fit_full, *params)
        plt.figure(figsize=(12, 6))
        plt.plot(x_fit_relative, y_fit, 'o', label='Fitting data')
        plt.plot(x_fit_full, y_fit_curve, '-', label='Fitted curve')
        plt.xlabel('Relative time [min]')
        plt.ylabel('Health indicator')
        plt.grid(True)
        plt.legend()
        plt.title('Initial curve fitting result')
        plt.tight_layout()
        plt.show(block=False)
        
    def plot_curve_tracking(t, y_target, y_hat):
        """
        Plot the dynamic fitting result.
        
        Parameters:
        -----------
        t : time array-like
            Time data for fitting
        y_target : array-like
            Health indicator data for fitting
        y_hat : array-like
            Model values for plotting fitted curve
        """
        plt.figure(figsize=(12, 6))
        plt.plot(t, y_target, label='True HI')
        plt.plot(t, y_hat, label='RLS Prediction', linestyle='--')
        plt.xlabel('Time from RUL start [min]')
        plt.ylabel('HI [p.u.]')
        plt.grid(True)
        plt.legend()
        plt.title('RLS Tracking Result')
        plt.tight_layout()
        plt.show(block=False)

    def plot_failure_prediction(t, time_to_failure, alpha=0.1):
        """
        Plot the failure prediction timeline with error range.
        
        Args:
            t: array-like, time values
            time_to_failure: array-like, predicted time to failure values
            alpha: float, error range factor (default: 0.1)
        """
        plt.figure(figsize=(12, 6))
        
        # Plot error range
        actual_time = t[-1]-t
        error = alpha * actual_time
        plt.fill_between(t-t[0], 
                        actual_time - error, 
                        actual_time + error, 
                        alpha=0.2, 
                        color='pink',  # Changed from 'blue' to 'pink'
                        label=f'±{int(alpha*100)}% Error Range')
        
        # Plot actual and predicted values
        plt.plot(t-t[0], actual_time, label='Actual Time to threshold')
        plt.plot(t-t[0], time_to_failure, 'o', markersize=5,markerfacecolor='none', label='Predicted Time to threshold')
        plt.plot(t-t[0], time_to_failure, color='dimgray', alpha=0.75)
        
        plt.xlabel('Time elpased after start [min]')
        plt.ylabel('Remaining time to threshold [min]')
        plt.title('Predicted threshold Time from RLS Model')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
