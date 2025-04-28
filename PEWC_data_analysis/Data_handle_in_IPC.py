 #%%
# This program do the same thing as Data handle but the motor status is calculated in here

import math
import csv
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import Motor_global_vars

# rul result operation (still developing)
def rul_result():
    # use dictionary to storage RUL result
    rul_est = {'Health_Indicator': 0, 'RUL_EN': 0, 'RUL_alpha': 0,
               'RUL_beta': 0, 'RUL_phi': 0, 'RUL(min)': 0, 'RUL_limit': 0, 't_N(min)': 0}
    return rul_est


def fft_test(signal_real: np.ndarray, signal_imag: np.ndarray, sampling_rate: float):
    signal_complex = signal_real + 1j * signal_imag
    signal_complex = signal_complex.flatten()
    N = len(signal_complex)
    fft_vals = np.fft.fft(signal_complex, n=N)
    fft_vals_shifted = np.fft.fftshift(fft_vals)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/sampling_rate))
    # fft_result = np.abs(fft_vals_shifted) / N
    fft_result=fft_vals_shifted / N
    return freqs, fft_result


# get motor operating condition
def get_motor_cond_list(motor_cond_raw):
    # conditions: 'Speed(Rpm)', 'Torque(N)', 'Power(KW)', 'Efficiency(%)', 'Efficiency_alarm'
    motr_cond_list = []
    motr_cond_list.append(u16_to_true_data(motor_cond_raw['speed'], pu_gain=Motor_global_vars.Base_Speed))
    motr_cond_list.append(u16_to_true_data(motor_cond_raw['torque'], pu_gain=Motor_global_vars.Base_Torque))
    motr_cond_list.append(u16_to_true_data(motor_cond_raw['power'], pu_gain=Motor_global_vars.Base_Power))
    # power is offset by 0.00001 to avoid divide by zero
    motr_cond_list.append(
        motr_cond_list[0] / 60 * 2 * math.pi * motr_cond_list[1] / (motr_cond_list[2] + 0.000001) * 100)
    motr_cond_list.append(int(motr_cond_list[3] < 90))
    return motr_cond_list


# get cn diagnosis result
def get_cn_sts_list(i_alpha, i_beta, debug=False, threshold=0.05):

     # normalize the current data
    i_alpha = i_alpha/Motor_global_vars.Base_current
    i_beta = i_beta/Motor_global_vars.Base_current

    # calculate winding fault result by raw current
    L = len(i_alpha)  # data length
    #  calculate fft
    freqs, fft_result_cplx = fft_test(i_alpha, i_beta, Motor_global_vars.sampling_rate)
    
    fft_result=np.abs(fft_result_cplx)  # get the magnitude of fft result
    
    # find frequency index of characteristic frequencies
    fund_freq_idx = np.argmax(fft_result)
    minus1_freq_idx = L - fund_freq_idx
    minus1_freq = freqs[minus1_freq_idx]
    fund_freq = freqs[fund_freq_idx]
    
    # find characteristic frequencies components
    fund_freq = fft_result_cplx[fund_freq_idx]
    minus1_freq = fft_result_cplx[minus1_freq_idx]

    # get phase angle of fundamental and negative sequence components
    fund_phase = np.angle(fund_freq, deg=True) 
    minus1_phase = np.angle(minus1_freq, deg=True)
    phase_offset = minus1_phase + fund_phase
    
    # find amplitude
    fund_freq_amp=np.abs(fund_freq)
    minus1_freq_amp=np.abs(minus1_freq)
    
    # calculate CN value
    CN_value = minus1_freq_amp / fund_freq_amp
    
    # calculate CN value as complex number with CN_value as radius and phase_offset as angle
    CN_value_complex = CN_value * np.exp(1j * np.deg2rad(phase_offset))
    
    # magnitude to rms
    motor_cn_sts = {
        'Icn_x': CN_value_complex.real*32768+32767,
        'Icn_y': CN_value_complex.imag*32768+32767,
        'I_rms': fund_freq_amp*32768+32767,
    }
    if debug:
        # plot the fft result
        plt.figure(figsize=(10, 5))
        # plt.stem(freqs, fft_result)
        plt.plot(freqs, 20*np.log10(fft_result))
        #  plot the characteristic frequencies
        plt.axvline(x=freqs[fund_freq_idx], color='r', linestyle='--', label=f"Fundamental frequency:{fund_freq:.1f} Hz")
        plt.axvline(x=freqs[minus1_freq_idx], color='r', linestyle='--', label=f"-1 frequency:{minus1_freq:.1f} Hz")
        plt.plot(freqs[fund_freq_idx], 20*np.log10(fund_freq_amp), 'ro', label=f"Fundamental frequency Amplitude:{fund_freq_amp:.6f} A")
        plt.plot(freqs[minus1_freq_idx], 20*np.log10(minus1_freq_amp), 'rx', label=f"-1 frequency Amplitude:{minus1_freq_amp:.6f} A")
        plt.axhline(y=20*np.log10(fund_freq_amp*0.05), color='k', linestyle='--', label='threshold')
        plt.xlim(-1000, 1000)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('FFT Amplitude (dB)')
        plt.title(f'FFT Result, CN Value: {CN_value:.6f}')
        plt.grid()
        plt.legend()
        plt.show(block=False)
        
        # plot the CN xy plot result
        plt.figure(figsize=(10, 5))
        # Draw a circle with radius = threshold
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = threshold * np.cos(theta)
        circle_y = threshold * np.sin(theta)
        plt.plot(circle_x, circle_y, 'k--', alpha=0.5)
        # Plot the CN value as a complex number (x marker)
        plt.plot([CN_value_complex.real], [CN_value_complex.imag], 'rx', markersize=10, label=f'CN Value: {CN_value:.3f}∠{phase_offset:.1f}°')

        plt.xlim(-(threshold+0.02), (threshold+0.02))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('FFT Amplitude (dB)')
        plt.title(f'FFT Result, CN Value: {CN_value:.6f}')
        plt.grid()
        plt.legend()
        plt.show(block=False)

    return motor_cn_sts, fund_freq, minus1_freq


# get the torque estimation
def estimate_torque(data_read, speed_v=3530, debug=False):

    """
    Estimate the motor torque based on voltage and current inputs.
    :param v_a_raw: Voltage alpha component
    :param v_b_raw: Voltage beta component
    :param i_alpha: Current alpha component
    :param i_beta: Current beta component
    :param speed_v: Motor speed in rpm (default: 900)
    :param debug: Boolean flag to enable debugging plots (default: False)
    :return: Estimated torque array
    """

    v_a_raw = np.array(data_read["Voltage alpha"])
    v_c_raw = np.array(data_read["Voltage beta"])
    i_alpha = np.array(data_read["Current alpha"])
    i_beta  = np.array(data_read["Current beta"])

    # offset calibration
    v_a_raw = v_a_raw - np.mean(v_a_raw)
    v_c_raw = v_c_raw - np.mean(v_c_raw)
    i_alpha = i_alpha - np.mean(i_alpha)
    i_beta = i_beta - np.mean(i_beta)

    # Preprocess the voltage data, transform the data to alpha-beta frame
    v_alpha = v_a_raw
    v_beta = (-v_c_raw + v_a_raw - v_c_raw) / np.sqrt(3)

    class EMF:
        def __init__(self):
            self.Alpha = 0.0
            self.Beta = 0.0
            self.Alpha_last = 0.0
            self.Beta_last = 0.0
            self.Alpha_LPF = 0.0
            self.Beta_LPF = 0.0
            self.Alpha_LPF_last = 0.0
            self.Beta_LPF_last = 0.0

    def emf_to_lpf(sampling_time, lpf_radius, emf_obj):
        lpf_radius_t = lpf_radius * sampling_time
        emf_coef1 = sampling_time / (lpf_radius_t + 2)
        emf_coef2 = (lpf_radius_t - 2) / (lpf_radius_t + 2)

        emf_obj.Alpha_LPF = emf_coef1 * (emf_obj.Alpha_last + emf_obj.Alpha) - emf_coef2 * emf_obj.Alpha_LPF_last
        emf_obj.Beta_LPF = emf_coef1 * (emf_obj.Beta_last + emf_obj.Beta) - emf_coef2 * emf_obj.Beta_LPF_last

        emf_obj.Alpha_last = emf_obj.Alpha
        emf_obj.Beta_last = emf_obj.Beta

        emf_obj.Alpha_LPF_last = emf_obj.Alpha_LPF
        emf_obj.Beta_LPF_last = emf_obj.Beta_LPF

    def flux_comp(omega_e, lpf_radius):
        if np.abs(omega_e) < 1:
            mag_comp = 1.0
        else:
            mag_comp = np.abs(omega_e) / np.sqrt(omega_e ** 2 + lpf_radius ** 2)

        phase_comp = -57.29578 * np.arctan2(lpf_radius, omega_e) / 360  # Degree to radians conversion

        return mag_comp, phase_comp

    # Compute necessary parameters
    fs = Motor_global_vars.sampling_rate  # Sampling rate
    flux_rs = 0.5  # Motor stator resistance
    tsim = 1 / fs  # Time step

    we = (speed_v / 60) * (np.pi * 2) # electrical angular velocity
    coef = 0.2
    cross_freq = 15.0

    intgr_bw_f = max((we / (np.pi * 2)) * coef, cross_freq)
    fast_wc = intgr_bw_f * (np.pi * 2)

    # Process each time step
    emf1 = EMF()
    alpha_lpf_values, beta_lpf_values = [], []
    alpha_raw_values, beta_raw_values = [], []

    for va, vb, ia, ib in zip(v_alpha, v_beta, i_alpha, i_beta):
        emf1.Alpha = va - (ia * flux_rs)
        emf1.Beta = vb - (ib * flux_rs)
        alpha_raw_values.append(emf1.Alpha)
        beta_raw_values.append(emf1.Beta)
        emf_to_lpf(tsim, fast_wc, emf1)
        alpha_lpf_values.append(emf1.Alpha_LPF)
        beta_lpf_values.append(emf1.Beta_LPF)

    # Flux compensation
    mag_comp2, phase_comp2 = flux_comp(we, fast_wc)

    # Apply phase and magnitude compensation
    alpha_compensated_values, beta_compensated_values = [], []
    for alpha, beta in zip(alpha_lpf_values, beta_lpf_values):
        ds = alpha * mag_comp2
        qs = beta * mag_comp2
        angle = phase_comp2
        sine, cosine = np.sin(angle), np.cos(angle)
        alpha_transformed = ds * cosine - qs * sine
        beta_transformed = qs * cosine + ds * sine
        alpha_compensated_values.append(alpha_transformed)
        beta_compensated_values.append(beta_transformed)

    # Torque estimation
    torque_v = 1.5 * 2 * ((np.array(alpha_compensated_values) * i_beta) - (np.array(beta_compensated_values) * i_alpha))
    torque_avg = np.mean(np.abs(torque_v[-500:]))
    # Power and efficiency estimation
    Power_M= torque_avg*speed_v*2*np.pi/60
    Power_E = 1.5*np.mean((v_alpha * i_alpha + v_beta * i_beta))
    efficiency = Power_M / Power_E * 100
    power_sts = {
        'Power_M': Power_M,
        'Power_E': Power_E,
        'Efficiency': efficiency,
        'Efficiency_alarm': int(efficiency < 90)
    }

    if debug :
        date_time=pd.to_datetime(data_read["Unix Time"], unit='s').strftime('%Y-%m-%d %H:%M:%S')

        print("Estimated Torque:", np.mean(torque_v[-Motor_global_vars.data_length:]))
        # print result
        for key, value in power_sts.items():
            print(f"{key}: {value}")
        time = np.arange(len(v_alpha)) * tsim

        # plot the flux values
        plt.figure(figsize=(10, 5))
        plt.plot(time, alpha_compensated_values, label='Flux Alpha')
        plt.plot(time, beta_compensated_values, label='Flux Beta ')
        plt.xlabel('Time (s)')
        plt.ylabel('Flux Values')
        plt.legend()
        plt.title('Flux Values'+date_time)
        plt.grid()
        plt.show(block=False)

        # plot the voltage values
        plt.figure(figsize=(10, 5))
        plt.plot(time, v_alpha, label='Voltage Beta (raw)')
        plt.plot(time, v_beta, label='Voltage Beta (raw)')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage Values')
        plt.legend()
        plt.title('Voltage Values')
        plt.grid()
        plt.show(block=False)

        # plot the current values
        plt.figure(figsize=(10, 5))
        plt.plot(time, i_alpha, label='Current Alpha (raw)')
        plt.plot(time, i_beta, label='Current Beta (raw)')
        plt.xlabel('Time (s)')
        plt.ylabel('Current Values')
        plt.legend()
        plt.title('Current Values')
        plt.grid()
        plt.show(block=False)

        # plot the EMF values
        plt.figure(figsize=(10, 5))
        plt.plot(time, alpha_raw_values, label='EMF Alpha (raw)')
        plt.plot(time, beta_raw_values, label='EMF Beta (raw)')
        plt.xlabel('Time (s)')
        plt.ylabel('EMF Values')
        plt.legend()
        plt.title('EMF Values')
        plt.grid()
        plt.show(block=False)

        # plot torque estimation
        plt.figure(figsize=(10, 5))
        plt.plot(time, torque_v, label='Torque (Voltage Model)')
        if len(torque_v) > 2500:
            plt.plot(time[2250:2500], torque_v[2250:2500], label='monitored torque region', color='red')
            torque_avg=np.mean(torque_v[2250:2500])
        else:
            plt.plot(time[-500:], torque_v[-500:], label='monitored torque region', color='red')
            torque_avg=np.mean(torque_v[-500:])
            
        plt.axhline(y=torque_avg, color='k', linestyle='--', label='Averaged torque')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (N.m)')
        plt.legend()
        plt.title(f'Torque Estimation :{torque_avg:.2f} (N.m)')
        plt.grid()
        plt.show(block=False)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('EMF Alpha (Raw)', color='tab:blue')
        ax1.plot(time, alpha_raw_values, label='EMF Alpha (Raw)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.set_ylabel('EMF Alpha (Filtered)', color='tab:red')
        # ax2.plot(time, alpha_lpf_values, label='EMF Alpha (Filtered)', color='tab:red')
        ax2.plot(time, alpha_compensated_values, label='EMF Alpha (Compensated)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')

        plt.title('EMF Alpha - Raw vs Filtered')
        plt.grid()
        plt.show(block=False)
    # return the torque value and the estimated flux
    return torque_v, alpha_compensated_values, beta_compensated_values, v_alpha, v_beta, power_sts

# rescale the Uint16 data to float
def u16_to_true_data(u16_data, pu_gain=1):
    float_data = (u16_data - 32768) / 32768 * pu_gain
    return float_data


# read the smple ccae result for ui test
def read_sample_ccae():
    file_path = "CCAE_sample.csv"
    ccae_dict = {}
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            ccae_dict[index] = row  # 保留整行數據，包括字串和數字
    return ccae_dict


# padding function to write specific format of data file
def pad_list_with_empty_strings(input_list, target_length=8):
    # if list length<target_length, pad it
    if len(input_list) < target_length:
        input_list.extend([""] * (target_length - len(input_list)))
    return input_list


# generate demagnetization report from raw flux data
def get_demag_report(flux_alpha, flux_beta, flux_fft):
    pm_strength = np.average(np.sqrt(flux_alpha ** 2 + flux_beta ** 2))
    pm_thres = min(pm_strength / Motor_global_vars.Base_flux, 1) * 100
    flux_thd = calculate_thd_with_fftshift(flux_fft)
    pm_alarm = int(pm_thres < 90)
    demag_status = [pm_strength, pm_thres, flux_thd, pm_alarm]
    return demag_status


# calculate thd of flux fft (need to check )
def calculate_thd_with_fftshift(fft_complex):
    """
    計算經 fftshift 處理後的 FFT 頻譜的 THD（總諧波失真）。

    Parameters:
        fft_complex (np.ndarray): 經 fftshift 的複數 FFT 頻譜數據。

    Returns:
        thd (float): THD 值（單位：百分比）。
    """
    # 1. 反向 fftshift 將頻譜恢復為未移位狀態
    fft_complex = np.fft.ifftshift(fft_complex)

    # 2. 計算幅值譜，忽略 DC 分量
    fft_magnitude = np.abs(fft_complex)
    fft_magnitude[0] = 0  # 忽略 DC 分量

    # 3. 找到基波的索引（最大幅值）
    fundamental_index = np.argmax(fft_magnitude)
    V1 = fft_magnitude[fundamental_index]  # 基波幅值

    # 4. 計算諧波總和（排除基波）
    fft_magnitude[fundamental_index] = 0  # 忽略基波
    harmonic_power = np.sum(fft_magnitude ** 2)

    # 5. 計算 THD
    thd = np.sqrt(harmonic_power) / V1  # 轉換為百分比

    return thd
    
    
   #%%
if __name__ == "__main__":
 
    # read the data while skip the first 2 row to avoid parse error
    df = pd.read_csv('../PEWC dataset/read test data/PEWC_test_2.csv', skiprows=1)
    # df = pd.read_csv('../PEWC dataset/read test data/PEWC_ver2_0217_RUL2.parquet', skiprows=1)
    # df = pd.read_csv('IPC_test_data/RUL_Data_5_962.csv', skiprows=1)

    v_a = df['V_alpha'].to_numpy()
    v_c = df['V_beta'].to_numpy()
    i_alpha = df['I_alpha'].to_numpy()
    i_beta = df['I_beta'].to_numpy()

    motor_cn_sts, fund_freq, minus1_freq = get_cn_sts_list(i_alpha, i_beta)
    print(f"fundamental frequency: {fund_freq} Hz")
    
    # print result
    for key, value in motor_cn_sts.items():
        print(f"{key}: {value}")
    #%%
    torque, alpha_compensated_values, beta_compensated_values, v_alpha, v_beta, power_sts = estimate_torque(v_a, v_c, i_alpha, i_beta, debug=True)
    #%%





