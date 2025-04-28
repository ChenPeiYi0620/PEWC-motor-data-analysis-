import numpy as np
import os 
import pandas as pd
import datetime
import Data_handle_in_IPC
import Motor_global_vars

def extract_cn_timelist(motor_time_list, timelist_name,output_dir, device_number, record_info, essemble_data_file,update=False, ref_time=0):
    
    parquet_dir = os.path.join(output_dir, "timelist_data", "parquet")
    output_file_parquet = os.path.join(parquet_dir, f"motor_time_list{device_number}_{record_info}.parquet")
    csv_dir = os.path.join(output_dir, "timelist_data", "csv")
    output_file_csv = os.path.join(csv_dir, f"motor_time_list{device_number}_{record_info}.csv")
    # motor_time_list = pd.read_parquet(output_file_parquet).to_dict(orient='list')
    
    if timelist_name not in motor_time_list or update:
        motor_cn_timelist=[]
        motor_cnx_timelist=[]
        motor_cny_timelist=[]
        basic_amplist=[]
    else:
        motor_cn_timelist = motor_time_list[timelist_name]
        basic_amplist = motor_time_list["basic_freq_list"]
        motor_cnx_timelist = motor_time_list['Icn_y']
        motor_cny_timelist = motor_time_list['Icn_y']
        
    if not motor_cn_timelist:
        # extract the time list data of motor cn state
        essemble_data = pd.read_hdf(essemble_data_file)
        # iterate through each row in essemble_data DataFrame
        for index, data_read in essemble_data.iterrows():
            cn_sts, _, _ = Data_handle_in_IPC. get_cn_sts_list(data_read["Current alpha"], data_read["Current beta"], debug=False)
            cn_x= (cn_sts['Icn_x']-32767)/32768
            cn_y= (cn_sts['Icn_y']-32767)/32768
            cn_magnitude = np.sqrt(cn_x**2 + cn_y**2)
            motor_cn_timelist.append(cn_magnitude)
            motor_cnx_timelist.append(cn_x)
            motor_cny_timelist.append(cn_y)
            basic_amplist.append((cn_sts['I_rms']-32767)/32768*Motor_global_vars.Base_current)
            
        # update the time list data
        motor_time_list[timelist_name] = motor_cn_timelist
        motor_time_list['Icn_y'] = motor_cny_timelist
        motor_time_list["basic_freq_list"] = basic_amplist
        # Save as both CSV and Parquet
        df = pd.DataFrame(motor_time_list)
        df.to_csv(output_file_csv, index=False, float_format='%.6f')
        df.to_parquet(output_file_parquet)
        print(f'{timelist_name} update for device {device_number} record {record_info} done...')
        
    return motor_time_list