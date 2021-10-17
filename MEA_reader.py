import McsPy.McsData
from McsPy import ureg, Q_
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.signal import stft
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from utils.config_reader import Config

import os



def butter_highpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def data_reader(df, file_path, AD, glutamate, character, date, config):
    chunk_size = config.chunk_size  # 4sec
    channel_raw_data = McsPy.McsData.RawData(file_path)
    AD = AD
    print("-------------------------------------------------------------------")
    print(channel_raw_data.comment)
    print(channel_raw_data.date)
    print(channel_raw_data.clr_date)
    print(channel_raw_data.date_in_clr_ticks)
    print(channel_raw_data.file_guid)
    print(channel_raw_data.mea_name)
    print(channel_raw_data.mea_sn)
    print(channel_raw_data.mea_layout)
    print(channel_raw_data.program_name)
    print(channel_raw_data.program_version)
    print("-------------------------------------------------------------------")

    analog_stream_0 = channel_raw_data.recordings[0].analog_streams[0]  # change this to 1 if filtered is reqd

    analog_stream_0_data = analog_stream_0.channel_data

    np_analog_stream_0_data = np.transpose(analog_stream_0_data)

    np_analog_stream_0_data_length = len(np_analog_stream_0_data)

    channel_ids = analog_stream_0.channel_infos.keys()
    print("Channel ids", channel_ids)

    print("channel data", np_analog_stream_0_data)

    for channel_id in channel_ids:
        time = analog_stream_0.get_channel_sample_timestamps(channel_id, 0, np_analog_stream_0_data_length)

        # scale time to seconds:
        scale_factor_for_second = Q_(1, time[1]).to(ureg.s).magnitude
        time_in_sec = time[0] * scale_factor_for_second
        duration = time_in_sec[len(time_in_sec) - 1] - time_in_sec[0]

        # removing the Glutamate recordings
        glutamate_ts = 120 * 25000
        if date == "2019_07_24":
            glutamate_ts = 60 * 25000

        length = len(analog_stream_0.get_channel_in_range(channel_id, 0, np_analog_stream_0_data_length)[0])

        if glutamate == "pre" and character == "GLU":
            # if a recording with glutamate and pre glutamate is needed: get the segment before glutamate_ts
            mea_signal = analog_stream_0.get_channel_in_range(channel_id, 0, np_analog_stream_0_data_length)[0][
                         :glutamate_ts - chunk_size * 5]
        elif glutamate == "post" and character == "GLU" and length > glutamate_ts + chunk_size * 4:
            # if a recording with glutamate and post glutamate needed: get the segment after glutamate_ts
            mea_signal = analog_stream_0.get_channel_in_range(channel_id, 0, np_analog_stream_0_data_length)[0][
                         glutamate_ts + chunk_size * 4:]
        elif glutamate == "post" and character == "GLU" and length <= glutamate_ts + chunk_size * 4:
            # if a recording with glutamate and post glutamate needed: test for signal length
            continue
        else:
            # not a recording with glutamate: process the whole signal
            mea_signal = analog_stream_0.get_channel_in_range(channel_id, 0, np_analog_stream_0_data_length)[0]

        signal_chunks = []

        # create completed signal chunks
        for i in range(0, len(mea_signal), chunk_size):
            signal_chunk = mea_signal[i:i + chunk_size]
            if len(signal_chunk) == chunk_size:
                signal_chunks.append(signal_chunk)
        print("Num of samples ", len(signal_chunks))

        for signal_chunk in signal_chunks:
            signal_chunk_filtered = butter_highpass_filter(signal_chunk, config.cut_off, config.sampling_frequency, 3)
            fft_signal = [abs(i) for i in fft(signal_chunk_filtered)][:int(chunk_size / 2)]  # upto 300Hz
            downsampled = np.mean(np.array(fft_signal).reshape(-1, 50), axis=1)

            row = pd.Series(data={
                'channel_id': file_path + str(channel_id),
                'fft': downsampled,
                'label': AD,
                'glutamate': glutamate,
            },
                name=len(df.index))

            df = df.append(row)
    print("-------------------------------------------------------------------")

    return df


def run_experiment(config):
    glutamate = config.glutamate

    for j in range(0, len(file_lists)):

        for file in file_lists[j]:
            print("processing {}".format(file))
            df = pd.DataFrame(
                columns=['channel_id', 'fft', 'label', "glutamate"]
            )
            date = file[0:10].replace("-", "_")
            ts = file[10:19]

            folder = date + "_h5"
            file = file + ".h5"
            path = os.path.join(config.h5_path, folder, file)

            label = j
            if classes[j][-3:] == "GLU" and glutamate == "pre":
                print("pre glutamate identified")
                label = j - 2
            #"6846":0
            #"63:1
            #"6848":2
            #"98":3
            df = data_reader(df, path, label, glutamate, classes[j][-3:], date, config)
            if len(df) > 0:
                df.to_pickle(
                    "./data/dataframes_pre_glut_with_margin/" + date + "_hfc_fft_100k_pre_glu_overlap_none_avg/" + str(
                        classes[label]) + file)


CL_6846_BASE = [
    # '2019-08-26T09-52-276846_C_base_2_A',
    # '2019-08-26T09-52-276846_C_base_2_B',
    # '2019-08-26T09-52-276846_C_base_2_C',
    # '2019-08-26T09-52-276846_C_base_2_D',
    # '2019-08-26T09-52-276846_C_base_2_E',
    # '2019-08-26T09-52-276846_C_base_2_F',
    #
    # "2019-09-27T10-24-59_6well_6846_A_base_2_A",
    # "2019-09-27T10-24-59_6well_6846_A_base_2_B",
    # "2019-09-27T10-24-59_6well_6846_A_base_2_C",
    # "2019-09-27T10-24-59_6well_6846_A_base_2_D",
    # "2019-09-27T10-24-59_6well_6846_A_base_2_E",
    # "2019-09-27T10-24-59_6well_6846_A_base_2_F",
]
CL_63_BASE = [
    # "2019-07-24T15-51-496-well_6-3_A_A",
    # "2019-07-24T15-51-496-well_6-3_A_B",
    # "2019-07-24T15-51-496-well_6-3_A_C",
    # "2019-07-24T15-51-496-well_6-3_A_D",
    # "2019-07-24T15-51-496-well_6-3_A_E",
    # "2019-07-24T15-51-496-well_6-3_A_F",
    #
    # "2019-08-26T10-20-526-3_A_base_2_A",
    # "2019-08-26T10-20-526-3_A_base_2_B",
    # "2019-08-26T10-20-526-3_A_base_2_C",
    # "2019-08-26T10-20-526-3_A_base_2_D",
    # "2019-08-26T10-20-526-3_A_base_2_E",
    # "2019-08-26T10-20-526-3_A_base_2_F",
    #
    # "2019-09-27T10-58-53_6well_6-3_A_base_2_A",
    # "2019-09-27T10-58-53_6well_6-3_A_base_2_B",
    # "2019-09-27T10-58-53_6well_6-3_A_base_2_C",
    # "2019-09-27T10-58-53_6well_6-3_A_base_2_D",
    # "2019-09-27T10-58-53_6well_6-3_A_base_2_E",
    # "2019-09-27T10-58-53_6well_6-3_A_base_2_F",
]
CL_98_BASE = [
    #   "2019-07-24T16-26-386-well_9-8_A_A",
    #   "2019-07-24T16-26-386-well_9-8_A_B",
    #   "2019-07-24T16-26-386-well_9-8_A_C",
    #   "2019-07-24T16-26-386-well_9-8_A_D",
    #   "2019-07-24T16-26-386-well_9-8_A_E",
    #   "2019-07-24T16-26-386-well_9-8_A_F",
    #
    # "2019-08-26T10-21-439-8_A_base_A",
    # "2019-08-26T10-21-439-8_A_base_B",
    # "2019-08-26T10-21-439-8_A_base_C",
    # "2019-08-26T10-21-439-8_A_base_D",
    # "2019-08-26T10-21-439-8_A_base_E",
    # "2019-08-26T10-21-439-8_A_base_F",
    #
    # "2019-09-27T10-59-27_6well_9-8_A_base_A",
    # "2019-09-27T10-59-27_6well_9-8_A_base_B",
    # "2019-09-27T10-59-27_6well_9-8_A_base_C",
    # "2019-09-27T10-59-27_6well_9-8_A_base_D",
    # "2019-09-27T10-59-27_6well_9-8_A_base_E",
    # "2019-09-27T10-59-27_6well_9-8_A_base_F",
]
CL_6848_BASE = [
    # '2019-08-26T09-54-456848_C_base_A',
    # '2019-08-26T09-54-456848_C_base_B',
    # '2019-08-26T09-54-456848_C_base_C',
    # '2019-08-26T09-54-456848_C_base_D',
    # '2019-08-26T09-54-456848_C_base_E',
    # '2019-08-26T09-54-456848_C_base_F',
    #
    #
    # "2019-09-27T10-26-11_6well_6848_A_base_A",
    # "2019-09-27T10-26-11_6well_6848_A_base_B",
    # "2019-09-27T10-26-11_6well_6848_A_base_C",
    # "2019-09-27T10-26-11_6well_6848_A_base_D",
    # "2019-09-27T10-26-11_6well_6848_A_base_E",
    # "2019-09-27T10-26-11_6well_6848_A_base_F",

]

CL_63_GLU = [
    # "2019-07-24T16-16-496-well_6-3_A+Glu1min_A",
    # "2019-07-24T16-21-186-well_6-3_B+Glu1min_B",
    # "2019-07-24T16-21-186-well_6-3_B+Glu1min_C", #remove for post glutamate
    # "2019-07-24T16-21-186-well_6-3_B+Glu1min_D",#remove for post glutamate
    # "2019-07-24T16-21-186-well_6-3_B+Glu1min_E",#remove for post glutamate
    # "2019-07-24T16-21-186-well_6-3_B+Glu1min_F",#remove for post glutamate
    # #
    # "2019-08-27T11-10-46_6well_6-3_25uMglu_2_A",
    # "2019-08-27T11-17-42_6well_6-3_B_25uMglu_2_B",
    # "2019-08-27T11-23-13_6well_6-3_C_25uMglu_2_C",
    # "2019-08-27T11-28-27_6well_6-3_D_25uMglu_2_D",
    # "2019-08-27T11-33-49_6well_6-3_E_25uMglu_2_E",
    # "2019-08-27T11-39-02_6well_6-3_F_25uMglu_2_F",
    #
    # "2019-09-02T10-46-22_6well_6-3_A_62.5uMglu_A",
    # "2019-09-02T10-53-34_6well_6-3_B_62.5uMglu_B",
    # "2019-09-02T10-58-56_6well_6-3_C_62.5uMglu_C",
    # "2019-09-02T11-04-18_6well_6-3_D_62.5uMglu_D",
    # "2019-09-02T11-09-57_6well_6-3_E_62.5uMglu_E",
    # "2019-09-02T11-15-09_6well_6-3_F_62.5uMglu_F",

    "2019-11-21T09-28-48_6well_6-3_A_glu_2_A",#2.30-4.30
    "2019-11-21T09-35-47_6well_6-3_B_glu_2_B",#2.00
    "2019-11-21T09-40-16_6well_6-3_C_glu_2_C",#2.00
    "2019-11-21T09-44-50_6well_6-3_D_glu_2_D",#2.20
    "2019-11-21T09-49-06_6well_6-3_E_glu_2_E",#1.30
    "2019-11-21T09-52-59_6well_6-3_F_glu_2_F",#1.30
]

CL_98_GLU = [
    #
    # "2019-07-24T16-51-276-well_9-8_C+Glu1min_C",
    # "2019-07-24T16-56-086-well_9-8_D+Glu1min_D",
    # "2019-07-24T16-56-086-well_9-8_D+Glu1min_E",#remove for post glutamate
    # "2019-07-24T16-56-086-well_9-8_D+Glu1min_F",#remove for post glutamate
    # "2019-07-24T16-56-086-well_9-8_D+Glu1min_A",#remove for post glutamate
    # "2019-07-24T16-56-086-well_9-8_D+Glu1min_B",#remove for post glutamate
    #
    #
    # "2019-08-27T11-11-10_6well_9-8_25uMglu_A",
    # "2019-08-27T11-17-50_6well_9-8_B_25uMglu_B",
    # "2019-08-27T11-23-40_6well_9-8_C_25uMglu_C",
    # "2019-08-27T11-28-56_6well_9-8_D_25uMglu_D",
    # "2019-08-27T11-34-12_6well_9-8_E_25uMglu_E",
    # "2019-08-27T11-39-34_6well_9-8_F_25uMglu_F",

    # "2019-08-27T12-03-27_6well_9-8_A_62.5uMglu_A",    #should be removed
    # "2019-08-27T12-08-52_6well_9-8_B_62.5uMglu_B",
    # "2019-08-27T12-14-09_6well_9-8_C_62.5uMglu_C",
    # "2019-08-27T12-19-28_6well_9-8_D_62.5uMglu_D",
    # "2019-08-27T12-24-40_6well_9-8_E_62.5uMglu_E",
    # "2019-08-27T12-29-55_6well_9-8_F_62.5uMglu_F",

    # "2019-09-02T10-40-04_6well_9-8_A_62.5uMglu_2_A",
    # "2019-09-02T10-45-34_6well_9-8_B_62.5uMglu_2_B",
    # "2019-09-02T10-51-08_6well_9-8_C_62.5uMglu_2_C",
    # "2019-09-02T10-56-25_6well_9-8_D_62.5uMglu_2_D",
    # "2019-09-02T11-01-43_6well_9-8_E_62.5uMglu_2_E",
    # "2019-09-02T11-07-02_6well_9-8_F_62.5uMglu_2_F",


    # "2019-11-21T09-29-32_6well_9-8_A_glu_A",#2.30-4.30
    # "2019-11-21T09-45-18_6well_9-8_B_glu_B",#2.00
    # "2019-11-21T09-40-58_6well_9-8_C_glu_C",#3.00
    # "2019-11-21T09-48-38_6well_9-8_D_glu_D",#1.30
    # "2019-11-21T09-52-25_6well_9-8_E_glu_E",#1.30
    # "2019-11-21T09-56-22_6well_9-8_F_glu_F",#1.00
    "2019-11-21T09-29-32_6well_9-8_A_glu_B", #to increase the number of samples
    "2019-11-21T09-29-32_6well_9-8_A_glu_C",#to increase the number of samples
    "2019-11-21T09-29-32_6well_9-8_A_glu_D",#to increase the number of samples
    "2019-11-21T09-29-32_6well_9-8_A_glu_E",#to increase the number of samples
    "2019-11-21T09-29-32_6well_9-8_A_glu_F",#to increase the number of samples
]
CL_6848_GLU = [
    # "2019-09-02T11-29-07_6well_6848_A_62.5uMglu_2_A",
    # "2019-09-02T11-34-19_6well_6848_B_62.5uMglu_2_B",
    # "2019-09-02T11-40-32_6well_6848_C_62.5uMglu_2_C",
    # "2019-09-02T11-46-20_6well_6848_D_62.5uMglu_2_D",
    # "2019-09-02T11-52-11_6well_6848_E_62.5uMglu_2_E",
    # "2019-09-02T11-57-58_6well_6848_F_62.5uMglu_2_F",
#     #

    # "2019-11-21T10-01-09_6well_6848_C_glu_C",#1.30
    # "2019-11-21T10-05-04_6well_6848_D_glu_D",#1.30
    # "2019-11-21T10-05-04_6well_6848_D_glu_E",#remove for post glutamate
    # "2019-11-21T10-05-04_6well_6848_D_glu_F",#remove for post glutamate
    # "2019-11-21T10-05-04_6well_6848_D_glu_A",#remove for post glutamate
    # "2019-11-21T10-05-04_6well_6848_D_glu_B",#remove for post glutamate
    "2019-11-21T10-01-09_6well_6848_C_glu_A", #to increase the number of samples
    "2019-11-21T10-01-09_6well_6848_C_glu_B", #to increase the number of samples
    "2019-11-21T10-01-09_6well_6848_C_glu_E", #to increase the number of samples
    "2019-11-21T10-01-09_6well_6848_C_glu_F", #to increase the number of samples
]

CL_6846_GLU = [
    # "2019-08-27T12-03-10_6well_6846_A_62.5uMglu_2_A",
    # "2019-08-27T12-08-30_6well_6846_B_62.5uMglu_2_B",
    # "2019-08-27T12-13-43_6well_6846_C_62.5uMglu_2_C",
    # "2019-08-27T12-19-06_6well_6846_D_62.5uMglu_2_D",
    # "2019-08-27T12-24-20_6well_6846_E_62.5uMglu_2_E",
    # "2019-08-27T12-29-37_6well_6846_F_62.5uMglu_2_F",
    # #
    # "2019-09-02T11-29-45_6well_6846_A_62.5uMglu_A",
    # "2019-09-02T11-34-58_6well_6846_B_62.5uMglu_B",
    # "2019-09-02T11-41-02_6well_6846_C_62.5uMglu_C",
    # "2019-09-02T11-46-52_6well_6846_D_62.5uMglu_D",
    # "2019-09-02T11-52-28_6well_6846_E_62.5uMglu_E",
    # "2019-09-02T11-58-17_6well_6846_F_62.5uMglu_F",
    #
    # "2019-11-21T10-00-26_6well_6846_A_glu_2_A",#1.30
    # "2019-11-21T10-00-26_6well_6846_A_glu_2_B",#remove for post glutamate
    # "2019-11-21T10-00-26_6well_6846_A_glu_2_C",#remove for post glutamate
    # "2019-11-21T10-04-30_6well_6846_D_glu_2_D",
    # "2019-11-21T10-08-52_6well_6846_E_glu_2_E",#1.30
    # "2019-11-21T10-12-10_6well_6846_F_glu_2_F",
    "2019-11-21T10-00-26_6well_6846_A_glu_2_B", #to increase the number of samples
    "2019-11-21T10-00-26_6well_6846_A_glu_2_C", #to increase the number of samples
    "2019-11-21T10-00-26_6well_6846_A_glu_2_D", #to increase the number of samples
    "2019-11-21T10-00-26_6well_6846_A_glu_2_E", #to increase the number of samples
    "2019-11-21T10-00-26_6well_6846_A_glu_2_F", #to increase the number of samples
]

file_lists = []
file_lists.append(CL_6846_BASE)
file_lists.append(CL_63_BASE)
file_lists.append(CL_6846_GLU)
file_lists.append(CL_63_GLU)
file_lists.append(CL_6848_BASE)
file_lists.append(CL_98_BASE)
file_lists.append(CL_6848_GLU)
file_lists.append(CL_98_GLU)

classes = ['CL_6846_BASE', 'CL_63_BASE', 'CL_6846_GLU', 'CL_63_GLU', 'CL_6848_BASE', 'CL_98_BASE', 'CL_6848_GLU',
           'CL_98_GLU']

if __name__ == "__main__":
    config = Config()

    config.glutamate = "pre"  # use "post" for post-glutamate recordings
    config.chunk_size = 100000  # 4s
    config.cut_off = 300
    config.sampling_frequency = 25000
    config.h5_path = "/media/tmalepathira/Tamasha/h5files/"

    run_experiment(config)
