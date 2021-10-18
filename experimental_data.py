import pandas as pd
import numpy as np

import random
import os

from incremental_framework import IncrementalLearner
from utils.config_reader import Config


def reshape(array):
    fft_features_real_array = np.array(array)
    print(fft_features_real_array.shape)  # (n, 2400)

    n_samples = fft_features_real_array.shape[0]
    length = fft_features_real_array.shape[1]
    n_channels = 9
    n_overall_samples = int(n_samples / n_channels)

    reshaped_array = fft_features_real_array.reshape(n_overall_samples, n_channels, length, order='F')
    print(reshaped_array.shape)  # (n/9, 9, 2400)

    swapped_axes_array = np.swapaxes(reshaped_array, 1, 2)
    print(swapped_axes_array.shape)  # (n/9, 2400, 9)

    swapped_axes_array = swapped_axes_array.reshape(swapped_axes_array.shape[0],
                                                    swapped_axes_array.shape[1] * swapped_axes_array.shape[2])

    return swapped_axes_array


def data_loader_new(wells_to_read, cell_line):
    cell_lines = ["6846", "63", "6848", "98"]

    labels_new_all = []
    fft_features_real_all = []
    days = ["2019_07_24", "2019_08_26", "2019_08_27", "2019_09_02", "2019_09_27", "2019_11_21"]

    for well_read in wells_to_read:
        labels_new_well = []
        fft_features_real_well = []
        for date in days:
            labels_new_increment = []
            fft_features_real_increment = []
            directory = "./data_final/" + date + "_hfc_fft_100k_pre_glu_overlap_none_avg/"

            files = []
            files_filtered = []

            for r, d, f in os.walk(directory):
                for file in f:
                    files.append(file)
            for f in files:
                cell_line_file = f.split("_")[1]
                if (f[-1] == well_read) and (cell_line_file == cell_line):
                    files_filtered.append(f)
                    print(f[-1], cell_line_file, "Selected")

            for f in files_filtered:
                df = pd.read_pickle(os.path.join(directory, f))
                if len(df.index) == 0:
                    print("length 0 df identified")
                    continue
                cell_line_file = f.split("_")[1]
                well = f[-1]
                cell_line_index = cell_lines.index(cell_line_file)

                if well == "D":
                    cell_line_index = cell_line_index + 4
                elif well == "E":
                    cell_line_index = cell_line_index + 8
                elif well == "F":
                    cell_line_index = cell_line_index + 12
                elif well == "A":
                    cell_line_index = cell_line_index + 16
                elif well == "B":
                    cell_line_index = cell_line_index + 20

                fft_features = df['fft'].tolist()
                length = len(fft_features)
                labels_new_increment.extend([cell_line_index] * int(length / 9))
                fft_features_real_increment.extend(reshape(fft_features))
            if len(fft_features_real_increment)>0:
                labels_new_well.append(labels_new_increment)
                fft_features_real_well.append(fft_features_real_increment)
        labels_new_all.append(labels_new_well)
        fft_features_real_all.append(fft_features_real_well)

    return fft_features_real_all, labels_new_all


def learn_model_new(config):
    fft_features_real_all, labels_all = data_loader_new(config.wells_to_read, config.cell_line)
    incremental_learner = IncrementalLearner(config)

    incremental_learner.train(np.array(fft_features_real_all), sim_name=config.sim_name,
                              traj_names=config.wells_to_read,
                              journal_visualizations=config.journal_visualizations,
                              results_dir=config.results_dir)


if __name__ == "__main__":
    config = Config()

    config.seed = 42
    config.model_name = "SONG"
    config.sim_name = "{}_exp".format(config.model_name)
    config.journal_visualizations = True
    config.results_dir = "./results/experimental/"

    config.wells_to_read = ["C", "D", "E", "F"]
    config.cell_line = "63"

    if config.model_name == "SONG":
        config.n_components = 2
        config.n_neighbors = 6
        config.spread_factor = 0.8
        config.lr = 1
        config.agility = 0.3
        config.min_dist = 0.1
        config.final_vector_count = 100
        config.epsilon = 1 - 1e-10
        config.max_age = 1
        config.so_steps = 20

    learn_model_new(config)
