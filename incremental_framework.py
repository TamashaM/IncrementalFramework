import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import models

import os
import itertools

from utils.plot_utils import get_s


class IncrementalLearner:
    def __init__(self, config):
        # TODO separate model parameters
        self.model_name = config.model_name
        if self.model_name == "SONG":
            self.model = models.create(self.model_name,
                                       n_neighbors=config.n_neighbors,
                                       spread_factor=config.spread_factor,
                                       lr=config.lr,
                                       agility=config.agility,
                                       n_components=config.n_components,
                                       min_dist=config.min_dist,
                                       final_vector_count=config.final_vector_count,
                                       max_age=config.max_age,
                                       epsilon=config.epsilon,
                                       so_steps=config.so_steps,
                                       )
        else:
            self.model = models.create(self.model_name)
        self.data = None
        self.num_increments = None

    def train(self, traj_points_all, results_dir=None, sim_name=None, traj_names=None,
              journal_visualizations=False):

        coordinate_dict = {
        }
        self.num_increments = traj_points_all[0].shape[0]

        subplot_x = 1
        subplot_y = self.num_increments
        fig, axs = plt.subplots(subplot_x, subplot_y, sharex=True, sharey=True, figsize=(15, 2), num=1)
        axs = axs.flatten()

        indices = np.arange(self.num_increments)
        num_traj = len(traj_points_all)

        for *increment_trajs, index in zip(*traj_points_all, indices):

            if self.data is None:
                self.data = [[], []]

                for i, increment_traj in enumerate(increment_trajs):
                    self.data[0].extend(increment_traj)
                    self.data[1].extend([index * num_traj + (i + 1)] * len(increment_traj))

            else:
                for i, increment_traj in enumerate(increment_trajs):
                    self.data[0].extend(increment_traj)
                    self.data[1].extend([index * num_traj + i] * len(increment_traj))

            features = np.array(self.data[0])
            labels = np.array(self.data[1])

            x_name = self.model_name + "1"
            y_name = self.model_name + "2"

            Y = self.model.fit_transform(features).T
            df = pd.DataFrame({x_name: Y[0], y_name: Y[1], 'label': labels})

            file_path = os.path.join(results_dir, "{}{}.pkl".format(sim_name, index))

            if not os.path.exists(file_path):
                df.to_pickle(file_path)

            color_palette = sns.color_palette("colorblind")

            if journal_visualizations:
                cmaps = iter([plt.cm.Blues,
                              sns.light_palette(color_palette[1], as_cmap=True),
                              sns.light_palette(color_palette[2], as_cmap=True),
                              sns.light_palette(color_palette[4], as_cmap=True),
                              ])
                markers = iter(["o", "*", "^", "+"])

            else:
                cmaps = itertools.cycle([sns.light_palette(i, as_cmap=True) for i in color_palette])
                markers = itertools.cycle(list(Line2D.markers.keys()))

            for i, increment_traj in enumerate(increment_trajs):
                print(i)
                color = next(cmaps)
                marker = next(markers)
                df_traj = df.loc[df['label'].isin(np.arange(i, num_traj * self.num_increments, num_traj))]

                x_centroids = df_traj.groupby(['label']).median()[x_name].tolist()
                y_centroids = df_traj.groupby(['label']).median()[y_name].tolist()

                x_y_coordinates = [[x, y] for x, y in zip(x_centroids, y_centroids)]
                coordinate_dict[traj_names[i]] = x_y_coordinates

                axs[index].scatter(df_traj[x_name], df_traj[y_name],
                                   c=df_traj["label"],
                                   cmap=color,
                                   s=get_s(df_traj, x_name, y_name),
                                   label =traj_names[i],
                                   vmin=-2,
                                   marker=marker,
                                   vmax=num_traj * self.num_increments,
                                   alpha=1)

                for j in range(len(x_centroids) - 1):
                    axs[index].annotate("", xytext=(x_centroids[j], y_centroids[j]),
                                        xy=(x_centroids[j + 1], y_centroids[j + 1]),
                                        arrowprops=dict(arrowstyle="->", color=color(0.8)))

                if index == len(indices) - 1:
                    df_traj.sort_values(by=['label']).to_csv(os.path.join(results_dir, "{}.csv".format(traj_names[i])),
                                                             index=False)
        plt.savefig(os.path.join(results_dir, "{}.png".format(sim_name)), bbox_inches='tight')
        plt.savefig(os.path.join(results_dir, "{}.svg".format(sim_name)), bbox_inches='tight')
        plt.legend()
        plt.show()
