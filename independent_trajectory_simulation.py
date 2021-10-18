from incremental_framework import IncrementalLearner
from trajectory import Trajectory
from utils.config_reader import Config

import random


def run_experiment(config):
    random.seed(config.seed)

    traj_points_all = []
    traj_names = []

    # create trajectories and a list of data points corresponding to each trajectory
    for i in range(config.num_traj):
        traj = Trajectory()

        traj_points_all.append(traj.get_traj_points(add_noise=config.add_noise, gap=config.gap))
        traj_names.append("traj_{}".format(i + 1))

    # create the incremental learner
    incremental_learner = IncrementalLearner(config)

    # train the learner
    incremental_learner.train(traj_points_all, sim_name=config.sim_name,
                              traj_names=traj_names,
                              journal_visualizations=config.journal_visualizations,
                              results_dir=config.results_dir)


if __name__ == "__main__":
    config = Config()

    config.seed = 42
    config.model_name = "PCA"
    config.sim_name = "{}_sim1_2".format(config.model_name)
    config.journal_visualizations = False
    config.results_dir = "./results/simulation/"
    config.num_traj = 2

    config.add_noise = False
    config.gap = 0

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

    run_experiment(config)
