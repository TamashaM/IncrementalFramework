from incremental_framework import IncrementalLearner
from trajectory import Trajectory
from utils.config_reader import Config
from utils.traj_utils import get_traj_name

import random
import numpy as np


def create_similar_trajectories(traj_1, traj_2, similarity_factor):
    """
    Create two new trajectories combining traj_1 and traj_2 based on the similarity factor.
    traj_3 = traj_1 * similarity_factor + traj_2 * (1-similarity_factor)
    traj_4 = traj_2 * similarity_factor + traj_1 * (1-similarity_factor)

    :return: traj_3, traj_4
    """
    inv_similarity_factor = 1 - similarity_factor

    # create trajectory 3 similar to 1 by a similarity factor
    traj_3_1 = lambda f_1, f_2: lambda t: np.array(
        [(i * similarity_factor + j * inv_similarity_factor) for i, j in zip(f_1(t), f_2(t))])
    # create trajectory 4 similar to 2 by a similarity factor
    traj_4_2 = lambda f_1, f_2: lambda t: np.array(
        [(i * inv_similarity_factor + j * similarity_factor) for i, j in zip(f_1(t), f_2(t))])

    traj_3 = Trajectory(traj_equation=traj_3_1(traj_1, traj_2))
    traj_4 = Trajectory(traj_equation=traj_4_2(traj_1, traj_2))

    return traj_3, traj_4


def run_experiment(config):
    random.seed(config.seed)

    # create trajectories
    traj_1 = Trajectory()
    traj_2 = Trajectory()

    # create two similarity trajectories
    traj_3, traj_4, = create_similar_trajectories(traj_1.traj_equation, traj_2.traj_equation,
                                                  config.similarity_factor)

    # create a list of data points corresponding to each trajectory
    traj_points_all = [traj_1.get_traj_points(add_noise=config.add_noise, gap=config.gap),
                       traj_2.get_traj_points(add_noise=config.add_noise, gap=config.gap),
                       traj_3.get_traj_points(add_noise=config.add_noise, gap=config.gap),
                       traj_4.get_traj_points(add_noise=config.add_noise, gap=config.gap)]

    # create the incremental learner
    incremental_learner = IncrementalLearner(config)

    # train the learner
    incremental_learner.train(traj_points_all, sim_name=config.sim_name,
                              traj_names=["traj_1", "traj_2", "traj_3", "traj_4"],
                              journal_visualizations=config.journal_visualizations,
                              results_dir=config.results_dir)


if __name__ == "__main__":
    config = Config()

    config.seed = 42
    config.model_name = "SONG"
    config.sim_name = "{}_sim2".format(config.model_name)
    config.journal_visualizations = True
    config.results_dir = "./results/simulation/"

    config.add_noise = False
    config.gap = 0
    config.similarity_factor = 0.8

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
