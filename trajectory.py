import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from song.song import SONG
from sklearn.utils import shuffle
from song.song import SONG
import umap
import seaborn as sns
import os
import models
import seaborn as sns

# num_dims = 100
# num_increments = 5
# max_time = 500
# points_per_increment = int(max_time / num_increments)
# gap = 25
# sim_factor = 0
# model_name = "SONG"
# noise = True

sim_name = "traj_1_15"


class Trajectory():
    """
    Generates a random independent trajectory
    """

    def __init__(self, num_dims=100, traj_equation=None):
        if traj_equation is None:
            self.coeffs = [random.random() for _ in range(num_dims)]
            self.constants = [random.random() for _ in range(num_dims)]
            self.powers = [random.randint(1, 2) for _ in range(num_dims)]

            self.traj_equation = lambda t: np.array(
                [(coeff * t + constant) ** power for coeff, constant, power in
                 zip(self.coeffs, self.constants, self.powers)])
        else:
            self.coeffs = traj_equation(1) - traj_equation(0)
            self.constants = traj_equation(0)
            self.powers = np.ones(len(traj_equation(0)))
            self.traj_equation = traj_equation

    def get_traj_name(self):
        """
        Iteratively creates the trajectory name
        """
        traj_equation = ""
        for coeff, constant, power in zip(self.coeffs, self.constants, self.powers):
            traj_equation += "({}*t+{})**{}".format(coeff, constant, power)

        return traj_equation

    def get_traj_points(self, max_time=500, num_increments=5, add_noise=False, gap=0):
        points_per_increment = int(max_time / num_increments)
        increments = np.arange(0, max_time, points_per_increment)

        traj_points = []
        for increment in increments:
            traj_increment_points = np.array([self.traj_equation(i) for i in range(increment, increment +
                                                                                   points_per_increment - gap)])
            if add_noise:
                traj_increment_points = traj_increment_points + np.random.normal(0,
                                                                                 np.std(traj_increment_points, axis=0),
                                                                                 traj_increment_points.shape) * 0.2

            traj_points.append(traj_increment_points)
        return np.array(traj_points)
