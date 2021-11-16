# IncrementalFramework

This repository contains the code related to the paper "An Incrementally Learned Visualization Framework for Longitudinal Multi-Electrode Array Experiments".

	1.incremental_framework.py
The code related to the incremental framework which can be used to incrementally train the model and generate partial visualizations at each session. This framework is used with both simulated and experimental data in scripts 2,3 and 4.

	2.independent_trajectory_simulation.py
Conains the functions for indepedent trajectory simulation corresponding to the Figures 2 and 3A in the paper. The number of trajectories, the noise and gaps can be varied within the script for future experiments. The dimension reduction technique is set to "SONG" by default, but can also be chnaged to either "PCA" or "UMAP" for obtain the correponding visualziations.
	
	3.relative_trajectory_simulation.py
Contains the functions for relative trajectory progression simulation corresponding to Figures 3B and C in the paper. similarity_factor could be varied to control the similarity between the secondary trajectories and the principal trajectories.

	4.experimental_data.py
Contains the functions corresponding to the experimental dataset of Alzheimer's and control organoids. By controlling the wells_to_read parameter, you can obtain visualizations for either for a single organoid (Figure 4) or multiple organoids of the same cell line(Figure 5).

	5.trajectory.py
Contains the functions to simulate random independent trajectories needed for scripts 2 and 3.

	6.MEA_reader.py
Contians the functios to preprocess the MEA data. The MEA recordings were segmented, high pass filtered, and then Fast Fourier Transformed.



	
