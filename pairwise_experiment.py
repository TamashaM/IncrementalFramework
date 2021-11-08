import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
import seaborn as sns


def draw_pairwise_graph(original_points, embedded_points, traj_name):

    data_pd = pairwise_distances(original_points)
    data_df = pd.DataFrame(np.array(data_pd).flatten())
    print(data_df)

    embedding_pd = pairwise_distances(embedded_points)
    embedding_df = pd.DataFrame(np.array(embedding_pd).flatten())
    print(embedding_df)

    max = data_df.iloc[:, -1].max()
    min = data_df.iloc[:, -1].min()
    gap = (max - min) / 50

    bin_labels = [i * gap for i in range(50)]
    bins = pd.cut(data_df.iloc[:, -1], bin_labels)
    grouped_embedding_df = embedding_df.groupby(bins)

    data_array = []
    for name, grouped in grouped_embedding_df:
        print(len(grouped.index))
        data_array.append(grouped.iloc[:, -1].tolist())

    pearson = pearsonr(np.array(data_pd).flatten(), np.array(embedding_pd).flatten())

    fig, axes = plt.subplots(2)
    g = sns.boxplot(data=data_array, ax=axes[0], linewidth=1, fliersize=0, color='white')

    g.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False)

    g.set_title("r = {}".format(round(pearson[0], 2)))

    hist_plot = sns.histplot(data_df, bins=50, ax=axes[1], edgecolor='black', color="black", legend=False)
    hist_plot.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=True,
        labelleft=False)
    hist_plot.set_xlabel("original distance")

    image_path = "./results/simulation/pairwise/{}".format(traj_name)
    plt.savefig(image_path + ".png", bbox_inches='tight')
    plt.savefig(image_path + ".svg", bbox_inches='tight')
    plt.show()
