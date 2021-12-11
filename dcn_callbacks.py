from tensorflow.keras.callbacks import Callback

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import os
from typing import Sequence


class PlotClusters2D(Callback):
    """
    This callback can be used to visualize and save intermediate clustering results (embeddings+assignments).
    It only works correctly for 2d data.
    """

    def __init__(self, X_all: np.array, true_assignments: Sequence, save_dir: str, every_x_epoch=5, figsize=(15, 9)):
        """

        :param X_all: All datapoints to visualize at the end of an epoch.
        :param true_assignments: Vector of cluster assignments for points X_all.
            DCN's assignments are contrasted against the <true_assignments> in the visualization.
            Set to None if no "true assignments" are known.
        :param save_dir: Directory to save plots in. Plots are going to be saved under <save_dir>/epoch_<epoch>_deep_clustering.png
        :param every_x_epoch: Specifies how often plots are generated and saved.
        :param figsize: Figure size in output image. Directly passed to matplotlib.
        """
        super(PlotClusters2D, self).__init__()
        self.figsize = figsize
        self.X_all = X_all
        self.true_assignments = true_assignments
        self.every_x_epoch = every_x_epoch
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.every_x_epoch == 0:
            sns.set_theme()

            latent_x = self.model.encoder(self.X_all)
            deep_assignments = self.model.get_assignment(latent_x)

            fig, ax = plt.subplots(1, 2, figsize=self.figsize)
            f1 = sns.scatterplot(x=latent_x[:, 0],
                                 y=latent_x[:, 1],
                                 hue=deep_assignments,
                                 ax=ax[0]
                                 )
            ax[0].scatter(self.model.centers[:, 0],
                          self.model.centers[:, 1],
                          marker='x',
                          c='cyan',
                          s=50
                          )

            f1 = f1.set_title('Deep assignments after epoch {0}.'.format(epoch + 1))
            f2 = sns.scatterplot(x=latent_x[:, 0],
                                 y=latent_x[:, 1],
                                 hue=self.true_assignments,
                                 ax=ax[1],
                                 )
            ax[1].scatter(self.model.centers[:, 0],
                          self.model.centers[:, 1],
                          marker='x',
                          c='cyan',
                          s=50)
            f2 = f2.set_title('True assignments after epoch {0}.'.format(epoch + 1))

            save_path = os.path.join(self.save_dir, 'epoch_{0}_deep_clustering.png'.format(epoch + 1))
            fig.savefig(save_path)
            plt.close()
        else:
            pass


class ClustLearningRateUpdater(Callback):
    """
    This callback class resets the cluster update count after every epoch.
     The counts are used in the k-means batch algorithm to calculate the learning rate.
     Not resetting it leads to very small center updates quickly.
    """

    def __init__(self):
        super(ClustLearningRateUpdater, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        # this is going to be used for calculating the learning rate in the cluster center gradient step.
        self.model.cluster_count = [0] * self.model.n_clusters
