from datetime import datetime as dt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import initializers

from sklearn.cluster import KMeans
from scipy.spatial import distance as sci_dist


class DCN(Model):
    """
    This is an implementation of the clustering approach introduced in "Towards K-means-friendly Spaces: Simultaneous Deep Learning and Clustering" (http://arxiv.org/abs/1610.04794).
    It uses self-supervised learning with a custom loss function to learn a mapping of the input data (from the observed space) to a k-means friendly latent space.
    That is, a representation that allows the k-means algorithm to identify meaningful clusters in data that is hard to cluster in the observed space.
    """

    def __init__(self, latent_dim: int, input_dim: int, lamda: float, auto_encoder_dims=None, n_clusters=4,
                 activation='relu'):
        """
        Initializes the DCN network: Creates the auto-encoder network and creates a k-means model.
        Note that the k-means model is not fit until the first call of train_step().

        :param latent_dim: Dimension of the latent space.
        :param input_dim: Number of examples in the input data.
        :param lamda: Coefficient of the clustering loss. Used to weigh it against the reconstruction loss.
        :param auto_encoder_dims: (optional). Neurons in the hidden layers in the auto-encoder. Must be provided as list of ints.
         The dimensions only describe the encoder part. For the decoder part, this is mirrored.
         Defaults to [100, 50, 10] such that the dimensions for the full auto encoder would look like this: input_dim -> 100 -> 50 -> 10 -> latent_dim -> 10 -> 50 -> 100 -> input_dim
        :param n_clusters: (optional). Number of clusters. Defaults to 4.
        :param activation: (optional). Activation function in the network. Everything that is accepted by keras is ok. Defaults to 'relu'.
        """
        super(DCN, self).__init__()

        ### AUTO-ENCODER PARAMS
        if auto_encoder_dims is None:
            auto_encoder_dims = [100, 50, 10]
        self.kernel_initializer = initializers.HeNormal(seed=43)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        ### DEFINE AUTO-ENCODER
        ##### ENCODER
        encoder_layers = [layers.Input(shape=(input_dim,)),
                          layers.BatchNormalization(axis=-1)]
        for layer_no, dim in enumerate(auto_encoder_dims):
            encoder_layers.append(layers.Dense(dim,
                                               activation=activation,
                                               kernel_initializer=self.kernel_initializer,
                                               name='encoder_dense_{0}'.format(layer_no)))
            encoder_layers.append(layers.BatchNormalization(axis=-1))
        encoder_layers.append(layers.Dense(latent_dim))
        self.encoder = tf.keras.Sequential(encoder_layers)
        ##### DECODER
        decoder_layers = [layers.Input(shape=(latent_dim,)),
                          layers.BatchNormalization(axis=-1)]
        for layer_no, dim in enumerate(reversed(auto_encoder_dims)):
            decoder_layers.append(layers.Dense(dim,
                                               activation=activation,
                                               kernel_initializer=self.kernel_initializer,
                                               name='decoder_dense_{0}'.format(layer_no)))
            decoder_layers.append(layers.BatchNormalization(axis=-1))
        decoder_layers.append(layers.Dense(input_dim))
        self.decoder = tf.keras.Sequential(decoder_layers)

        ### CLUSTER PARAMS
        self.n_clusters = n_clusters
        self.data_dim = latent_dim
        self.centers = None
        self.cluster_count = [0] * self.n_clusters
        self.lamda = lamda  # to weigh cluster loss (against reconstruction loss)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=43)

    def init_centers(self, X):
        """
        We use sklearn's implementation of k-means++ to initialize the cluster centers.
        """

        init_km_model = KMeans(n_clusters=self.n_clusters, random_state=43, init='k-means++')
        init_km_model.fit(X)

        self.centers = init_km_model.cluster_centers_.copy()

    def update_center(self, x, assigned_center):
        """
        Update cluster centers according to equation (8) in orig. paper, i.e., via batch k-means.
        """

        self.cluster_count[assigned_center] += 1
        learning_rate = 1.0 / self.cluster_count[assigned_center]
        old_center = self.centers[assigned_center]

        self.centers[assigned_center] = (1 - learning_rate) * old_center + learning_rate * x

    def get_assignment(self, X):
        """
        Returns vector of cluster labels for all points in X, i.e., label of closest cluster centers.
        i-th entry corresponds to X[i].
        """
        return sci_dist.cdist(X, self.centers).argmin(
            axis=1)  # this gives a vector of assignments, where the i-th entry corresponds to point i's label

    def _loss(self, x, y_true, nearest_center_label):
        """
        Calculates the two parts of equation (5) in orig. paper, i.e. reconstruction loss and cluster loss.
        """
        x_latent = self.encoder(x)
        x_recon = self.decoder(x_latent)
        clustering_loss = 0  # tf.constant(0, dtype='float32')
        for x_point, assigned_center in zip(x_latent, nearest_center_label):
            clustering_loss += tf.norm(x_point - self.centers[assigned_center],
                                       ord='euclidean')

        clustering_loss = tf.constant(self.lamda * clustering_loss, dtype='float32')
        recon_loss = tf.norm(x_recon - y_true, ord='euclidean')  # tf.constant(0, dtype='float32')

        return recon_loss, clustering_loss

    def pretrain(self, data, epochs, batch_size=32, verbose=False):
        """
        The paper suggests to pretrain the auto-encoder part of the model.
        This bit only trains the auto-encoder bit of the model and ignores the "cluster-friendliness" of the representation.
        """
        # create copy because we are going to shuffle the data
        data_pretrain = data.copy()

        for _epoch in range(epochs):

            # get batches
            n_batches = int(len(data_pretrain) / batch_size)

            batches = [data_pretrain[index * batch_size:(index + 1) * batch_size] for index in range(n_batches)]

            # for each batch, perform forward pass and gradient step
            for batch in batches:
                with tf.GradientTape() as tape:
                    # get encoding
                    x_latent = self.encoder(batch)
                    # get decoding
                    x_recon = self.decoder(x_latent)
                    # calculate reconstruction loss: How far is the reconstructed data from the real data
                    pre_loss = tf.norm(x_recon - batch, ord='euclidean')

                trainable_vars = self.trainable_variables
                gradients = tape.gradient(pre_loss, trainable_vars)

                # apply gardients
                self.optimizer.apply_gradients(
                    (grad, var)
                    for (grad, var) in zip(gradients, trainable_vars)
                    if grad is not None
                )
            if verbose:
                print('{0} - Epoch {1} - Reconstruction loss: {2}.'.format(dt.now().strftime('%Y-%m-%d %H:%m:%S'),
                                                                           _epoch,
                                                                           pre_loss
                                                                           ))
        pass

    def train_step(self, data):
        """
        This implements Algorithm 1 in the orig. paper for one mini-batch.
        """
        x, y_true = data

        # oberserve loss function wrt all trainable vars (which is the default in keras such that we do not need to tell it explicitly)
        with tf.GradientTape() as tape:

            # get "kmeans friendly" embedding
            x_latent = self.encoder(x)
            # y_pred = self.decoder(x_latent)  # this happens in _loss, actually

            if self.centers is None:
                # if this is the first calculation after pretraining, kmeans had not been initialized yet
                self.init_centers(x_latent)

            ### Loss calculation (equation (5) in orig paper)
            # get assignments of points in X (closest cluster centers) to calculate cluster loss
            nearest_center_label = self.get_assignment(x_latent)
            reconstruction_loss, clustering_loss = self._loss(x, y_true, nearest_center_label)
            # print(reconstruction_loss, clustering_loss)
            total_loss = reconstruction_loss + clustering_loss  # self._loss already uses lambda

        # UPDATE NETWORK PARAMS (gradient step in equation (6) in orig paper)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, trainable_vars)
            if grad is not None
        )  # if we wanted to ignore the recon_loss, then the decoder part has no gradient and we want to avoid the warn messages

        # update centers after updating assignments (kmeans gradient step update in equation (8) in orig paper)
        nearest_center_label = self.get_assignment(x_latent)
        for x, assigned_center in zip(x_latent, nearest_center_label):
            self.update_center(x, assigned_center)

        return {"loss": total_loss,
                "recon_loss": reconstruction_loss,
                "cluster_loss": clustering_loss}
