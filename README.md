# dcn-deep-clustering-networks
This repository contains an implementation of the clustering approach introduced in "Towards K-means-friendly Spaces: Simultaneous Deep Learning and Clustering" (http://arxiv.org/abs/1610.04794).

The algorithm uses self-supervised learning with a custom loss function to learn a mapping of the input data (the observed space) to a k-means friendly latent space.

Through this approach, the k-means algorithm can identify meaningful clusters in data that is hard to cluster in the observed space.

