# ConVAE

Conditional Variational Autoencoder

## Overview

Conditional Variational Autoencoder implementation in tensorflow using [NIST dataset](https://catalog.data.gov/dataset/nist-handprinted-forms-and-characters-nist-special-database-19)
NIST dataset I am using is sepeated by class and contain 62 handwritten characters. structure of dataset and guide can be refered in [this pdf](https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf). All images are first resized to 64 * 64 compare to the original size 128 * 128.

The aim of is project is to create a conditional VAE model that are capable learning the distribution of NIST dataset and generate empirical distribution of handwritten charaters given ASCII character (0-9, a-z, A-Z).
