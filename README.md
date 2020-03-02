# Conditional Variational Autoencoder implementation


## Overview

Conditional Variational Autoencoder implementation in tensorflow using [NIST dataset](https://catalog.data.gov/dataset/nist-handprinted-forms-and-characters-nist-special-database-19)
NIST dataset I am using is sepeated by class and contain 62 handwritten characters. structure of dataset and guide can be refered in [this pdf](https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf). All images are first resized to 64 * 64 compare to the original size 128 * 128.

The aim of is project is to create a conditional VAE model that are capable learning the distribution of NIST dataset and generate empirical distribution of handwritten charaters given ASCII character (0-9, a-z, A-Z).

![model structure](/images/CVAE_structure.PNG)

## Project setting

Default Hyper-parameter setting

| Hyper-parameter | Default Value |
| --- | --- |
| Dimenstion of latent variable | 20 |
| Epoch | 20 |
| Batch Size | 100 |
| Learning Rate | 0.0001 |
| Decay Rate | 0.98 |

## User Guide

To run this project, user have to download NIST data set from this [link](https://s3.amazonaws.com/nist-srd/SD19/by_class.zip). After the download is completed, user can *run data_processing.py* to preprocess the images data, including rescaling images, converting classes to labels, and saving labels and images into a numpy array. Afterward, proceed to run *main.py* to train the model. Result of each epoch will be saved into a generated directory *generated_images* in the same working directory. Saved model (checkpoint) will as well saved in the same directory for future use or further training.

## Performance

Current Model Performance with 174,415 images
Below are result generated from current model
<img src="/images/generated_output_0.png" alt="1st epoch" width="250" height="250">
<img src="/images/generated_output_1.png" alt="2st epoch" width="250" height="250">
<img src="/images/generated_output_2.png" alt="3st epoch" width="250" height="250">
<img src="/images/generated_output_3.png" alt="4st epoch" width="250" height="250">
<img src="/images/generated_output_4.png" alt="5st epoch" width="250" height="250">
<img src="/images/generated_output_19.png" alt="19st epoch" width="250" height="250">
