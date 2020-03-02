"""
Author : Tham Yik Foong
Student ID : 20200786
Project title : Conditional Variational Autoencoder implementation with Pythonn

Academic integrity statement :
I, Tham Yik Foong, have read and understood the School's Academic Integrity Policy, as well as guidance relating to
this module, and confirm that this submission complies with the policy. The content of this file is my own original
work, with any significant material copied or adapted from other sources clearly indicated and attributed.
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


class Model:
    """
    Conditional Variational Autoencoder model implemented with Tensorflow
    This module provide a plug and play interface of CVAE for other project powered by with Tensorflow
    """
    def __init__(self, shape, dim_label, hidden_layer_list, dim_z):
        """
        :param shape: Dimension of input data. e.g. [28,28]
        :param dim_label: Shape of label vector
        :param hidden_layer_list: Number of layers and dimension of each layer, usually symmetric. e.g. [1024, 1024]
        :param dim_z: Dimension of latent variable, compressibility of model
        """
        self.shape = shape
        self.dim_label = dim_label
        self.hidden_layer_list = hidden_layer_list  # shape of hidden layer for encoder and decoder
        self.dim_z = dim_z
        self.out_layer = self.shape[0] * self.shape[1]  # height * weight, channel are removed.
        self.initializer = tf.variance_scaling_initializer()

    def relu(self, input):
        return tf.nn.relu(input)

    def dense(self, inputs, units, name):
        return tf.layers.dense(inputs=inputs,
                               units=units,
                               name=name,
                               kernel_initializer=self.initializer)

    def encoder(self, X, C):
        """
        Encoder Network compressing input into a set of mean and standard deviation
        :param X: input data
        :param C: condition
        :return: vector of mean and standard deviation
        """
        with tf.variable_scope("encoder"):
            X_input = tf.concat((X, C), axis=1)
            # X_input = X
            net = self.relu(self.dense(X_input, self.hidden_layer_list[0], name="Dense_1"))
            net = self.relu(self.dense(net, self.hidden_layer_list[1], name="Dense_2"))
            net = self.dense(net, self.dim_z * 2, name="Dense_3")
            mean = net[:, :self.dim_z]
            std = tf.nn.softplus(net[:, self.dim_z:]) + 1e-6

        return mean, std

    def decoder(self, Z, C):
        """
        Decoder Network compressing latent variable and a given condition into an output
        :param Z: Latent distribution
        :param C: Condition
        :return: Network output
        """
        with tf.variable_scope("decoder"):
            z_input = tf.concat((Z, C), axis=1)
            net = self.relu(self.dense(z_input, self.hidden_layer_list[2], name="Dense_1"))
            net = self.relu(self.dense(net, self.hidden_layer_list[3], name="Dense_2"))
            net = tf.nn.sigmoid(self.dense(net, self.out_layer, name="Dense_3"))

        return net

    def CVAE(self, X, C):
        """
        CVAE model compost of input, output layer, encoder and decoder network and calculate loss function
        :param X: input data
        :param C: condition
        :return:
        """
        # reshape input layer
        input_layer = tf.reshape(X, [-1, self.out_layer])

        # splitting encoder into mean and std
        mean, std = self.encoder(input_layer, C)
        # calculate latent variable z by sampling from a standard normal and multiply by std and plus mean
        # reparameterization trick - to allow optimization
        z = mean + std * tf.random_normal(tf.shape(mean, out_type=tf.int32), 0, 1, dtype=tf.float32)

        # feed latent variable z and condition to decoder
        output = self.decoder(z, C)
        output = tf.clip_by_value(output, 1e-7, 1 - 1e-7)

        # loss function
        likelihood = tf.reduce_mean(input_layer * tf.log(output) + (1 - input_layer) * tf.log(1 - output))
        KL_Div = tf.reduce_mean(0.5 * (1 - tf.log(tf.square(std) + 1e-7) + tf.square(mean) + tf.square(std)))
        ELBO = -1 * likelihood + KL_Div

        return z, output, ELBO, likelihood, KL_Div, mean, std, input_layer

    def optim_op(self, loss, learning_rate, global_step):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,
                                                                                 global_step=global_step)
        return optimizer
