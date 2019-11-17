import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class Model:
    def __init__(self, shape, dim_label, hidden_layer_list, dim_z):
        self.shape = shape
        self.dim_label = dim_label
        self.hidden_layer_list = hidden_layer_list  # shape of hidden layer for encoder and decoder
        self.dim_z = dim_z
        # height * weight * channel
        self.out_layer = self.shape[0] * self.shape[1] * self.shape[2]  # 64 * 64 * 1
        self.initializer = tf.compat.v1.variance_scaling_initializer()

    def relu(self, input):
        return tf.nn.relu(input)

    def dense(self, inputs, units, name):
        return tf.compat.v1.layers.dense(inputs=inputs,
                                         units=units,
                                         name=name,
                                         kernel_initializer=self.initializer)

    def encoder(self, X, C):
        with tf.compat.v1.variable_scope("encoder"):
            X_input = tf.concat((X, C), axis=1)
            net = self.relu(self.dense(X_input, self.hidden_layer_list[0], name="Dense_1"))
            net = self.relu(self.dense(net, self.hidden_layer_list[1], name="Dense_2"))
            net = self.dense(net, self.dim_z * 2, name="Dense_3")
            mean = net[:, :self.dim_z]
            std = tf.nn.softplus(net[:, self.dim_z:]) + 1e-6

        return mean, std

    def decoder(self, Z, C):
        with tf.compat.v1.variable_scope("decoder"):
            z_input = tf.concat((Z, C), axis=1)
            net = self.relu(self.dense(z_input, self.hidden_layer_list[2], name="Dense_1"))
            net = self.relu(self.dense(net, self.hidden_layer_list[3], name="Dense_2"))
            net = tf.nn.sigmoid(self.dense(net, self.out_layer, name="Dense_3"))

        return net

    def CVAE(self, X, C):
        # reshape input layer
        input_layer = tf.reshape(X, [-1, self.out_layer])

        # splitting encoder into mean and std
        mean, std = self.encoder(input_layer, C)
        # calculate latent variable z by sampling from a standard normal and multiply by std and plus mean
        # reparameterization trick - to allow optimization
        z = mean + std * tf.compat.v1.random_normal(tf.shape(mean, out_type=tf.int32), 0, 1, dtype=tf.float32)

        # feed latent variable z and condition to decoder
        output = self.decoder(z, C)
        clipped_output = tf.clip_by_value(output, 1e-7, 1 - 1e-7)
        tf.identity(clipped_output, name="output")

        # loss function
        likelihood = tf.reduce_mean(
            input_layer * tf.compat.v1.log(clipped_output) + (1 - input_layer) * tf.compat.v1.log(1 - clipped_output))
        KL_Div = tf.reduce_mean(0.5 * (1 - tf.compat.v1.log(tf.square(std) + 1e-8) + tf.square(mean) + tf.square(std)))
        ELBO = -1 * likelihood + KL_Div

        return z, output, ELBO, likelihood, KL_Div

    def optim_op(self, loss, learning_rate, global_step):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,
                                                                                           global_step=global_step)
        return optimizer
