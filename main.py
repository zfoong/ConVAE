"""
Author : Tham Yik Foong
Student ID : 20200786
Project title : Conditional Variational Autoencoder implementation with Pythonn

Academic integrity statement :
I, Tham Yik Foong, have read and understood the School's Academic Integrity Policy, as well as guidance relating to
this module, and confirm that this submission complies with the policy. The content of this file is my own original
work, with any significant material copied or adapted from other sources clearly indicated and attributed.
"""

import math
import numpy as np
from model import *
import matplotlib.pyplot as plt
import time
from data_processing import create_label_dictionary, id_to_character
from scipy.stats import norm
import tensorflow.compat.v1 as tf


def plot_generated_image(image, label, i, size=64):
    """
    Plot or save generated image into 'generated_images' directory
    """
    canvas = np.reshape(image, (size, size))
    print("plotting image for label '{}'".format(label))
    plt.figure(figsize=(8, 8))
    plt.suptitle("image for label '{}' in epoch {}".format(label, i))
    plt.imshow(canvas, cmap="gray")
    plt.savefig("generated_images/generated_output_{}.png".format(i))
    plt.close()


def next_batch(batch, images, labels, batch_size):
    batch_x = images[batch * batch_size: batch * batch_size + batch_size, :, :]
    batch_y = labels[batch * batch_size: batch * batch_size + batch_size, :]
    return batch_x, batch_y


def get_total_batch(images, batch_size):
    return len(images) // batch_size


def shuffle_data(data, label):
    """
    Code originated from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    answered by user 'mtrw'
    """
    assert len(data) == len(label)
    p = np.random.permutation(len(data))
    return data[p], label[p]


def main():
    # initialize hyper-parameters
    dim_z = 20
    learning_rate = 0.0001
    decay_rate = 0.98
    batch_size = 128
    epoch = 20
    img_size = 64

    # load image and label array
    print("Loading data")
    images = np.load("nist_images.npy")
    labels = np.load("nist_labels.npy")
    print("shuffling data")
    images, labels = shuffle_data(images, labels)
    print("data shuffled!")

    # cross validation
    split = math.ceil(len(images) * 0.7)
    train_x = images[:split]
    train_y = labels[:split]
    test_x = images[split:]
    test_y = labels[split:]

    id_dict = create_label_dictionary()
    _, height, width = np.shape(train_x)
    dim_label = np.shape(train_y)[1]

    X = tf.placeholder(dtype=tf.float32, shape=[None, height, width], name="Input")
    # condition - pass in label to train, and act as an input during prediction
    C = tf.placeholder(dtype=tf.float32, shape=[None, dim_label], name="labels")
    model = Model([height, width], dim_label, [1000, 500, 500, 1000], dim_z)
    z, output, loss, likelihood, kl_d, mean, std, il = model.CVAE(X, C)
    global_step = tf.Variable(0, trainable=False)

    total_batch = get_total_batch(train_x, batch_size)
    learning_rate_decayed = learning_rate * decay_rate ** (global_step / total_batch)
    optim_op = model.optim_op(loss, learning_rate_decayed, global_step)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    start_time = time.time()
    writer = tf.summary.FileWriter("output", sess.graph)  # utilizing tensorboard for graph inspection

    for i in range(epoch):
        loss_val = 0
        batch = 0
        for j in range(total_batch):
            batch_x, batch_y = next_batch(batch, train_x, train_y, batch_size)
            batch += 1
            feed_dict = {X: batch_x, C: batch_y}
            l, lr, op, g, gi, lh, kl, me, st, il_ = sess.run(
                [loss, learning_rate_decayed, optim_op, global_step, output, likelihood, kl_d, mean, std, il],
                feed_dict=feed_dict)
            loss_val += l / total_batch

            if j is 1:
                id = np.argmax(batch_y[0])
                lb = id_to_character(id_dict, id)
                plot_generated_image(gi[0], lb, i, img_size)
                if dim_z is 20:
                    # plotting latent variable's distribution
                    fig, axes = plt.subplots(5, 4)
                    fig.suptitle("latent variable's distribution")
                    for o in range(0, 20):
                        x_axis = np.arange(-2, 2, 0.001)
                        _m = me[0][o]
                        _std = st[0][o]
                        axes[o % 5][o % 4].plot(x_axis, norm.pdf(x_axis, _m, _std))
                    plt.savefig('latent_variable_distribution.png')
                    plt.close(fig)

        sec = int(time.time() - start_time)
        print(
            "Epoch: {} | loss: {} | likelihood : {} | KL: {} | lr: {} | Time: {} sec\n".format(i, loss_val, lh, kl,
                                                                                                   lr, sec))

    print("learning finished")
    writer.close()
    saved_model_path = saver.save(sess, "saved_model.ckpt")  # saving a checkpoint for further training or prediction
    print("model saved in {}".format(saved_model_path))
    sess.close()


if __name__ == "__main__":
    main()
