import datetime
import math
import numpy as np
from model import *
import matplotlib.pyplot as plt
import time
from data_processing import create_label_dictionary, id_to_character


def plot_generated_image(image):
    size = 64
    canvas = np.reshape(image, (size, size))
    print("plot image")
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap="gray")


def next_batch(batch, images, labels, batch_size):
    batch_x = images[batch * batch_size: batch * batch_size + batch_size, :, :, :]
    batch_y = labels[batch * batch_size: batch * batch_size + batch_size, :]
    return batch_x, batch_y


def get_total_batch(images, batch_size):
    batch_size = batch_size
    return len(images) // batch_size


def shuffle_data(data, label):
    assert len(data) == len(label)
    p = np.random.permutation(len(data))
    return data[p], label[p]


if __name__ == "__main__":
    # initialize hyper-parameters
    dim_z = 9
    learning_rate = 0.01
    decay_rate = 0.99
    batch_size = 100
    epoch = 10

    # load image and label array
    print("Loading data")
    images = np.load("nist_images_test.npy")
    labels = np.load("nist_labels_test.npy")
    print("shuffling data")
    images, labels = shuffle_data(images, labels)
    print("data shuffled!")
    # cross validation
    split = math.ceil(len(images) * 0.7)
    # create training and testing data set
    train_x = images[:split]
    train_y = labels[:split]
    test_x = images[split:]
    test_y = labels[split:]

    id_dict = create_label_dictionary()
    _, height, width, channel = np.shape(train_x)
    dim_label = np.shape(train_y)[1]
    X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, height, width, channel], name="Input")
    # condition - pass in label to train, and act as an input during prediction
    C = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, dim_label], name="labels")
    model = Model([_, height, width, channel], dim_label, [1000, 500, 500, 1000], dim_z)
    z, output, loss, likelihood, kl_d = model.CVAE(X, C)

    # latent = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, dim_z], name="latent_variable")
    global_step = tf.compat.v1.Variable(0, trainable=False)
    # generated_image = model.decoder(latent, C)

    total_batch = get_total_batch(train_x, batch_size)
    learning_rate_decayed = learning_rate * decay_rate ** (global_step / total_batch)
    optim_op = model.optim_op(loss, learning_rate_decayed, global_step)

    sess = tf.compat.v1.Session()
    # sess.run(tf.compat.v1.initialize_all_variables())
    sess.run(tf.compat.v1.global_variables_initializer())
    start_time = time.time()
    writer = tf.compat.v1.summary.FileWriter("output", sess.graph)

    for i in range(epoch):
        loss_val = 0
        batch = 0
        for j in range(total_batch):
            batch_x, batch_y = next_batch(batch, train_x, train_y, batch_size)
            batch += 1
            feed_dict = {X: batch_x, C: batch_y}
            l, lr, op, g, gi, lh, kl = sess.run([loss, learning_rate_decayed, optim_op, global_step, output, likelihood, kl_d],
                                                feed_dict=feed_dict)
            loss_val += l / total_batch

            if i is (epoch - 1) and j is (total_batch - 1):
                for m in enumerate(gi):
                    id = np.argmax(train_y)
                    lb = id_to_character(id_dict, id)
                    print("plotting image for label '{}'".format(lb))
                    plot_generated_image(gi[m])

        sec = int(time.time() - start_time)
        print("Epoch: {} | loss: {} | lr: {} | Time: {} sec\n".format(i, loss_val, lr, sec))

    print("learning finished")
    writer.close()
    sess.close()
