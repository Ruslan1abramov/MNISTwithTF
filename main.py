import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import sklearn.metrics as skm
import math

from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

###
file_name = "output.txt"
test1 = "Liner Regression"
test2 = "Convolution layer with FC RelU layer"
# hyper parameters
learning_rate = 0.0001
nb_batches = 13000
mini_batch = 50

########
# Convolution Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 32         # There are 32 of these filters.

# Convolution Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 64         # There are 64 of these filters.

# Fully-connected layer.
fc_size = 1024             # Number of neurons in fully-connected layer.

############
data = input_data.read_data_sets('MNIST_data', one_hot=True)
data.test.cls = np.array([label.argmax() for label in data.test.labels])
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1
# Number of classes, one class for each of 10 digits.
num_classes = 10


def accuracy_values(y_true_vec, y_pred):
    acc = skm.accuracy_score(y_true_vec, y_pred)
    precision = skm.precision_score(y_true_vec, y_pred, average='macro')
    recall = skm.recall_score(y_true_vec, y_pred, average='macro')
    f1_score = skm.f1_score(y_true_vec, y_pred, average='macro')
    return acc, precision, recall, f1_score


def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def optimize(num_iterations, mini_batch_size, x_vec, y_vec, session, optimizer, accuracy):
    start_time = time.time()
    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(mini_batch_size)

        feed_dict_train = {x_vec: x_batch,
                           y_vec: y_true_batch}

        if i % 100 == 0:
            print_accuracy(session, accuracy, feed_dict_test)
        session.run(optimizer, feed_dict=feed_dict_train)

    # the time-usage.
    return round(time.time() - start_time)


def print_accuracy(session_sent, accuracy, feed_dict_acc):
    # Use TensorFlow to compute the accuracy.
    acc = session_sent.run(accuracy, feed_dict=feed_dict_acc)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))
    return acc


def print_confusion_matrix(data_sent, session_sent, y_pred_cls, feed_dict_confusion):
    # Get the true classifications for the test-set.
    cls_true = data_sent.test.cls

    # Get the predicted classifications for the test-set.
    cls_pred = session_sent.run(y_pred_cls, feed_dict=feed_dict_confusion)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

    return accuracy_values(cls_true, cls_pred)


def plot_example_errors(session_sent, correct_prediction, y_pred_cls, feed_dict_plot):
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image,
    correct, cls_pred = session_sent.run([correct_prediction, y_pred_cls],
                                         feed_dict=feed_dict_plot)

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_weights(session_sent, weights):
    # Get the values for the weights from the TensorFlow variable.
    w = session_sent.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i < 10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# init tf
x = tf.placeholder(tf.float32, [None, img_size_flat], name="x")
y_true = tf.placeholder(tf.float32, [None, num_classes], name="y_true")
y_true_cls = tf.placeholder(tf.int64, [None], name="y_true__cls")
feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}
session = tf.Session()


def liner_regression_no_hidden_layer(mini_batch_size=mini_batch):
    weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
    biases = tf.Variable(tf.zeros([num_classes]))

    total_weights = img_size_flat * num_classes + num_classes

    logits = tf.matmul(x, weights) + biases
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, axis=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session.run(tf.global_variables_initializer())

    print_accuracy(session, accuracy, feed_dict_test)
    plot_example_errors(session, correct_prediction, y_pred_cls, feed_dict_test)

    computing_time = optimize(nb_batches, mini_batch_size, x, y_true, session, optimizer, accuracy)
    print_accuracy(session, accuracy, feed_dict_test)
    plot_example_errors(session, correct_prediction, y_pred_cls, feed_dict_test)
    plot_weights(session, weights)
    acc_vals = print_confusion_matrix(data, session, y_pred_cls, feed_dict_test)
    # file_name, test_name, number_of_layers, number_of_weights, batch_size, cpu_time, acc_vals)
    write_results_to_file(file_name=file_name, test_name=test1, learning=learning_rate, batch_size=mini_batch_size,
                          number_of_layers=0, number_of_weights=total_weights,
                          acc_vals=acc_vals, cpu_time=computing_time, set_size=nb_batches)


def liner_regression_2_hidden_layers(layer_size, mini_batch_size=mini_batch):

    w1 = tf.Variable(tf.random_uniform([img_size_flat, layer_size], -1, 1, seed=0))
    w2 = tf.Variable(tf.random_uniform([layer_size, layer_size], -1, 1, seed=0))
    w3 = tf.Variable(tf.random_uniform([layer_size, num_classes], -1, 1))
    b1 = tf.Variable(tf.zeros([layer_size]), name="Biases1")
    b2 = tf.Variable(tf.zeros([layer_size]), name="Biases2")
    b3 = tf.Variable(tf.zeros([num_classes]), name="Biases3")

    z1 = tf.matmul(x, w1) + b1
    z2 = tf.matmul(z1, w2) + b2
    z3 = tf.matmul(z2, w3) + b3

    total_weights = img_size_flat * layer_size + layer_size + layer_size*layer_size + layer_size \
                    + layer_size * num_classes + num_classes

    y_pred = tf.nn.softmax(z3)
    y_pred_cls = tf.argmax(y_pred, axis=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=z3, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session.run(tf.global_variables_initializer())

    print_accuracy(session, accuracy, feed_dict_test)
    plot_example_errors(session, correct_prediction, y_pred_cls, feed_dict_test)

    computing_time = optimize(nb_batches, mini_batch_size, x, y_true, session, optimizer, accuracy)
    print_accuracy(session, accuracy, feed_dict_test)
    plot_example_errors(session, correct_prediction, y_pred_cls, feed_dict_test)
    plot_weights(session, w1)
    acc_vals = print_confusion_matrix(data, session, y_pred_cls, feed_dict_test)
    write_results_to_file(file_name=file_name, test_name=test1, learning=learning_rate, batch_size=mini_batch_size,
                          number_of_layers=2, number_of_weights=total_weights,
                          acc_vals=acc_vals, cpu_time=computing_time, set_size=nb_batches)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x_vector, weights):
    return tf.nn.conv2d(x_vector, weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x_vector):
    return tf.nn.max_pool(x_vector, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def plot_example_errors_conv(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix_conv(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def print_test_accuracy_conv(x_vec, y_true_vec, y_pred_cls, keep_prob,
                             show_example_errors=False, show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + mini_batch, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x_vec: images,
                     y_true_vec: labels,
                     keep_prob: 1}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors_conv(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix_conv(cls_pred=cls_pred)

    return accuracy_values(cls_true, cls_pred)


def layer_convol(layers, mini_batch_size = mini_batch):

    #  first layer
    W_conv1 = weight_variable([filter_size1, filter_size1, 1, num_filters1])
    b_conv1 = bias_variable([num_filters1])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    layer1_before_relu = conv2d(x_image, W_conv1) + b_conv1
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    if layers == 1:     # we need to do davide the rows by 16 if we have only onw layer
        h_pool1 = max_pool_2x2(max_pool_2x2(h_conv1))

        #  1024 fc layer
        W_fc1 = weight_variable([7 * 7 * num_filters1, fc_size])
        b_fc1 = bias_variable([fc_size])
        h_pool1_flat = tf.reshape(h_pool1, [-1, 7 * 7 * num_filters1])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

        #   applying dropout
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #   out layer
        W_fc2 = weight_variable([fc_size, num_classes])
        b_fc2 = bias_variable([num_classes])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        total_weights = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    else:
        h_pool1 = max_pool_2x2(h_conv1)
        layer1_with_relu = h_pool1

        #  second layer
        W_conv2 = weight_variable([filter_size2, filter_size2, num_filters1, num_filters2])
        b_conv2 = bias_variable([num_filters2])
        layer2_before_relu = conv2d(h_pool1, W_conv2) + b_conv2
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        layer2_with_relu = h_pool2

        #  1024 fc layer
        W_fc1 = weight_variable([7 * 7 * num_filters2, fc_size])
        b_fc1 = bias_variable([fc_size])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * num_filters2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #   applying dropout
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #   out layer
        W_fc2 = weight_variable([fc_size, num_classes])
        b_fc2 = bias_variable([num_classes])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        total_weights = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    y_pred = tf.nn.softmax(y_conv)
    y_pred_cls = tf.argmax(y_pred, axis=1)

    # training the model
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_conv))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session.run(tf.global_variables_initializer())

    print_test_accuracy_conv(x, y_true, y_pred_cls, keep_prob)
    start_time = time.time()
    for i in range(nb_batches):
        batch = data.train.next_batch(mini_batch_size)
        if i % 100 == 0:
            train_accuracy = session.run(accuracy, feed_dict={x: batch[0], y_true: batch[1], keep_prob: 1})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        session.run(optimizer, feed_dict={x: batch[0], y_true: batch[1], keep_prob: 1})

    session.run(accuracy, feed_dict={x: batch[0], y_true: batch[1], keep_prob: 1})
    # Print the time-usage.
    computing_time = round(time.time() - start_time)
    acc_vals = print_test_accuracy_conv(x, y_true, y_pred_cls, keep_prob,
                                        show_example_errors=True, show_confusion_matrix=True)
    write_results_to_file(file_name=file_name, test_name=test2, learning=learning_rate, batch_size=mini_batch_size,
                          number_of_layers=1+layers, number_of_weights=total_weights,
                          acc_vals=acc_vals, cpu_time=computing_time, set_size=nb_batches)

    if layers == 2:
        return layer1_before_relu, layer1_with_relu, layer2_before_relu, layer2_with_relu


def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def write_results_to_file(file_name, test_name, learning, set_size, number_of_layers, number_of_weights, batch_size, cpu_time, acc_vals):
    with open(file_name, 'a') as results_file:
        results_file.write(
            "\n{0} [learning rate = {1},training set size = {2},"
            " batch size = {3}, network depth = {4}, number of weights = {5}]"
            "\nCPU time  :  {6}sec\n"
            "Accuracy : {7} , Precision : {8} , Recall {9} , F-Score : {10}\n\n".format(
                test_name, learning, set_size, batch_size, number_of_layers, number_of_weights,  cpu_time, acc_vals[0], acc_vals[1],
                acc_vals[2], acc_vals[3]))


def run_test(net_type, number_of_hidden_in_liner=200, cnn_layers=1, batch_size=mini_batch):
    if net_type == 'liner 2 hidden':
        liner_regression_2_hidden_layers(number_of_hidden_in_liner)
    elif net_type == 'liner no hidden':
        liner_regression_no_hidden_layer()
    elif net_type == 'cnn':
        #  plotting the digit
        if cnn_layers == 2:
            #  control digit
            plot_image(data.test.images[18])
            layers = layer_convol(cnn_layers, batch_size)
            plot_conv_layer(layers[0], data.test.images[18])
            plot_conv_layer(layers[1], data.test.images[18])
            plot_conv_layer(layers[2], data.test.images[18])
            plot_conv_layer(layers[3], data.test.images[18])
        else:
            layer_convol(cnn_layers, batch_size)


cnn = 'cnn'
#  layers_of_convol=1 //  layers_of_convol=2
layers_of_convol = 1
liner_2_hidden = 'liner 2 hidden'
number_of_hidden = 200
liner_no_hidden = 'liner no hidden'


#   running the code without wrapper function
# liner_regression_no_hidden_layer()
# liner_regression_2_hidden_layers(200)
# layer_convol(1, 100)
# layer_convol(1, 50)
# layer_convol(2, 100)
# layer_convol(2, 50)

#   running the code with wrapper function
#   net_type = cnn for running cnn
#   layers_of_convol = 1 for cnn with one convolution
#   layers_of_convol = 2 for cnn with 2 convolutions
#   net_type = liner_no_hidden for running liner regression with no layers
#   net_type = liner_2_hidden for running liner regression with 2 layers
#   number_of_hidden = 200 the amount of neurons in each layer
#
run_test(net_type=cnn, cnn_layers=2, batch_size=100)










