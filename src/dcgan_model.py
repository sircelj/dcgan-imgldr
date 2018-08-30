import tensorflow as tf
import math


def fix_size(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha=alpha)


def relu(x):
    return tf.nn.relu(x)


def conv(x, out_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='conv'):
    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', [k_h, k_w, x.get_shape()[-1], out_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=stddev))
        tf.summary.histogram('kernel', kernel)
        return tf.nn.conv2d(input=x, filter=kernel, strides=[1, d_h, d_w, 1], padding='SAME')


def deconv(x, out_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='deconv'):
    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', [k_h, k_w, out_shape[-1], x.get_shape()[-1]],
                                 initializer=tf.random_normal_initializer(stddev=stddev))

        tf.summary.histogram('kernel', kernel)
        return tf.nn.conv2d_transpose(x, kernel, out_shape, strides=[1, d_h, d_w, 1])


def linear(x, out_size, stddev=0.2, bias_start=0.0, name="linear"):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        matrix = tf.get_variable("matrix", [shape[1], out_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [out_size],
                               initializer=tf.constant_initializer(bias_start))

        return tf.matmul(x, matrix) + bias


def batchnorm(x, is_training, name='batch_norm'):
    with tf.variable_scope('batch_norm'):
        return tf.layers.batch_normalization(x, training=is_training, name=name)


def discriminator(X, is_training, disc_dim=64):
    """Discriminator network

    Parameters
    ----------
    X : tensor,
        Input image
    is_training : boolean or tensor,
        Parameter for batch_norm
    disc_dim : int,
        Number of kernels for the first layer

    Returns
    -------
    probability, logits: tuple (tensor, tensor)
        probability: Calculated probability that the image is real,
        logits: Logits for the probability
    """
    with tf.variable_scope('discriminator'):
        batch_size = X.get_shape().as_list()[0]
        with tf.variable_scope("conv_1"):
            h1 = lrelu(batchnorm(conv(X, disc_dim, name="conv_1"), is_training, name="batch_1"))
        with tf.variable_scope("conv_2"):
            h2 = lrelu(batchnorm(conv(h1, disc_dim*2, name="conv_2"), is_training, name="batch_2"))
        with tf.variable_scope("conv_3"):
            h3 = lrelu(batchnorm(conv(h2, disc_dim*4, name="conv_3"), is_training, name="batch_3"))
        with tf.variable_scope("conv_4"):
            h4 = lrelu(batchnorm(conv(h3, disc_dim*8, name="conv_4"), is_training, name="batch_4"))
        with tf.variable_scope("logits"):
            # logits = linear(tf.reshape(h4, [batch_size, -1]), 1, name="linear")
            logits = linear(tf.contrib.layers.flatten(h4), 1, name="linear")
        return tf.nn.sigmoid(logits), logits


def generator(z, start_height=4, start_width=4, out_channels=3, is_training=True):
    """Generator network

    Parameters
    ----------
    z : tensor, shape (batch_size, latent_size, )
        Input latent vector
    start_height : int
        Output height of image divided by 16
    start_width : int
        Output width of image divided by 16
    out_channels : int
        Number of desired output channels
    is_training : boolean or tensor
        Parameter for batch_norm

    Returns
    -------
    h4 : tensor, shape (batch_size, start_height * 16, start_width * 16, out_channels)
        Generated batch of images.
    """

    first_size = 512

    with tf.variable_scope('generator'):
        batch_size = z.get_shape().as_list()[0]
        with tf.variable_scope("reshape"):
            h0 = linear(z, first_size * start_height * start_width)
            h0 = tf.reshape(h0, [-1, start_height, start_width, first_size])
            h0 = relu(batchnorm(h0, is_training))
        with tf.variable_scope("deconv_1"):
            h1 = deconv(h0, [batch_size, start_height * 2, start_width * 2, first_size // 2])
            h1 = relu(batchnorm(h1, is_training))
        with tf.variable_scope("deconv_2"):
            h2 = deconv(h1, [batch_size, start_height * 4, start_width * 4, first_size // 4])
            h2 = relu(batchnorm(h2, is_training))
        with tf.variable_scope("deconv_3"):
            h3 = deconv(h2, [batch_size, start_height * 8, start_width * 8, first_size // 8])
            h3 = relu(batchnorm(h3, is_training))
        with tf.variable_scope("deconv_4"):
            h4 = deconv(h3, [batch_size, start_height * 16, start_width * 16, out_channels])
            h4 = tf.nn.tanh(h4)

        return h4
