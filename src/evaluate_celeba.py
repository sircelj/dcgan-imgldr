import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from image_loader import plotimage_row
from dcgan_model import generator


def get_images(batch_size=64, image_dimensions=[64, 64, 3], z_size=100, vec=None,
               checkpoint_dir="celeba_output/2018-07-30_22h50m26s_DCGAN_S/generator"):
    print("Setting up the Graph")
    with tf.Graph().as_default():
        with tf.variable_scope("placeholder"):
            z = tf.placeholder(tf.float32, [batch_size, z_size], name="batch_noise")  # noise
            tf.summary.histogram('Noise', z)

        with tf.variable_scope("GAN"):
            G = generator(z, None, None, image_dimensions[2], is_training=False)

        gen_saver = tf.train.Saver()

        with tf.Session() as sess:
            latest_check = tf.train.latest_checkpoint(checkpoint_dir)
            print(latest_check)
            gen_saver.restore(sess, latest_check)

            if vec is None:
                vec = np.random.uniform(-1., 1., [64, z_size])
            return sess.run(G, feed_dict={z: vec})


def plot_interpolate(vec1, vec2, n_im=10, z_size=100):
    """
    Sample n_im vectors from interpolating two vectors and plot
    their generated images.

    Parameters
    ----------
    vec1 : ndarray
        First vector.
    vec2 : ndarray
        Second vector.
    n_im : int
        Number of interpolated vectors.
    z_size : int
        Size of the vectors.

    """
    # Define latent vectors
    z = np.random.uniform(-1, 1, [64, z_size])
    z[0] = vec1
    z[n_im + 1] = vec2
    for i in range(1, n_im + 1):
        t = i / (n_im + 1)
        z[i] = vec1 * (1 - t) + vec2 * t

    images = get_images(vec=z)[:n_im+1]
    plotimage_row(images)
    plt.show()
    # plt.savefig("interpolation/inteprolation_14.pdf", bbox_inches='tight')


if __name__ == "__main__":
    plot_interpolate(np.random.uniform(-1, 1, 100), np.random.uniform(-1, 1, 100))
