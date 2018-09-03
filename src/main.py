import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from image_loader import ImageLoader, plotimage
from datetime import datetime
import os
import sys


# def train(batch_size=64, image_dimensions=[218, 178, 3], z_size=100):
def train(imageloader, batch_size=64, image_dimensions=[64, 64, 3], z_size=100, num_of_epochs=100,
          num_D_updates=1, num_G_updates=2, restart_from=None, logdir="DCGAN_12"):
    from dcgan_model import discriminator, generator

    print("Setting up the Graph")
    with tf.Graph().as_default():
        with tf.variable_scope("placeholder"):
            # Raw image
            X = tf.placeholder(tf.float32, [batch_size] + image_dimensions, name="batch_images")
            # X = tf.placeholder(tf.float32, [None] + image_dimensions)
            is_training = tf.placeholder(tf.bool, name="is_training")
            epoch_var = tf.Variable(0, trainable=False, name='epoch_var')
            step_var = tf.Variable(0, trainable=False, name='step_var')
            increment_step_var = tf.assign_add(step_var, 1)
            # Noise
            z = tf.placeholder(tf.float32, [batch_size, z_size], name="batch_noise")  # noise
            # z = tf.placeholder(tf.float32, [None, z_size])  # noise
            tf.summary.histogram('Noise', z)

        with tf.variable_scope("GAN"):
            G = generator(z, start_height=4, start_width=4,
                          out_channels=image_dimensions[2], is_training=is_training)
            D_real, D_real_logits = discriminator(X, is_training=is_training)
            tf.get_variable_scope().reuse_variables()
            D_fake, D_fake_logits = discriminator(G, is_training=is_training)

        with tf.variable_scope("prediction"):
            tf.summary.histogram('real', D_real)
            tf.summary.histogram('fake', D_fake)

            tf.summary.scalar('real', tf.reduce_mean(D_real))
            tf.summary.scalar('fake', tf.reduce_mean(D_fake))

        with tf.variable_scope("D_loss"):
            D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits,
                                                                                 labels=tf.ones_like(D_real_logits)))
            D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                                                 labels=tf.zeros_like(D_fake_logits)))
            D_loss = D_loss_real + D_loss_fake

            tf.summary.scalar('D_loss_real', tf.reduce_mean(D_loss_real))
            tf.summary.scalar('D_loss_fake', tf.reduce_mean(D_loss_fake))
            tf.summary.scalar('D_loss', tf.reduce_mean(D_loss))

        with tf.variable_scope("G_loss"):
            G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                                            labels=tf.ones_like(D_fake_logits)))

            tf.summary.scalar('G_loss', tf.reduce_mean(G_loss))

        with tf.variable_scope("train"):
            # Since batch_norm is used, the update_ops_D/G give the moving average variables
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Used for batch normalisation
            update_ops_D = [var for var in update_ops if 'discriminator' in var.name]
            update_ops_G = [var for var in update_ops if 'generator' in var.name]

            tvar = tf.trainable_variables()

            with tf.variable_scope("D"), tf.control_dependencies(update_ops_D):
                D_vars = [var for var in tvar if 'discriminator' in var.name]
                optimizer_D = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9)
                D_train = optimizer_D.minimize(D_loss, var_list=D_vars)

            with tf.variable_scope("G"), tf.control_dependencies(update_ops_G):
                G_vars = [var for var in tvar if 'generator' in var.name]
                optimizer_G = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9)
                G_train = optimizer_G.minimize(G_loss, var_list=G_vars)

        # Init the savers
        D_saver_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='GAN/discriminator') + \
                       tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train/D')
        G_saver_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='GAN/generator') +\
                       tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train/G')
        gen_saver = tf.train.Saver(var_list=G_saver_vars + [epoch_var, step_var],
                                   max_to_keep=num_of_epochs, filename='generator')
        dis_saver = tf.train.Saver(var_list=D_saver_vars + [epoch_var, step_var],
                                   max_to_keep=num_of_epochs, filename='discriminator')

        with tf.Session() as sess:
            # Make the output dir or continue from given checkpoint (restart_from)
            if restart_from:
                output_dir = restart_from + "/"
                latest_check_gen = tf.train.latest_checkpoint(output_dir + 'generator')
                latest_check_dis = tf.train.latest_checkpoint(output_dir + 'discriminator')
                print(latest_check_gen, latest_check_dis)
                gen_saver.restore(sess, latest_check_gen)
                dis_saver.restore(sess, latest_check_dis)
            else:
                sess.run(tf.global_variables_initializer())
                # Make the new directory name
                time_string = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
                output_dir = "celeba_output/" + time_string + "_" + logdir + "/"
            output_im_dir = output_dir + "images/"
            if not os.path.exists(output_dir):
                os.makedirs(output_im_dir)

            # Set ImageLoaders starting epoch
            curr_epoch = sess.run(epoch_var)
            imageloader.set_epoch(curr_epoch)

            print("Setting up the summary")
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(output_dir)
            writer.add_graph(sess.graph)

            print("Start learning")
            while curr_epoch < num_of_epochs:

                if (curr_epoch < celeba.epoch or sess.run(step_var) == 0) and celeba.epoch % 1 == 0:
                    print("")
                    curr_epoch = celeba.epoch
                    bleh = sess.run(tf.assign(epoch_var, curr_epoch))
                    bleh2 = sess.run(epoch_var)

                    # Write epoch summary
                    batch_X = celeba.get_new_batch()
                    batch_noise = np.random.uniform(-1, 1, [batch_size, z_size])

                    D_loss_print = sess.run(D_loss, feed_dict={X: batch_X, z: batch_noise, is_training: False})
                    G_loss_print = sess.run(G_loss, feed_dict={z: batch_noise, is_training: False})

                    s = sess.run(merged_summary, feed_dict={X: batch_X, z: batch_noise, is_training: False})
                    writer.add_summary(s, curr_epoch)
                    writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=sess.run(step_var))
                    print("epoch:%d   G_loss:%f   D_loss:%f\n" % (curr_epoch, G_loss_print, D_loss_print))

                    # Save checkpoint
                    if curr_epoch % 1 == 0:
                        samples = sess.run(G, feed_dict={z: np.random.uniform(-1., 1., [batch_size, z_size]),
                                                         is_training: False})

                        # Save images/audio... using the given imageloader
                        imageloader.epoch_save(samples, output_im_dir, curr_epoch)

                        # Make checkpoint
                        gout = gen_saver.save(sess, output_dir + "generator/ckpt", global_step=epoch_var)
                        dout = dis_saver.save(sess, output_dir + "discriminator/ckpt", global_step=epoch_var)
                        print("gout: " + gout)
                        print("dout: " + dout)
                        print("")

                # Discriminator update
                for _ in range(num_D_updates):
                    batch_X = celeba.get_new_batch()
                    batch_noise = np.random.uniform(-1, 1, [batch_size, z_size])
                    sess.run(D_train, feed_dict={X: batch_X, z: batch_noise, is_training: True})

                # Generator update
                for _ in range(num_G_updates):
                    batch_noise = np.random.uniform(-1, 1, [batch_size, z_size])
                    sess.run(G_train, feed_dict={z: batch_noise, is_training: True})

                # Loading bar
                if curr_epoch == celeba.epoch:
                    sys.stdout.write('\r')
                    sys.stdout.write("[%-60s] %d%%" % ('=' * int(60 * celeba.image_index / celeba.number_of_images),
                                                       int(100 * celeba.image_index / celeba.number_of_images)))
                    sys.stdout.flush()

                bleh = sess.run(increment_step_var)


if __name__ == '__main__':
    batch_size = 64

    print("Setting up ImageLoader")
    celeba = ImageLoader('../img_align_celeba/', batch_size=batch_size)
    # celeba = ImageLoader('../img_small/', batch_size=batch_size)

    train(imageloader=celeba, batch_size=batch_size)
    # train(restart_from='celeba_output/2018-07-30_22h50m26s_DCGAN_S')
    # train(restart_from='celeba_output/2018-07-13_23h03m37s_DCGAN_S', num_of_epochs=30)
