import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from model import *

from skimage import measure

from tensorflow.python.client import device_lib 


def train():
    tf.reset_default_graph()

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    gen_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32, name='generated_image')
    real_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32, name='groundtruth_image')
    Gz = generator(gen_in)
    Dx = discriminator(real_in)
    Dg = discriminator(Gz, reuse=True)
    lr_v = tf.Variable(lr_init)
    real_in_bgr = tf.map_fn(lambda img: RGB_TO_BGR(img), real_in)
    Gz_bgr = tf.map_fn(lambda img: RGB_TO_BGR(img), Gz)

    psnr=0
    ssim=0
    d_loss = -tf.reduce_mean(tf.log(1-Dx) + tf.log(Dg)) * ADVERSARIAL_LOSS_FACTOR
    g_loss = ADVERSARIAL_LOSS_FACTOR * -tf.reduce_mean(tf.log(1-Dg)) +  PIXEL_LOSS_FACTOR * (get_pixel_loss(real_in, Gz)) 
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    d_solver = tf.train.AdamOptimizer(4*lr_v).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_solver = tf.train.AdamOptimizer(lr_v).minimize(g_loss, var_list=g_vars)


    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        saver = initialize(sess)
        initial_step = global_step.eval()

        start_time = time.time()
        n_batches = 200
        total_iteration = n_batches * N_EPOCHS
        # print([n.attr['_output_shapes'] for n in tf.get_default_graph().as_graph_def(add_shapes=True).node])
        for index in range(initial_step, total_iteration):
            training_batch = load_next_training_batch()
            groundtruth_batch = load_next_groundtruth_batch()
            
            _, d_loss_cur = sess.run([d_solver, d_loss], feed_dict={gen_in: training_batch, real_in: groundtruth_batch})
            _, g_loss_cur = sess.run([g_solver, g_loss], feed_dict={gen_in: training_batch, real_in: groundtruth_batch})

            if(index + 1) % SKIP_STEP == 0:

                saver.save(sess, CKPT_DIR, index)
                image = sess.run(Gz, feed_dict={gen_in: training_batch})
                labels = sess.run(Dx, feed_dict={real_in : groundtruth_batch})
                labels_2 = sess.run(Dx, feed_dict={real_in : image})
                # new_lr_decay = lr_decay**(index // SKIP_STEP)
                # lr_v.assign(lr_init * new_lr_decay)
                img_save=(image[1]+1)/2
                cv2.imwrite(IMG_DIR+'val_%d.png' % (index+1), img_save*255)
                print(np.sum((labels>0.5)),np.sum((labels_2<0.5)))
                print(
                    "Step {}/{} Gen Loss: ".format(index + 1, total_iteration) + str(g_loss_cur)+ " Disc Loss: " + str(
                        d_loss_cur))



if __name__=='__main__':
    # print(device_lib.list_local_devices())
    training_dataset_init()
    groundtruth_dataset_init()
    train()
