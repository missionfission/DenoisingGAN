import time
import os
import re
import sys
import glob
import scipy.misc
from itertools import cycle
import cv2

import tensorflow as tf
import numpy as np

from utils import *
from model import *
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,250)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


from skimage import measure
TESTING_SET_DIR= '../rephrase_data/degrade_test/'
GROUNDTRUTH_TEST_SET_DIR='../rephrase_data/test/'
OUT_DIR='../rephrase_data/output_small/'
BATCH_SIZE = 20
def test():
    tf.reset_default_graph()

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    gen_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32, name='generated_image')
    real_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32, name='groundtruth_image')
    Gz = generator(gen_in)  
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = initialize(sess)
        initial_step = global_step.eval()
        start_time = time.time()
        initial_step = 0
        total_iteration = 2  
        for index in range(initial_step, total_iteration):
            test_batch=next(test_pool)
            test_image = (test_batch)*std+mean
            ground_test_batch=next(ground_test_pool) 
            out_image = sess.run(Gz, feed_dict={gen_in: test_batch})
            out_image = (out_image)*std+mean
            for j in range(20):
                img_save=np.hstack((test_image[j],out_image[j],ground_test_batch[j]))
                cv2.putText(img_save,'Distorted         Generated       GroundTruth', bottomLeftCornerOfText,  font,   fontScale, fontColor, lineType)
                cv2.imwrite(OUT_DIR+'out_%d.png' % (20*index+j), img_save)

testing_dataset_init()
testing_truth_dataset_init()
test()
