import os
import re
import sys
import glob
import scipy.misc
from itertools import cycle
import cv2
import numpy as np
import tensorflow as tf
lr_init = 0.00001
lr_decay=0.1
BATCH_SIZE = 8
BATCH_SHAPE = [BATCH_SIZE, 256, 256, 3]
SKIP_STEP = 200
N_EPOCHS = 1000
CKPT_DIR = './Checkpoints/'
IMG_DIR = './Images/'
GRAPH_DIR = './Graphs/'
TRAINING_SET_DIR= '../rephrase_data/degrade/'
GROUNDTRUTH_SET_DIR='../rephrase_data/train/'
TESTING_SET_DIR= '../rephrase_data/degrade_test/'
GROUNDTRUTH_TEST_SET_DIR='../rephrase_data/test/'
OUT_DIR='../rephrase_data/output_small/'
ADVERSARIAL_LOSS_FACTOR = 64
PIXEL_LOSS_FACTOR = 0.1

def initialize(sess):
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(GRAPH_DIR, sess.graph)

    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_DIR))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    return saver

def load_next_training_batch():
    batch = next(pool)
    return batch

def training_dataset_init():
    filelist = glob.glob(TRAINING_SET_DIR + '/*/*.jpg')
    batch = np.array([cv2.resize(cv2.imread(fname),(256,256)) for fname in filelist],dtype=float)
    print(np.mean(batch,axis=(1,2,3)).shape)
    batch-=np.mean(batch,axis=(1,2,3),keepdims=True)
    batch/=batch.std(axis=(1,2,3),keepdims=True)
    batch = np.array(split(batch, BATCH_SIZE))
    global pool
    pool = cycle(batch)


def groundtruth_dataset_init():
    filelist = glob.glob(GROUNDTRUTH_SET_DIR + '/*/*.jpg')
    ground_batch = np.array([cv2.resize(cv2.imread(fname),(256,256)) for fname in filelist],dtype=float)
    ground_batch-=np.mean(ground_batch,axis=(1,2,3),keepdims=True)
    ground_batch/=ground_batch.std(axis=(1,2,3),keepdims=True)
    ground_batch = np.array(split(ground_batch, BATCH_SIZE))
    print(ground_batch.shape)
    global ground_pool
    ground_pool = cycle(ground_batch)


def load_next_groundtruth_batch():
    batch = next(ground_pool)
    return batch

def testing_dataset_init():
    filelist = glob.glob(TESTING_SET_DIR + '/*/*.jpg')
    print(len(filelist))
    ground_batch = np.array([cv2.resize(cv2.imread(fname),(256,256)) for fname in filelist],dtype=float)
    global mean, std
    mean=np.mean(ground_batch,axis=(0,1,2,3),keepdims=True)
    std=ground_batch.std(axis=(0,1,2,3),keepdims=True)
    ground_batch-=mean
    ground_batch/=std
    print(mean,std)
    ground_batch = np.array(split(ground_batch, BATCH_SIZE))
    print(ground_batch.shape)
    global test_pool
    test_pool = cycle(ground_batch)

def testing_truth_dataset_init():
    filelist = glob.glob(GROUNDTRUTH_TEST_SET_DIR + '/*/*.jpg')
    ground_batch = np.array([cv2.resize(cv2.imread(fname),(256,256)) for fname in filelist],dtype=float)
    ground_batch = np.array(split(ground_batch, BATCH_SIZE))
    print(ground_batch.shape)
    global ground_test_pool
    ground_test_pool = cycle(ground_batch)

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return 0.6 * x + 0.4 * abs(x)

def get_pixel_loss(target,prediction):
    pixel_difference = target - prediction
    pixel_loss = tf.nn.l2_loss(pixel_difference)
    return pixel_loss


