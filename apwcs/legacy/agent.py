
import tensorflow as tf
import numpy as np

from apwcs.model import DQN

tf.app.flags.DEFINE_boolean("train", False, "Game mode")
FLAGS = tf.app.flags.FLAGS

MAX_EPISODE = 5000
TARGET_UPDATE_INTERVAL = 5
TRAIN_INTERVAL = 1
OBSERVE = 1

# offload: 1 / onload: 0
NUM_ACTION = 2

def train():
    sess = tf.Session()

