import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import sys
import io
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Flatten, Dense, Activation, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow.keras.optimizers import SGD, Adam