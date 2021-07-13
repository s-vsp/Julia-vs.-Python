import numpy as np
import pandas as pd
import tensorflow as tf
#import h5py
import sys
import io
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Flatten, Dense, Activation, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow.keras.optimizers import SGD, Adam

"""
Many-to-many  sort of sequence data proceeding.
Long Short Term Memory Recurrent Neural Network to create $uicideBoy$ songs' titles.

"""



def text_slicer(data,col):
    # Cutting off the pre written indices
    """
    Parameters:
    -----------
    data - dataframe
    col - names of columns from the dataframe
    """

    for i, song in data.iterrows():   
        if i <= 8:
            song[col] = song[col][3:]
        elif i > 8 and i < 99:
            song[col] = song[col][4:]
        elif i >= 99:
            song[col] = song[col][5:]
    return data


def text_samples_generator(text_file, temp_sequence):
    # Creating the labeled samples from a text file
    """
    Parameters:
    -----------
    text_file - text file to be processed
    temp_sequence - temporary integer to reduce/cut the chars into learning samples
    """
    
    X, y = [], []

    # Using a mapping function (dictionary) to make the samples as lists of integers on default
    chars = sorted(list(set(text_file)))
    char_map_to_int = {char: integer for integer, char in enumerate(chars)}

    for i in range(0,len(text_file)-temp_sequence,1):
        samples = text_file[i:i+temp_sequence]
        target = text_file[i+temp_sequence]
        X.append([char_map_to_int[char] for char in samples])
        y.append(char_map_to_int[target])
    
    return np.array(X), np.array(y)



def generator_text(text_file, batch_size, steps):
    # Creating the labeled samples from a text file
    """
    Parameters:
    -----------
    text_file - text file to be processed
    batch_size - 
    """

    chars = sorted(list(set(text_file)))
    char_map_to_int = {char: integer for integer, char in enumerate(chars)}

    text_as_ints = np.array([char_map_to_int[char] for char in text_file])

    total_batches_length = batch_size * steps
    number_of_batches = int(len(text_file) / total_batches_length)

    if (number_of_batches*total_batches_length + 1) > len(text_file):
        number_of_batches = number_of_batches - 1
    
    X = text_as_ints[0:number_of_batches*total_batches_length]
    y = text_as_ints[1:number_of_batches*total_batches_length+1]
    # X and y should have the same shape

    batched_X, batched_y = np.split(X, batch_size), np.split(y, batch_size)

    X = np.stack(batched_X)
    y = np.stack(batched_y)

    return X, y
    


def batch_split(X, y, steps):

    batch_size, batch_length = X.shape
    number_of_batches = int(batch_length/steps)

    for batch in range(number_of_batches):
        yield (X[:, batch*steps:(batch+1)*steps], y[:, batch*steps:(batch+1)*steps])



class LSTM_RNN(object):

    def __init__(self, num_classes, batch_size=64, steps=30, lstm_size=256, num_layers=1, learning_rate=0.0001, keep_prob=0.5, grad_clip=5, sampling_mode=False) -> None:
        self.num_classes = num_classes
        self.batch_size= batch_size
        self.steps = steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.grad_clip = grad_clip

        self.graph = tf.Graph()
        with self.graph.as_default():

            #sampling defining whether the graph is destinated to work through the process of learning or sampling

            self.build(sampling=sampling)
            self.saver = tf.train.Saver()
            self.init_op = tf.compat.v1.global_variables_initializer()


    def build(self, sampling_mode):

        if sampling_mode==True:
            batch_size = 1
            steps = 1
        elif sampling_mode==False:
            batch_size = self.batch_size
            steps = self.steps
        


    

def text_process(): 

    ### Text preprocessing ###

    songs_list = pd.read_csv("SB_songs_list.csv", header=None, names=["Title"])
    text_slicer(songs_list, "Title")

    titles_list = [] # Creating a list containing the titles

    for song in songs_list["Title"]:
        titles_list.append(song)
    
    # Creating a text file with the titles
    with open('SB_titles_text_file.txt', 'w', encoding="utf-8") as file:  
        for i in titles_list:
            file.write('%s\n' % i)
    
    titles = open("SB_titles_text_file.txt", "r", encoding="utf-8").read().lower()

    seq = 15
    X, y = text_samples_generator(titles, seq)
    
    # Reshaping samples (X) to be suitable for a LSTM RNN
    X = np.reshape(X, (len(X), seq, 1))

    # Categorical one-hot encoding
    y = tf.keras.utils.to_categorical(y)

    return X, y



def run_model_train(X, y):

    ### Creating the model ###


    model = keras.Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), activation="tanh", return_sequences=True)) 
    for i in range(1,4): # adding 3 hidden layers
        model.add(LSTM(256, activation="tanh", return_sequences=True))
        if i == 2:
            model.add(Dropout(0.2)) ### ADDING DROPOUT LAYER ###

    model.add(Flatten()) 
    model.add(Dense(y.shape[1])) 
    model.add(Activation("softmax"))
    model.add(Dropout(0.2))
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss="categorical_crossentropy")

    print(model.summary())

    # Callback function
    path = "LSTM-RNN-model-weights-improvement-{epoch:03d}-{loss:.5f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(path, monitor="loss", verbose=1, mode="min", save_best_only=True)
    callbacks = [checkpoint]
    
    ### Training ###

    model.fit(X, y, batch_size=32, epochs=5, verbose=1, callbacks=callbacks, validation_split=0.2, validation_data=None, shuffle=True, initial_epoch=0)

    return model


if __name__ == "__main__":

    X, y = text_process()
    #model = run_model_train(X, y)
    print(X.shape, y.shape)


    
    