import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Flatten, Dense, Activation


"""
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
    
    return X, y



def run():

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
    
    # Getting all the chars used in titles as a sorted list
    chars = sorted(list(set(titles)))
    
    # Generating the samples
    temp_seq = 20
    X, y = text_samples_generator(titles,temp_seq)

    # Reshaping samples (X) to be suitable for a LSTM RNN
    X = np.reshape(X, (len(X), temp_seq, 1))

    #Normalization:
    X = X / len(chars)

    # Categorical one-hot encoding class labels
    y = tf.keras.utils.to_categorical(y)


    ### Creating the model ###

    model = keras.Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), activation="tanh", return_sequences=True)) # Output shape = (None, 20, 256)
    for i in range(1,4):
        model.add(LSTM(256, activation="tanh", return_sequences=True))
    model.add(Flatten()) # Output shape = (None, 5120)
    model.add(Dense(y.shape[1])) # Output shape = (None, 46)
    model.add(Activation("softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    model.summary



if __name__ == "__main__":
    run()