#def run():

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

    int_map_to_char = {integer: char for integer, char in enumerate(chars)}
    
    # Generating the samples
    temp_seq = 50
    X, y = text_samples_generator(titles,temp_seq)

    # Reshaping samples (X) to be suitable for a LSTM RNN
    X = np.reshape(X, (len(X), temp_seq, 1))

    # Categorical one-hot encoding class labels
    y = tf.keras.utils.to_categorical(y)


    ### Creating the model ###

    model = keras.Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), activation="tanh", return_sequences=True)) # Output shape = (None, 20, 256)
    for i in range(1,5):
        model.add(LSTM(256, activation="tanh", return_sequences=True))
    model.add(Flatten()) # Output shape = (None, 5120)
    model.add(Dense(y.shape[1])) # Output shape = (None, 46)
    model.add(Activation("softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    print(model.summary())

    # Callback function
    path = "LSTM-RNN-model-weights-improvement-{epoch:03d}-{loss:.5f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(path, monitor="loss", verbose=1, mode="min", save_best_only=True)
    callbacks = [checkpoint]
    


    ### Training ###

    model.fit(X, y, batch_size=32, epochs=30, verbose=1, callbacks=callbacks, validation_split=0.2, validation_data=None, shuffle=True, initial_epoch=0)
    
    weights = "LSTM-RNN-model-weights-improvement-20-2.88230-bigger.hdf5"
    model.load_weights(weights)
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    t = np.random.randint(0, len(X)-1)
    s = X[t]
    s = np.reshape(s, (temp_seq,))

    #print(''.join([int_map_to_char[value] for value in s]))

    s = list(s)

    print("------------------------------------------")
    # Generate Charachters :
    for i in range(10):
        x = np.reshape(s, ( 1, len(s), 1))
        prediction = model.predict(x, verbose = 0)
        index = np.argmax(prediction)
        result = int_map_to_char[index]
        #seq_in = [int_chars[value] for value in pattern]
        sys.stdout.write(result)
        s.append(index)
        s = s[1:len(s)]
    print("\n------------------------------------------")

    print(len(s))








    weights = "LSTM-RNN-model-weights-improvement-004-9.12776-bigger.hdf5"
    model.load_weights(weights)
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    t = np.random.randint(0, len(X)-1)
    s = X[t]
    s = np.reshape(s, (seq,))

    chars = sorted(list(set(titles)))
    int_map_to_char = {integer: char for integer, char in enumerate(chars)}

    print(''.join([int_map_to_char[value] for value in s]))

    s = list(s)

    print("------------------------------------------")
    # Generate Charachters :
    for i in range(15):
        x = np.reshape(s, ( 1, len(s), 1))
        prediction = model.predict(x, verbose = 0)
        index = np.argmax(prediction)
        result = int_map_to_char[index]
        #seq_in = [int_chars[value] for value in pattern]
        sys.stdout.write(result)
        s.append(index)
        s = s[1:len(s)]
    print("\n------------------------------------------")

    #print(len(s))