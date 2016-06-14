from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam


def nn(train_values, train_classes_binary, test_values, test_classes_binary):

    model = Sequential()

    model.add(Dense(4096, input_dim=16063, init='lecun_uniform'), activation='relu')
    model.add(Dropout(0.5))

    model.add(Dense(1024, init='lecun_uniform'), activation='relu')
    model.add(Dropout(0.5))

    model.add(Dense(256, init='lecun_uniform'), activation='relu')
    model.add(Dropout(0.5))

    model.add(Dense(64, init='lecun_uniform'), activation='relu')
    model.add(Dropout(0.5))

    model.add(Dense(14, init='lecun_uniform'), activation='softmax')




    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    model.fit(train_values, train_classes_binary,
              nb_epoch=20,
              batch_size=16)
    score = model.evaluate(test_values, test_classes_binary, batch_size=16)
    return score
