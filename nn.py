from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, Adadelta, SGD


def nn(train_values, train_classes_binary, test_values, test_classes_binary):

    model = Sequential()

    model.add(Dense(4096, input_dim=16063, init='uniform', activation='tanh'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    # model.add(Dense(1024, init='uniform', activation='tanh'))
    # model.add(Dropout(0.5))

    # model.add(Dense(256, init='uniform', activation='tanh'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    # model.add(Dense(64, init='uniform', activation='tanh'))
    # model.add(Dropout(0.5))

    model.add(Dense(14, init='uniform', activation='softmax'))

    adam = Adam()
    rmsprop = RMSprop()

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    history = model.fit(train_values, train_classes_binary,
                        batch_size=16,
                        nb_epoch=100,
                        verbose=2,
                        # validation_split=0.1,
                        validation_data=(test_values, test_classes_binary),
                        shuffle=True)

    print('training finished')

    score = model.evaluate(test_values, test_classes_binary, batch_size=16, verbose=1)
    return score, history
