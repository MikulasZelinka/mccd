from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


def nn(train_values, train_classes_binary, test_values, test_classes_binary):

    model = Sequential()

    model.add(Dense(4096, input_dim=16063, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))

    # model.add(Dense(1024, init='uniform', activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(256, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))

    # model.add(Dense(64, init='uniform', activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(14, init='uniform', activation='softmax'))




    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    model.fit(train_values, train_classes_binary,
              nb_epoch=100,
              batch_size=16,
              validation_data=(test_values, test_classes_binary))
    score = model.evaluate(test_values, test_classes_binary, batch_size=16, verbose=1)
    return score
