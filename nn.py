from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, Adadelta, SGD
import matplotlib.pyplot as plt


def nn(train_values, train_classes_binary, test_values, test_classes_binary):

    model = Sequential()

    model.add(Dense(512, input_dim=820, init='uniform', activation='tanh'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(Dense(1024, init='uniform', activation='tanh'))
    # model.add(Dropout(0.5))

    model.add(Dense(128, init='uniform', activation='tanh'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(Dense(64, init='uniform', activation='tanh'))
    # model.add(Dropout(0.5))

    model.add(Dense(14, init='uniform', activation='softmax'))

    adam = Adam()
    rmsprop = RMSprop()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
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

    plt.figure(figsize=(16, 12))
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Train data vs validation data accuracy')
    # plt.colorbar()
    # tick_marks = numpy.arange(len(desc))
    # plt.xticks(tick_marks, desc, rotation=45)
    # plt.yticks(tick_marks, desc)
    # plt.tight_layout()
    plt.plot(list(range(1, 101)), history.history['acc'])
    plt.plot(list(range(1, 101)), history.history['val_acc'])

    plt.legend(['train', 'val'], loc='upper left')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()



    return score, history
