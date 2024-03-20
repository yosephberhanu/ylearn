# NN
# https://victorzhou.com/blog/intro-to-neural-networks/
# https://aosabook.org/en/500L/optical-character-recognition-ocr.html
# https://www.youtube.com/watch?v=o64FV-ez6Gw
# https://github.com/Sentdex/nnfs_book
# Decision Trees
# https://lethalbrains.com/learn-ml-algorithms-by-coding-decision-trees-439ac503c9a4
from ylearn.nn.activation import Activation
from ylearn.nn.layers import Dense, Dropout

from ylearn.nn.models import Sequential

# from ylearn.nn.loss import SoftmaxCategoricalCrossEntropy, CategoricalCrossEntropy, Accuracy

# from ylearn.preprocessing import one_hot
from ylearn.nn.optimizers import Adam

# Numpy
import numpy as np

# NNFS(Activation)
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

mdl = Sequential()


X, y = spiral_data(samples=100, classes=3)

mdl.add(Dense(n_output=512, n_input=2, l2=5e-4))
mdl.add(Dropout(0.1))
mdl.add(Dense(n_output=3, l2=5e-4, activation=Activation("softmax")))

mdl.build(
    optimizer=Adam(learning_rate=0.05, decay=5e-7),
    loss_function=CategoricalCrossEntropy(),
)
# mdl.desc()
mdl.fit(X, y, epochs=100)

# # Test

# X_test, y_test = spiral_data(samples = 100, classes = 3)
# y_hat = mdl.predict(X_test)
# acc = Accuracy()
# print(acc.calculate(y_hat, y_test))


# https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
# model = Sequential()
# model.add(LSTM(
#     256,
#     input_shape=(network_input.shape[1], network_input.shape[2]),
#     return_sequences=True
# ))
# model.add(Dropout(0.3))
# model.add(LSTM(512, return_sequences=True))
# model.add(Dropout(0.3))
# model.add(LSTM(256))
# model.add(Dense(256))
# model.add(Dropout(0.3))
# model.add(Dense(n_vocab))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
