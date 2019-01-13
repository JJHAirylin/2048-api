import keras
from keras.layers import Input, Dense, Conv2D, concatenate, Flatten, BatchNormalization, Activation
from keras.models import Model
import numpy as np

import sys
sys.path.append("./")
from game2048.getdata import read_data_all

# define model
inputs = Input((4,4,11))
conv = inputs
FILTERS = 128
conv41 = Conv2D(filters=FILTERS,kernel_size=(4, 1),kernel_initializer='he_uniform')(conv)
conv14 = Conv2D(filters=FILTERS,kernel_size=(1, 4),kernel_initializer='he_uniform')(conv)
conv22 = Conv2D(filters=FILTERS,kernel_size=(2, 2),kernel_initializer='he_uniform')(conv)
conv33 = Conv2D(filters=FILTERS,kernel_size=(3, 3),kernel_initializer='he_uniform')(conv)
conv44 = Conv2D(filters=FILTERS,kernel_size=(4, 4),kernel_initializer='he_uniform')(conv)

hidden = concatenate([Flatten()(conv41),Flatten()(conv14),Flatten()(conv22),Flatten()(conv33),Flatten()(conv44)])
x = BatchNormalization()(hidden)
x = Activation('relu')(hidden)

for width in[512,128]:
    x = Dense(width,kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(hidden)
outputs = Dense(4,activation='softmax')(x)
model = Model(inputs,outputs)
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# get training data
x_train, y_train = read_data_all('TrainFor128-256.csv', 1)
model.fit(x_train, y_train, batch_size = 512, epochs=50)
model.save('model.h5')
