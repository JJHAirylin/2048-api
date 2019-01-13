import keras
from keras.models import Model, load_model
import numpy as np
import sys
sys.path.append("./")
from game2048.getdata import read_data_select, read_data_all

model = load_model('model_get128_1.h5')
# get training data
x_train_1, y_train_1 = read_data_all('TrainFor64-128.csv', 1)
x_train_2, y_train_2 = read_data_select('TrainForAll.csv', 1, 64)
x_train = np.append(x_train_1, x_train_2, axis=0)
y_train = np.append(y_train_1, y_train_2,axis=0)
model.fit(x_train, y_train, batch_size = 512, epochs=50)
model.save('model_get128_1.h5')