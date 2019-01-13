from keras.models import Model, load_model

model_0 = load_model('model_get64_5.h5')
model_64 = load_model('model_get128_1.h5')
model_128 = load_model('model_get256_2.h5')
model_256 = load_model('model_get512_2.h5')
model_512 = load_model('model_get1024.h5')