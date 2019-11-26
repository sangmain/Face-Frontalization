import numpy as np
from keras.layers import Input, Dense, Conv2D, Lambda, concatenate, MaxPool2D, Reshape, Flatten, UpSampling2D
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras import backend as K
from keras import objectives
from keras.losses import mse, kullback_leibler_divergence
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# parameters
batch_size = 32
epoch = 20000
z_dim = 2
img_size = 128
n_labels = 13
color = 3
patience = 1500
n_hidden = 32
line = 55

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

createFolder("./model/{}".format(str(line)))
createFolder("./graph/{}".format(str(line)))

# load data
x = np.load("./npy/cvaex_origin_image_sample_{}_c_n14.npy".format(str(img_size)))
y = np.load("./npy/cvaey_origin_image_sample_{}_c_n14.npy".format(str(img_size)))
x = x.reshape(x.shape[0],x.shape[1],x.shape[2], color)
x = x.astype("float32") / 255
y = to_categorical(y, num_classes= n_labels)
x_train = x[:160 * n_labels]
y_train = y[:160 * n_labels]
x_val = x[160 * n_labels:180 * n_labels]
y_val = y[160 * n_labels:180 * n_labels]
x_test = x[180 * n_labels:]
y_test = y[180 * n_labels:]

# encoder
inputs = Input(shape = (img_size, img_size, color), name = 'encoder_input')
condition = Input(shape = (n_labels,), name= 'labels')
layer_c = Dense(img_size * img_size * color)(condition)
layer_c = Reshape((img_size, img_size, color))(layer_c)

x = concatenate([inputs, layer_c])
x_encoded = Conv2D(n_hidden // 2, 3, padding='same', activation='relu')(x)
x_encoded = MaxPool2D(2,2)(x_encoded)
x_encoded = Conv2D(n_hidden, 3, padding='same', activation='relu')(x_encoded)
x_encoded = MaxPool2D(2,2)(x_encoded)

z_shape = K.int_shape(x_encoded)

x_encoded = Flatten()(x_encoded)
x_encoded = Dense(img_size // 2, activation="relu")(x_encoded)
z_mean = Dense(z_dim)(x_encoded)
z_log_val = Dense(z_dim)(x_encoded)

def sampling(args):
    from keras import backend as K
    z_mean, z_log_val = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape = (batch, dim))
    return z_mean + K.exp(0.5 * z_log_val) * epsilon

z = Lambda(sampling, output_shape=(z_dim,))([z_mean, z_log_val])

encoder = Model([inputs, condition], [z_mean, z_log_val, z])
encoder.summary()

z_input = Input(shape=(z_dim,))
z_con = concatenate([z_input, condition])
layer_z = Dense(z_shape[1] * z_shape[2] * z_shape[3], activation="relu")(z_con)
layer_z = Reshape((z_shape[1], z_shape[2], z_shape[3]))(layer_z)    # 16 16 16

# decoder
z_decoded = Conv2D(n_hidden, 3, padding='same', activation='relu')(layer_z)
z_decoded = UpSampling2D(2,)(z_decoded)
z_decoded = Conv2D(n_hidden, 3, padding='same', activation='relu')(z_decoded)
z_decoded = UpSampling2D(2,)(z_decoded)
z_decoded = Conv2D(n_hidden // 2, 3, padding='same', activation='relu')(z_decoded)

outputs = Conv2D(color, 3, padding='same', activation='relu')(z_decoded)

decoder = Model([z_input, condition], outputs)
decoder.summary()

outputs = decoder([encoder([inputs, condition])[2], condition])
cvae = Model([inputs, condition], outputs)

# loss
def loss():
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= img_size * img_size * color
    kl_loss = 1 + z_log_val - K.square(z_mean) - K.exp(z_log_val)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss = 0.5 * K.sum(K.exp(z_log_val) + K.square(z_mean) -1. - z_log_val, axis = 1)
    cvae_loss = K.mean(reconstruction_loss + kl_loss)
    return cvae_loss

# build model
cvae = Model([inputs, condition], outputs)
cvae.add_loss(loss())
cvae.compile(optimizer='adam')
# cvae.summary()

# train
import keras
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

tb_hist = keras.callbacks.TensorBoard(log_dir='./swh/graph/'+str(line), histogram_freq = 0, write_graph = True, write_images=True)

history = cvae.fit([x_train, y_train], shuffle=True, epochs=epoch, batch_size=batch_size, validation_data=([x_val, y_val], None), verbose=1, callbacks=[tb_hist, es])

cvae.save("./swh/model/{4}/cvae_{0}_{2}_{1}_{3}.h5".format(str(img_size), str(epoch), str(batch_size), str(patience), str(line)))
encoder.save("./swh/model/{4}/cvae_encoder_{0}_{2}_{1}_{3}.h5".format(str(img_size), str(epoch), str(batch_size), str(patience), str(line)))
decoder.save("./swh/model/{4}/cvae_decoder_{0}_{2}_{1}_{3}.h5".format(str(img_size), str(epoch), str(batch_size), str(patience), str(line)))
print("Saved model {0}_{2}_{1}_{3}".format(str(img_size), str(epoch), str(batch_size), str(patience)))

# loss & val_loss save graph
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'r', label = "Training loss")
plt.plot(epochs, val_loss, 'b', label = "Validation loss")
plt.title("Training and validtaion loss ({})".format(str(line)))
plt.legend()

plt.savefig('./swh/graph/{}/loss_{}_{}_{}_{}.jpg'.format(str(line), str(img_size), str(epoch), str(batch_size), str(patience)))
plt.show()