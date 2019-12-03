################ This model is to check if feature extraction from image is successful with the current model


import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import random
from glob import glob

from my_callbacks import My_Callback
from keras_vggface.vggface import VGGFace
from keras import backend as K
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, BatchNormalization, LeakyReLU, ReLU, Concatenate, Input


import sys
import os
import time

shape = 128
batch_size = 32
epochs = 2000
epoch_interval = 20

try_cnt = '05'

save_path = os.path.join("./RESULT", try_cnt)
print(save_path)

############ 데이터 읽어오기
X_train = np.load("./npy/merged_x.npy")
Y_train = np.load("./npy/merged_y.npy")


X_test = np.load("./npy/merged_xtest.npy")
Y_test = np.load("./npy/merged_ytest.npy")

print("X_train: ", X_train.shape)
print("Y_Train: ", Y_train.shape)

X_train = X_train / 127.5 - 1
Y_train = Y_train / 127.5 - 1

X_test = X_test / 127.5 - 1
# Y_train = Y_train / 127.5 - 1
if len(X_train) == 0 or len(Y_train) == 0:
    print("empty")
    sys.exit()

random.shuffle(X_train)

# Y_train = mul_y(Y_train, 13)

########### 모델 구성
vgg = VGGFace(include_top=False, model="vgg16", input_shape=(shape, shape, 3), weights= 'vggface')
vgg.trainable = False

def build_model():
    inputs = Input(shape=(128, 128, 3))

    last_layer = vgg.get_layer('pool5').output
    conv2 = vgg.get_layer("conv2_2").output
    conv3 = vgg.get_layer("conv3_3").output
    conv4 = vgg.get_layer("conv4_3").output
    conv5 = vgg.get_layer("conv5_3").output
  
    x = Conv2D(512, (3,3), padding='same')(last_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    layers = Concatenate(axis=-1) ([x, conv5])
    x = Conv2D(512, (3,3), padding='same')(layers)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    layers = Concatenate(axis=-1) ([x, conv4])
    x = Conv2D(512, (3,3), padding='same')(layers)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    layers = Concatenate(axis=-1) ([x, conv3])
    x = Conv2D(256, (3,3), padding='same')(layers)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    layers = Concatenate(axis=-1) ([x, conv2])
    x = Conv2D(128, (3,3), padding='same')(layers)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    decoder = Conv2D(3, (3,3), padding='same')(x)

    model = Model(vgg.input, decoder)
    
    for layer in model.layers:
            layer.trainable = False
            if layer.name == "pool5":
                break

    model.compile(loss='mse', optimizer='adam')

    model.summary()

    return model

def show_graph(history):
    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'test loss'], loc='upper left')
    plt.show()

    plt.plot(history.history['score_r2'])
    plt.plot(history.history['val_score_r2'])
    plt.title('model r2_score')
    plt.ylabel('r2_score')  
    plt.xlabel('epoch')
    plt.legend(['train r2_score', 'test r2_score'], loc='upper left')
    plt.show()


def train():
    model = build_model()

    if not os.path.isdir(save_path):
            os.makedirs(save_path)
    
    cb = My_Callback(X_test, Y_test, save_path, epoch_interval=20)

    model.fit(X_train, Y_train , batch_size=batch_size, epochs=epochs, callbacks=[cb])

train()