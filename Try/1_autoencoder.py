import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import random
from glob import glob

from datagenerator_read_dir_face import DataGenerator, DataGenerator_predict
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
epochs = 600
epoch_interval = 20

try_cnt = '\\08'

data_path = "D:\\Data\\128\\merged\\"
save_path = 'D:\\Sangmin\\BITProjects\\RESULT\\' + try_cnt
print(save_path)

############ 데이터 읽어오기
X_train = glob(data_path +"X_train/*jpg")
Y_train = glob(data_path +"Y_train/*jpg")
X_test = glob(data_path + "X_test/*jpg")

if len(X_train) == 0 or len(Y_train) == 0:
    print("empty")
    sys.exit()

random.shuffle(X_train)

# Y_train = mul_y(Y_train, 13)
DG = DataGenerator(X_train, Y_train, batch_size= batch_size, dim= (shape, shape))
DGP = DataGenerator_predict(X_test, batch_size= batch_size, dim= (shape, shape))

########### 모델 구성
# vgg = VGGFace(include_top=False, model="vgg16", input_shape=(shape, shape, 3), weights= 'vggface')
# vgg.trainable = False

def build_model():
    inputs = Input(shape=(128, 128, 3))

    x = Conv2D(32, (3,3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(2)(x)
    
    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(2)(x)
  
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(2)(x)



    x = Conv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)


    decoder = Conv2D(3, (3,3), padding='same')(x)

    # model = Model(vgg.input, decoder)c
    model = Model(inputs, decoder)

    # for layer in model.layers:
    #         layer.trainable = False
    #         if layer.name == "pool5":
    #             break

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
    print('Batch Length: ', DG.__len__())
    for i in range(epochs):
        start = time.time()
        for j in range(DG.__len__()):
            X_train, _ = DG.__getitem__(j)
            loss = model.train_on_batch(X_train, X_train)


        print ('\nTraining epoch: %d \nLoss: %f'
            % (i + 1, loss))

        if i % epoch_interval == 0:
            model.save(save_path + '/model_'+ str(i) + '.h5')            
            X_test = DGP.__getitem__(0)

            y_pred = model.predict(X_test) * 0.5 + 0.5
            train_pred = model.predict(X_train) * 0.5 + 0.5
            X_test = X_test * 0.5 + 0.5
            X_train = X_train * 0.5 + 0.5
            for k in range(len(X_test)):
                fig = plt.figure(figsize=(8, 2))

                plot = fig.add_subplot(1, 4, 1)
                plot.set_title('X_train')
                plt.imshow(X_train[k])

                ############ prediction image
                # pred = y_pred[0].reshape(shape,shape, 3)    

                plot = fig.add_subplot(1, 4, 2)
                plot.set_title('train_pred')
                plt.imshow(train_pred[k])

                plot = fig.add_subplot(1, 4, 3)
                plot.set_title('x_test')
                plt.imshow(X_test[k])

                ############ prediction image
                # pred = y_pred[0].reshape(shape,shape, 3)    

                plot = fig.add_subplot(1, 4, 4)
                plot.set_title('predict')
                plt.imshow(y_pred[k])

                plt_savepath = '%d-%d.png' % (i, k)
                plt.savefig(save_path +'\\' + plt_savepath)
                plt.close()
                


        DG.on_epoch_end()
        DGP.on_epoch_end()

        end = time.time()
        print("time per epoch: ", format(end - start, '.2f'))



train()