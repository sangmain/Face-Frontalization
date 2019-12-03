import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import random
from glob import glob

from datagenerator import DataGenerator, DataGenerator_predict
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

try_cnt = '07'

data_path = "./../merged_ext//"
save_path = os.path.join("./RESULT", try_cnt)
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
vgg = VGGFace(include_top=False, model="vgg16", input_shape=(shape, shape, 3), weights= 'vggface')
vgg.trainable = False

def build_model():
    inputs = Input(shape=(128, 128, 3))

    last_layer = vgg.get_layer('pool5').output
    # vgg_model.summary()
    conv2 = vgg.get_layer("conv2_2").output
    conv3 = vgg.get_layer("conv3_3").output
    conv4 = vgg.get_layer("conv4_3").output
    conv5 = vgg.get_layer("conv5_3").output



    x = Conv2D(512, (3,3), padding='same')(last_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    layers = Concatenate(axis=-1) ([x, conv5])
    x = Conv2D(512, (3,3), activation='relu', padding='same')(layers)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    layers = Concatenate(axis=-1) ([x, conv4])
    x = Conv2D(512, (3,3), activation='relu', padding='same')(layers)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    layers = Concatenate(axis=-1) ([x, conv3])
    x = Conv2D(256, (3,3), activation='relu', padding='same')(layers)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    layers = Concatenate(axis=-1) ([x, conv2])
    x = Conv2D(128, (3,3), activation='relu', padding='same')(layers)
    x = LeakyReLU()(x)
    x = UpSampling2D(2)(x)

    decoder = Conv2D(3, (3,3), padding='same')(x)

    model = Model(vgg.input, decoder)
    for layer in model.layers:
            layer.trainable = False
            # print(layer.get_weights())
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

from keras_preprocessing import image
def plot_image(images, preds, index, fig):
        img = images[index-1]
        img = image.array_to_img(img)
        plot = fig.add_subplot(2, 3, index)
        plot.set_xticks([])
        plot.set_yticks([])
        plt.imshow(img)

        plot = fig.add_subplot(2, 3, index + 3)
        plot.set_xticks([])
        plot.set_yticks([])
        pred = image.array_to_img(preds[index - 1 ])
        plt.imshow(pred)

def train():
    model = build_model()

    if not os.path.isdir(save_path):
            os.makedirs(save_path)
    print('Batch Length: ', DG.__len__())
    for i in range(epochs):
        start = time.time()
        for j in range(DG.__len__()):
            X_train, Y_train = DG.__getitem__(j)
            loss = model.train_on_batch(X_train, Y_train)


        print ('\nTraining epoch: %d \nLoss: %f'
            % (i + 1, loss))

        if i % epoch_interval == 0:
            model.save(save_path + '/model_'+ str(i) + '.h5')           
            X_test = DGP.__getitem__(0)
            print(X_test.shape)
            for k in range(10):
                images=[]
                for l in range(3):
                    num = random.randint(0, len(X_test)-1)
                    print('random: ', num)
                    img_tensor = X_test[num]
                    images.append(img_tensor)

                images = np.array(images)
                print(images.shape)

                proc_images = images         
                y_pred = model.predict(proc_images)
                # y_pred = y_pred * 0.5 + 0.5

                fig = plt.figure()
                ############ x_test image
                # test = rand_img.reshape(shape,shape, 3)

                plot_image(images, y_pred, 1, fig)
                plot_image(images, y_pred, 2, fig)
                plot_image(images, y_pred, 3, fig)


                # plt.show()
                plt.savefig(save_path + "\\" + str(i) + "_" + str(k) + ".png")

                plt.close()
                


        DG.on_epoch_end()
        DGP.on_epoch_end()

        end = time.time()
        print("time per epoch: ", format(end - start, '.2f'))



train()


