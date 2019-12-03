import numpy as np
import matplotlib.pyplot as plt

import random
from keras import backend as K
from keras.models import load_model, Model
import glob
import PIL.Image as pilimg
from keras.applications.vgg19 import VGG19

from keras.optimizers import Adam

from keras.models import load_model, Model, Input, model_from_json
import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt
import numpy
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
import keras.backend as K
import sys

import ntpath


shape = 128
batch_size = 64
is_127 = True

class VGG_LOSS(object):
    def __init__(self, image_shape):
        self.image_shape = image_shape
    def vgg19_loss(self, true, prediction):
        vgg19 = VGG19(include_top = False, weights = 'imagenet', input_shape = (self.image_shape))
        # Make trainable as False

        vgg19.trainable = False

        for layer in vgg19.layers:
            layer.trainable = False
        
        model = Model(inputs = vgg19.input, outputs = vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        return K.mean(K.square(model(true) - model(prediction)))

def score_r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



# x_test = np.load("lsm/n_npy/" + str(shape) + "/origin_x.npy")

x_files = glob.glob('D:\\Data\\New_data\\X_test\\*jpg')
y_files = glob.glob('D:\\Data\\New_data\\Y_test\\*jpg')
# test_path = glob.glob('D:\Sangmin\Bitprojects\\file\\*jpg')

if len(x_files) == 0:
    print("empty")
    sys.exit()


if is_127:                                              
    loss = VGG_LOSS(image_shape=(shape, shape, 3))
    model = load_model("lsm/model/" + str(shape) + "/gan.h5")
    # model2 = load_model("lsm/model/" + str(shape) + "/vgg_300_kr.h5", custom_objects= {'vgg19_loss' : loss.vgg19_loss})
    # x_test = x_test_org
else:
    # model = load_model("lsm/model/" + str(shape) + "/model.h5")
    # model = load_model("lsm/model/" + str(shape) + "/all.h5", custom_objects={"score_r2": score_r2})
    model.summary()
    # x_test = x_test.astype('float32') /255

def plot_image(images, preds, index):
    img = images[index-1]
    img = image.array_to_img(img)
    plot = fig.add_subplot(2, 3, index)
    plot.set_xticks([])
    plot.set_yticks([])
    plt.imshow(img)

    plot = fig.add_subplot(2, 3, index + 3)
    plot.set_xticks([])
    plot.set_yticks([])
    plt.imshow(preds[index-1])

from keras.preprocessing import image
for i in range(101):      
    ############ 난수 인덱스 생성
    images = []
    for j in range(3):
        num = random.randint(0, len(x_files)-1)
        print('random: ', num)

        img = image.load_img(x_files[num], target_size=(shape, shape))
        img_tensor = image.img_to_array(img)
        images.append(img_tensor)

    images = np.array(images)
    print(images.shape)

    proc_images = images / 255. * 2. - 1 

    ############ 예측
    y_pred = model.predict(proc_images, batch_size=batch_size)
    # y_pred = model2.predict(y_pred, batch_size=batch_size)

    if is_127:
        y_pred = y_pred * 0.5 + 0.5

    fig = plt.figure()
    ############ x_test image
    # test = rand_img.reshape(shape,shape, 3)

    plot_image(images, y_pred, 1)
    plot_image(images, y_pred, 2)
    plot_image(images, y_pred, 3)


    # plt.show()
    plt.savefig(str(i) + ".png")


    ############ true y image
    # label = y_test[num]
    # label = label.reshape(shape,shape, 3)
    # plot = fig.add_subplot(1, 3, 3)
    # plot.set_title('true y')
    # plt.imshow(label)

    # from sklearn.metrics import r2_score
    # r2_y_predict = r2_score(label, pred)
    # print("R2: ", r2_y_predict)
