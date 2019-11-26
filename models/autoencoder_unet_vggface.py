from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.engine import Model
from keras.layers import Flatten, Dense, Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, UpSampling2D, Concatenate
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from datagenerator import DataGenerator, DataGenerator_predict
from glob import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

height = 128
width = 128
channels = 3
batch_size = 64
epochs = 1000
line = 100
n_show_image = 1
vgg = VGGFace(include_top=False, model="vgg16", weights= 'vggface', input_shape=(128, 128, 3))
vgg.trainable = False
optimizer = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999)
number = 0

X_train = glob("D:/128_X/*jpg")
Y_train = glob("D:/128_Y/*jpg")
X_predict = glob("D:/testdata/*jpg")

class TestModel():

    def __init__(self):
        self.height = height
        self.width = width
        self.channels = channels
        self.batch_size = batch_size
        self.epochs = epochs
        self.line = line
        self.n_show_image = n_show_image
        self.vgg = vgg
        self.optimizer = optimizer
        self.DG = DataGenerator(X_train, Y_train, batch_size = batch_size, dim=(128,128))
        self.DGP = DataGenerator_predict(X_predict, batch_size = batch_size, dim=(128,128))
        self.number = number

        self.AE = self.build_AE()
        self.AE.compile(loss = 'mse', optimizer = self.optimizer)
        # self.AE.fit_generator(self.DG)
        # self.AE.save("TESTAE.H5")

    def conv2d_block(self, layers, filters, kernel_size = (4, 4), strides = 2, momentum = 0.8, alpha = 0.2):
        input = layers

        layer = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(input)
        layer = BatchNormalization(momentum = momentum)(layer)
        output = LeakyReLU(alpha = alpha)(layer)

        return output

    def Conv2DTranspose_block(self, layers, filters, kernel_size = (4, 4), strides = 2, momentum = 0.8, alpha = 0.2):
        input = layers

        layer = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(input)
        layer = BatchNormalization(momentum = momentum)(layer)
        output = LeakyReLU(alpha = alpha)(layer)
    
        return output

    def build_AE(self):

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

    def train(self, epochs, batch_size, save_interval):
        for i in range(epochs):
            for j in range(self.DG.__len__()):
                side_images, front_images = self.DG.__getitem__(j)
                
                AE_loss = self.AE.train_on_batch(side_images, front_images)
                                
                print ('\nTraining epoch : %d \nTraining batch : %d / %d \nLoss of generator : %f'
                        % (i + 1, j + 1, self.DG.__len__(), AE_loss))
                
                # if j % save_interval == 0:
                #     # save_path = 'D:/Generated Image/Training' + str(line) + '/'
                #     # self.save_image(epoch = i, batch = j, front_image = front_images, side_image = side_images, save_path = save_path)

                #     predict_side_images = self.DGP.__getitem__(j)
                #     predict_front_images = self.AE.predict_on_batch(predict_side_images)
                #     save_path = 'D:/Generated Image/Predict' + str(line) + '/'
                #     self.save_image(epoch = i, batch = j, front_image = predict_front_images, side_image = predict_side_images, save_path = save_path)
            
            if i % 1 == 0:
                # print(i)
                # save_path = 'D:/Generated Image/Training' + str(line) + '/'
                # self.save_image(epoch = i, batch = j, front_image = front_images, side_image = side_images, save_path = save_path)

                predict_side_images = self.DGP.__getitem__(0)
                save_path = 'D:/Generated Image/Predict' + str(line) + '/'
                self.save_predict_image(epoch = i, batch = j, side_image = predict_side_images, save_path = save_path)
                self.AE.save("D:/Generated Image/Predict{2}/{1}_{0}.h5".format(str(i), str(line), str(line)))

            self.DG.on_epoch_end()
            self.DGP.on_epoch_end()


    def save_image(self, epoch, batch, front_image, side_image, save_path):
        
        generated_image = 0.5 * self.AE.predict(side_image) + 0.5
        front_image = (255 * ((front_image) + 1)/2).astype(np.uint8)     
        side_image = (255 * ((side_image)+1)/2).astype(np.uint8)
        
        for i in range(self.batch_size):
            plt.figure(figsize = (8, 2))

            plt.subplots_adjust(wspace = 0.6)

            for m in range(n_show_image):
                generated_image_plot = plt.subplot(1, 3, m + 1 + (2 * n_show_image))
                generated_image_plot.set_title('Generated image (front image)')
                plt.imshow(generated_image[i])

                original_front_face_image_plot = plt.subplot(1, 3, m + 1 + n_show_image)
                original_front_face_image_plot.set_title('Origninal front image')
                plt.imshow(front_image[i])

                original_side_face_image_plot = plt.subplot(1, 3, m + 1)
                original_side_face_image_plot.set_title('Origninal side image')
                plt.imshow(side_image[i])

                # Don't show axis of x and y
                generated_image_plot.axis('off')
                original_front_face_image_plot.axis('off')
                original_side_face_image_plot.axis('off')

                self.number += 1

                # plt.show()

            save_path = save_path

            # Check folder presence
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            save_name = '%d-%d-%d.png' % (epoch, batch, i)
            save_name = os.path.join(save_path, save_name)
        
            plt.savefig(save_name)
            plt.close()

    def save_predict_image(self, epoch, batch, side_image, save_path):
            
        generated_image = 0.5 * self.AE.predict(side_image) + 0.5    
        side_image = (255 * ((side_image)+1)/2).astype(np.uint8)
        
        for i in range(self.batch_size):
            plt.figure(figsize = (8, 2))

            plt.subplots_adjust(wspace = 0.6)

            for m in range(n_show_image):
                generated_image_plot = plt.subplot(1, 2, m + 1 + n_show_image)
                generated_image_plot.set_title('Generated image (front image)')
                plt.imshow(generated_image[i])

                original_side_face_image_plot = plt.subplot(1, 2, m + 1)
                original_side_face_image_plot.set_title('Origninal side image')
                plt.imshow(side_image[i])

                # Don't show axis of x and y
                generated_image_plot.axis('off')
                original_side_face_image_plot.axis('off')

                self.number += 1

                # plt.show()

            save_path = save_path

            # Check folder presence
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            save_name = '%d-%d-%d.png' % (epoch, batch, i)
            save_name = os.path.join(save_path, save_name)
        
            plt.savefig(save_name)
            plt.close()


if __name__ == '__main__':
    AE = TestModel()
    AE.train(epochs = epochs, batch_size = batch_size, save_interval = n_show_image)
