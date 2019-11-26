from keras_vggface.vggface import VGGFace # VGGFACE
from keras.engine import Model
from keras.layers import Flatten, Dense, Input, Conv2D, BatchNormalization, LeakyReLU, ReLU, add, Concatenate, Conv2DTranspose
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
epochs = 100
line = 115
n_show_image = 1
vgg = VGGFace(include_top=False, model="vgg16", input_shape=(128, 128, 3), weights= 'vggface')
vgg.trainable = False
# vgg.summary()
optimizerD = Adam(lr = 2e-6, beta_1 = 0.5, decay = 1e-5)
optimizerC = Adam(lr = 2e-4, beta_1 = 0.5)
number = 0

X = glob("D:/128_X/*jpg")
X_train = X[119889:] # 50
X_predict = X[:119889]
Y = glob("D:/128_Y/*jpg")
Y_train = Y[119889:]

class vggGan():
    def __init__(self):
        self.height = height
        self.width = width
        self.channels = channels
        self.batch_size = batch_size
        self.epochs = epochs
        self.line = line
        self.n_show_image = n_show_image
        self.vgg = vgg
        self.optimizerD = optimizerD
        self.optimizerC = optimizerC
        self.DG = DataGenerator(X_train, Y_train, batch_size = batch_size)
        self.DGP = DataGenerator_predict(X_predict, batch_size = batch_size)
        self.number = number

        patch = int(self.height / 2**1)
        self.disc_patch = (patch, patch, 3)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = self.optimizerD)

        self.generator = self.build_generator()

        self.discriminator.trainable = False

        side = Input(shape = (self.height, self.width, self.channels))
        front = Input(shape = (self.height, self.width, self.channels))
        image = self.generator(side)

        valid = self.discriminator([front, image])

        self.combined = Model([side, front], [image, valid])
        self.combined.compile(loss = ['mae', "mse"], loss_weights=[100, 1], optimizer = self.optimizerC)

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
        # output = LeakyReLU(alpha = alpha)(layer)
        output = ReLU()(layer)
     
        return output

    def build_discriminator(self):
        input_A = Input(shape = (self.height, self.width, self.channels))
        input_B = Input(shape = (self.height, self.width, self.channels))
        input_layer = Concatenate(axis=-1) ([input_A, input_B])
        layer = self.conv2d_block(input_layer, 64)
        # layer = self.conv2d_block(layer, 128)
        # layer = self.conv2d_block(layer, 256)
        # layer = self.conv2d_block(layer, 512)
        # layer = self.conv2d_block(layer, 1024)
        output = Conv2D(filters = 3, kernel_size = (4, 4), strides = 1, padding = "same")(layer)
        model = Model([input_A, input_B], output)
        model.summary()
        return model

    def build_generator(self):
        input = self.vgg.get_layer("pool5").output
        conv2 = self.vgg.get_layer("conv2_2").output
        conv3 = self.vgg.get_layer("conv3_3").output
        conv4 = self.vgg.get_layer("conv4_3").output
        conv5 = self.vgg.get_layer("conv5_3").output
        layers = self.Conv2DTranspose_block(input, 512)
        layers = Concatenate(axis=-1) ([layers, conv5])
        layers = self.Conv2DTranspose_block(layers, 512)
        layers = Concatenate(axis=-1) ([layers, conv4])
        layers = self.Conv2DTranspose_block(layers, 256)
        layers = Concatenate(axis=-1) ([layers, conv3])
        layers = self.Conv2DTranspose_block(layers, 128)
        layers = Concatenate(axis=-1) ([layers, conv2])
        output = Conv2DTranspose(filters = 3, kernel_size = (4, 4), strides = 2, activation = 'tanh', padding = 'same')(layers)
        model = Model(self.vgg.input, output)
        for layer in model.layers:
            layer.trainable = False
            if layer.name == "pool5":
                break
        # model.summary()
        return model

    def train(self, epochs, batch_size, save_interval):
        fake = np.zeros((batch_size,) + self.disc_patch)
        real = np.ones((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch in range(self.DG.__len__()):
                side_images, front_images = self.DG.__getitem__(batch)

                generated_images = self.generator.predict(side_images)

                discriminator_fake_loss = self.discriminator.train_on_batch([front_images, generated_images], fake)
                discriminator_real_loss = self.discriminator.train_on_batch([front_images, front_images], real)
                discriminator_loss = np.add(discriminator_fake_loss, discriminator_real_loss)

                generator_loss = self.combined.train_on_batch([side_images, front_images], [front_images, real])

                print ('\nTraining epoch : %d \nTraining batch : %d / %d \nLoss of discriminator : %f \nLoss of generator : %f'
                    % (epoch + 1 , batch + 1, self.DG.__len__(), discriminator_loss, generator_loss[1]))

                # if batch % save_interval == 0:
                #     save_path = 'D:/Generated Image/Training' + str(line) + '/'
                #     self.save_image(epoch = epoch, batch = batch, front_image = front_images, side_image = side_images, save_path = save_path)
                    # self.generator.save(save_path+"{1}_{0}.h5".format(str(batch), str(line)))

            if epoch % 1 == 0:
                # print(i)
                save_path = 'D:/Generated Image/Training' + str(line) + '/'
                self.save_image(epoch = epoch, batch = batch, front_image = front_images, side_image = side_images, save_path = save_path)

                predict_side_images = self.DGP.__getitem__(0)
                save_path = 'D:/Generated Image/Predict' + str(line) + '/'
                self.save_predict_image(epoch = epoch, batch = batch, side_image = predict_side_images, save_path = save_path)
                self.generator.save("D:/Generated Image/Predict{2}/{1}_{0}.h5".format(str(epoch), str(line), str(line)))

            self.DG.on_epoch_end()
            self.DGP.on_epoch_end()

    def save_image(self, epoch, batch, front_image, side_image, save_path):
        
        generated_image = (0.5 * self.generator.predict(side_image) + 0.5)
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
            
        generated_image = 0.5 * self.generator.predict(side_image) + 0.5    
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
    vgggan = vggGan()
    vgggan.train(epochs = epochs, batch_size = batch_size, save_interval = n_show_image)