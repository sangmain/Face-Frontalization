import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, MaxPool2D
from keras.models import Model, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from datagenerator import DataGenerator, DataGenerator_predict
from glob import glob

# load data
X = glob("D:/128_X/*jpg")
X_train = X[119889:] # 50
X_predict = X[:119889]
Y = glob("D:/128_Y/*jpg")
Y_train = Y[119889:]

# parameters
height = 128
width = 128
channels = 3
z_dimension = 512
batch_size = 64
epochs = 10000
line = 3
n_show_image = 1
datagenerator = DataGenerator(X_train, Y_train, batch_size = batch_size)
datagenerator_p = DataGenerator_predict(X_predict, batch_size = batch_size)
optimizerD = Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999)
optimizerG = Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999)


def conv2d_block(layers, filters, kernel_size = (4, 4), strides = 2, momentum = 0.8, alpha = 0.2):
    input = layers

    layer = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(input)
    layer = BatchNormalization(momentum = momentum)(layer)
    output = LeakyReLU(alpha = alpha)(layer)

    return output

def Conv2DTranspose_block(layers, filters, kernel_size = (4, 4), strides = 2, momentum = 0.8, alpha = 0.2):
    input = layers

    layer = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(input)
    layer = BatchNormalization(momentum = momentum)(layer)
    output = LeakyReLU(alpha = alpha)(layer)

    return output
    
class Gan():
    def __init__(self):
        self.height = height
        self.width = width
        self.channels = channels
        self.z_dimension = z_dimension
        self.batch_size = batch_size
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.number = 0

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizerD, metrics = ['accuracy'])

        self.generator = self.build_generator()

        self.discriminator.trainable = False

        z = Input(shape = (self.height, self.width, self.channels))
        image = self.generator(z)

        valid = self.discriminator(image)

        self.combined = Model(z, valid)
        self.combined.compile(loss = 'binary_crossentropy', optimizer = optimizerG)

    def build_discriminator(self):
        input = Input(shape = (self.height, self.width, self.channels))

        layers = conv2d_block(input, 16)
        layers = conv2d_block(layers, 32)
        layers = conv2d_block(layers, 64)
        layers = conv2d_block(layers, 128)
        layers = conv2d_block(layers, 256)
        layers = conv2d_block(layers, 512)
        output = Conv2D(1, kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'sigmoid')(layers)
        
        model = Model(input, output)
        model.summary()
        return model

    def build_generator(self):
        input = Input(shape = (self.height, self.width, self.channels))

        layers = conv2d_block(input, 16)
        layers = conv2d_block(layers, 32)
        layers = conv2d_block(layers, 64)
        layers = conv2d_block(layers, 128)
        layers = conv2d_block(layers, 256)
        layers = conv2d_block(layers, 512)
        layers = conv2d_block(layers, 512)   

        layers = Conv2DTranspose_block(layers, 512)
        layers = Conv2DTranspose_block(layers, 256)
        layers = Conv2DTranspose_block(layers, 128)
        layers = Conv2DTranspose_block(layers, 64)
        layers = Conv2DTranspose_block(layers, 32)
        layers = Conv2DTranspose_block(layers, 16)
        output = Conv2DTranspose(filters = 3, kernel_size = (4, 4), strides = 2, activation = 'tanh', padding = 'same')(layers)
  
        model = Model(input, output)
        # model.summary()
        return model

    def train(self, epochs, batch_size, save_interval):
        fake = np.zeros((batch_size, 1, 1, 1))
        real = np.ones((batch_size, 1, 1, 1))

        for epoch in range(epochs):
            for batch in range(datagenerator.__len__()):
                side_images, front_images = datagenerator.__getitem__(batch)

                generated_images = self.generator.predict(side_images)

                discriminator_fake_loss = self.discriminator.train_on_batch(generated_images, fake)
                discriminator_real_loss = self.discriminator.train_on_batch(front_images, real)
                discriminator_loss = np.add(discriminator_fake_loss, discriminator_real_loss) * 0.5

                generator_loss = self.combined.train_on_batch(side_images, real)

                print ('\nTraining epoch : {} \nTraining batch : {} / {} \nLoss of discriminator : {} \nLoss of generator : {}'
                        .format(epoch + 1 , batch + 1, datagenerator.__len__(), discriminator_loss, generator_loss))

                # if batch % save_interval == 0:
                #     save_path = 'D:/Generated Image/Training' + str(line) + '/'
                #     self.save_image(epoch = epoch, batch = batch, front_image = front_images, side_image = side_images, save_path = save_path)
                #     # self.generator.save(save_path+"{1}_{0}.h5".format(str(batch), str(line)))

            if epoch % 1 == 0:
                # print(i)
                save_path = 'D:/Generated Image/Training' + str(line) + '/'
                self.save_image(epoch = epoch, batch = batch, front_image = front_images, side_image = side_images, save_path = save_path)

                predict_side_images = datagenerator_p.__getitem__(0)
                save_path = 'D:/Generated Image/Predict' + str(line) + '/'
                self.save_predict_image(epoch = epoch, batch = batch, side_image = predict_side_images, save_path = save_path)
                self.generator.save("D:/Generated Image/Predict{2}/{1}_{0}.h5".format(str(epoch), str(line), str(line)))

            datagenerator.on_epoch_end()
            datagenerator_p.on_epoch_end()

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
    dcgan = Gan()
    dcgan.train(epochs = epochs, batch_size = batch_size, save_interval = n_show_image)