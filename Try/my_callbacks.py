import datetime
import tensorflow as tf
import keras.backend as K

from keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt
import random

class My_Callback(tf.keras.callbacks.Callback):

    def __init__(self, x_test, y_test, save_path, epoch_interval=1):
        self.X_test = x_test
        self.Y_test = y_test
        self.epoch_interval = epoch_interval
        self.save_path = save_path

                
    def plot_image(self, images, preds, index, fig):
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

    def on_epoch_end(self, epoch, logs=None):
        print("epoch end")
        if epoch % self.epoch_interval == 0:
            self.model.save(self.save_path + '/model_'+ str(epoch) + '.h5')

            for i in range(10):
                images=[]
                for j in range(3):
                    num = random.randint(0, len(self.X_test)-1)
                    print('random: ', num)
                    img_tensor = self.X_test[num]
                    images.append(img_tensor)

                images = np.array(images)
                print(images.shape)

                proc_images = images         
                y_pred = self.model.predict(proc_images)
                # y_pred = y_pred * 0.5 + 0.5

                fig = plt.figure()
                ############ x_test image
                # test = rand_img.reshape(shape,shape, 3)

                self.plot_image(images, y_pred, 1, fig)
                self.plot_image(images, y_pred, 2, fig)
                self.plot_image(images, y_pred, 3, fig)


                # plt.show()
                plt.savefig(self.save_path + "\\" + str(epoch) + "_" + str(i) + ".png")

                plt.close()
