import datetime
import os
import time

import numpy as np
from keras import Input, Model

class GAN:
    def __init__(self, generator, discriminator, vgg, data_loader, optimizer, good_shape, bad_shape, disc_patch):
        self.optimizer = optimizer

        self.generator = generator
        self.generator.summary()

        self.discriminator = discriminator
        self.discriminator.summary()

        self.vgg = vgg
        # self.vgg.summary()

        self.data_loader = data_loader

        # --------------------------

        self.disc_patch = disc_patch
        self.img_shape = good_shape

        good_img = Input(shape=good_shape)
        bad_img = Input(shape=bad_shape)

        created_good_model = self.generator(bad_img)
        created_features = self.vgg(created_good_model)

        self.discriminator.trainable = False

        valid_features = self.discriminator(created_good_model)

        self.gan = Model([bad_img, good_img], [valid_features, created_features])
        self.gan.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=self.optimizer)
        self.gan.summary()

    def save_model_to_disk(self, model, path, name):
        model_json = model.to_json()
        filename = os.path.join(path, "{}.json".format(name))
        with open(filename, 'w') as json_file:
            json_file.write(model_json)

        filename = os.path.join(path, "{}.h5".format(name))
        model.save_weights(filename)
        print("Model successfully saved to disk!")

    def train(self, epochs, batch_size=1):
        all_start_time = time.time()

        for epoch in range(epochs):

            epoch_start_time = time.time()

            good_imgs, bad_imgs = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            created_good_imgs = self.generator.predict(bad_imgs)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated = Fake)
            real_d_loss = self.discriminator.train_on_batch(good_imgs, valid)
            fake_d_loss = self.discriminator.train_on_batch(created_good_imgs, fake)
            d_loss = 0.5 * np.add(real_d_loss, fake_d_loss)

            good_imgs, bad_imgs = self.data_loader.load_data(batch_size)

            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            features_vgg = self.vgg.predict(good_imgs)

            # Train the generators
            g_loss = self.gan.train_on_batch([bad_imgs, good_imgs], [valid, features_vgg])

            print("{} epochs, {} time passed this epoch, {} total time passed".format(epoch, time.time() - epoch_start_time, time.time() - all_start_time))

            if epoch % 500 == 0:
                self.save_model_to_disk(self.generator, "./Models/",
                                        "model_{}_{}_{}".format(datetime.datetime.now().date().day,
                                                                self.img_shape[0],
                                                                epoch))

