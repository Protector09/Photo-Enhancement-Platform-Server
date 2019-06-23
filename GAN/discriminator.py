from keras import Input, Model
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense


class Discriminator:
    def __init__(self, input_image_h, input_image_w, image_channels, no_filters):
        self.input_image_h = input_image_h
        self.input_image_w = input_image_w
        self.image_channels = image_channels
        self.input_image_shape = (self.input_image_h, self.input_image_w, self.image_channels)

        self.no_filters = no_filters

        self.model = self.create_discriminator_net()
        self.model.trainable = False

    def get_discriminator_model(self):
        return self.model

    def discriminator_block(self, input, no_filters, strides=1, add_batch_norm=True):
        conv2d = Conv2D(no_filters, kernel_size=3, strides=strides, padding='same')(input)
                                # name="discriminator_block_conv2d"
        leaky_relu = LeakyReLU(alpha=0.2)(conv2d)
                                # name="discriminator_block_leaky_relu"

        if add_batch_norm:
            batch_norm = BatchNormalization(momentum=0.8)(leaky_relu)
                                # name="discriminator_block_batch_norm"

            return batch_norm

        return leaky_relu

    def create_discriminator_net(self):
        input_img = Input(shape=self.input_image_shape, name="create_discr_input")

        discr_1 = self.discriminator_block(input_img, self.no_filters, add_batch_norm=False)
        discr_2 = self.discriminator_block(discr_1, self.no_filters, strides=2)
        discr_3 = self.discriminator_block(discr_2, self.no_filters * 2)
        discr_4 = self.discriminator_block(discr_3, self.no_filters * 2, strides=2)
        discr_5 = self.discriminator_block(discr_4, self.no_filters * 4)
        discr_6 = self.discriminator_block(discr_5, self.no_filters * 4, strides=2)
        discr_7 = self.discriminator_block(discr_6, self.no_filters * 8)
        discr_8 = self.discriminator_block(discr_7, self.no_filters * 8, strides=2)

        dense_1 = Dense(self.no_filters * 16)(discr_8)
        leaky_relu = LeakyReLU(alpha=0.2)(dense_1)
        dense_2 = Dense(1, activation='sigmoid')(leaky_relu)

        return Model(input_img, dense_2)
