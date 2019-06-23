from keras import Input, Model
from keras.layers import Conv2D, Activation, BatchNormalization, Add, UpSampling2D


class Generator:
    def __init__(self, input_image_h, input_image_w, image_channels, no_filters, no_residual_blocks):
        self.input_image_h = input_image_h
        self.input_image_w = input_image_w
        self.image_channels = image_channels
        self.input_image_shape = (self.input_image_h, self.input_image_w, self.image_channels)

        self.no_filters = no_filters
        self.no_residual_blocks = no_residual_blocks

        self.model = self.create_generator_net()

    def get_generator_model(self):
        return self.model

    def residual_block(self, input, no_filters):
        conv2d_1 = Conv2D(no_filters, kernel_size=3, strides=1, padding='same')(input)
                            # name="residual_block_conv2d_1"
        relu_1 = Activation('relu')(conv2d_1)
                            # name="residual_block_relu_1"
        batch_norm_1 = BatchNormalization(momentum=0.8)(relu_1)
                            # name="residual_block_batch_norm_1"

        conv2d_2 = Conv2D(no_filters, kernel_size=3, strides=1, padding='same')(batch_norm_1)
                            # name="residual_block_conv2d_2"
        batch_norm_2 = BatchNormalization(momentum=0.8)(conv2d_2)
                            # name="residual_block_batch_norm_2"
        add = Add()([batch_norm_2, input])
                            # name="residual_block_add"

        return add

    def deconv2d(self, input):
        up_sample2d = UpSampling2D(size=2)(input)
                            # name="deconv2d_upsample2d_1"
        conv2d_1 = Conv2D(256, kernel_size=3, strides=1, padding='same')(up_sample2d)
                            # name="deconv2d_conv2d_1"
        relu_1 = Activation('relu')(conv2d_1)
                            # name="deconv2d_relu"
        return relu_1

    def create_generator_net(self):
        original_img = Input(shape=self.input_image_shape, name="create_gen_input")

        # Pre-residual block
        conv2d_1 = Conv2D(64, kernel_size=9, strides=1, padding='same', name="create_gen_conv2d_1")(original_img)
        relu_1 = Activation('relu', name="create_gen_relu_1")(conv2d_1)

        # Propogate through residual blocks
        residual = self.residual_block(relu_1, self.no_filters)
        for _ in range(self.no_residual_blocks - 1 - 1):  # -1 to decrease size by a factor of 4
            residual = self.residual_block(residual, self.no_filters)


        # Post-residual block
        conv2d_2 = Conv2D(64, kernel_size=3, strides=1, padding='same', name="create_gen_conv2d_2")(residual)
        batch_norm_1 = BatchNormalization(momentum=0.8, name="create_gen_batch_norm_1")(conv2d_2)
        add = Add(name="create_gen_add")([batch_norm_1, relu_1])

        conv2d_extra1 = Conv2D(64, kernel_size=3, strides=2, padding='same', name="test1")(add)
        relu_extra1 = Activation('relu', name='test2')(conv2d_extra1)

        conv2d_extra2 = Conv2D(64, kernel_size=3, strides=2, padding='same', name="test3")(relu_extra1)
        relu_extra2 = Activation('relu', name='test4')(conv2d_extra2)


        deconv2d_1 = self.deconv2d(relu_extra2)
        # deconv2d_1 = self.deconv2d(add)
        deconv2d_2 = self.deconv2d(deconv2d_1)

        # Generate high resolution output
        generated_img = Conv2D(self.image_channels, kernel_size=9, strides=1, padding='same', activation='tanh',
                               name="create_gen_generated_img")(deconv2d_2)

        return Model(original_img, generated_img)
