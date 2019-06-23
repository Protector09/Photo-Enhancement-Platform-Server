from keras import Input, Model
from keras.applications import VGG19


class VGG:
    def __init__(self, input_image_h, input_image_w, image_channels, optimizer):

        self.input_image_h = input_image_h
        self.input_image_w = input_image_w
        self.image_channels = image_channels
        self.input_image_shape = (self.input_image_h, self.input_image_w, self.image_channels)

        self.optimizer = optimizer

        self.model = self.create_vgg()
        self.model.trainable = False

    def get_vgg_model(self):
        return self.model

    def create_vgg(self):
        """
        Pre-trained VGG19 model that outputs image features extracted at the third block of the model
        """
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.input_image_shape)

        img_features = vgg(img)

        return Model(img, img_features)
