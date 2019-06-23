import cv2
from keras.engine.saving import model_from_json
from keras.optimizers import Adam

from GAN.discriminator import Discriminator
from GAN.gan import GAN
from GAN.generator import Generator
from GAN.vgg import VGG
from GAN.data_loader import DataLoader

from utils.processingEnums import Processes

import numpy as np
import tensorflow as tf

from utils.image_processor import ImageProcessor
from config import ROOT_DIR
from os.path import join

if __name__ == "__main__":
    bad_image_h = 224
    bad_image_w = 224
    bad_image_channels = 3

    good_image_h = bad_image_h
    good_image_w = bad_image_w
    good_image_channels = 3
    dataset_path = "E:/AiDatasets/Licenta/SuperResolutionDataset/div2k_crops/"

    # processing_type = Processes.DEBLUR
    # processing_type = Processes.DENOISE
    processing_type = Processes.SUPER_RESOLUTION

    train = True
    # train = False

    if train:

        good_image_shape = (good_image_h, good_image_w, good_image_channels)
        bad_image_shape = (bad_image_h, bad_image_w, bad_image_channels)

        no_filters = 64
        no_residual_blocks = 16

        optimizer = Adam(0.0001, 0.5)

        patch = int(np.round(good_image_h / 2 ** 4))
        disc_patch = (patch, patch, 1)


        discriminator = Discriminator(input_image_h=good_image_h, input_image_w=good_image_w,
                                      image_channels=good_image_channels, no_filters=no_filters)
        discriminator = discriminator.get_discriminator_model()
        discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


        generator = Generator(input_image_h=bad_image_h, input_image_w=bad_image_w, image_channels=bad_image_channels,
                              no_filters=no_filters, no_residual_blocks=no_residual_blocks)
        generator = generator.get_generator_model()


        vgg = VGG(input_image_h=good_image_h, input_image_w=good_image_w, image_channels=good_image_channels,
                  optimizer=optimizer)
        vgg = vgg.get_vgg_model()
        vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


        data_loader = DataLoader(dataset_name=dataset_path, img_res=(good_image_h, good_image_w), processing_type=processing_type)


        gan = GAN(generator=generator, discriminator=discriminator, vgg=vgg, data_loader=data_loader, optimizer=optimizer,
                  good_shape=good_image_shape, bad_shape=bad_image_shape, disc_patch=disc_patch)
        gan.train(epochs=1, batch_size=1)

    else:

        with open('E:/Licenta/Licenta/GAN/Models/super_resolution_model.json', 'r') as json_file:
            loaded_model = model_from_json(json_file.read())

        loaded_model.load_weights("E:/Licenta/Licenta/GAN//Models/super_resolution_model.h5")
        # loaded_model.summary()

        global model_graph
        model_graph = tf.get_default_graph()

        # Test on image
        processor = ImageProcessor()

        image = cv2.imread("E:/AiDatasets/Licenta/Test/data/test1.jpg")

        cv2.imwrite("E:/AiDatasets/Licenta/Test/results/test1_3_original.jpg", image)

        img_h, img_w, _ = image.shape
        image = cv2.resize(image, (img_w // 4, img_h // 4))
        image = cv2.resize(image, (img_w, img_h))
        cv2.imwrite("E:/AiDatasets/Licenta/Test/results/test1_1_downRes.jpg", image)

        new_image = processor.process_image(image=image, model=loaded_model, model_graph=model_graph, overlay=0)
        cv2.imwrite("E:/AiDatasets/Licenta/Test/results/test1_2_generated.jpg", new_image)
