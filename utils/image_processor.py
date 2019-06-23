import logging
import random
import time

import cv2
import numpy as np
from copy import deepcopy

from keras import backend as K

from PIL import ImageEnhance, Image
import tensorflow as tf

class ImageProcessor:
    def __init__(self):
        pass

    def add_noise(self, image, ammount=10):
        """
        Adds pepper noise to an image
        :param ammount:
        :return:
        """

        new_image = deepcopy(image)

        for noise_count in range(ammount):
            rand_size = random.randint(1, 6) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rand_size, rand_size))
            kernel = (kernel - 1) // 255

            x = random.randint(0, 224 - 13)
            y = random.randint(0, 224 - 7)

            new_image[x:x + rand_size, y:y + rand_size, 0] = new_image[x:x + rand_size, y:y + rand_size, 0] * kernel
            new_image[x:x + rand_size, y:y + rand_size, 1] = new_image[x:x + rand_size, y:y + rand_size, 1] * kernel
            new_image[x:x + rand_size, y:y + rand_size, 2] = new_image[x:x + rand_size, y:y + rand_size, 2] * kernel

        return new_image

    def _reduce_yellow(self, new_img, ammount):
        new_img[:, :, :2] -= ammount
        return new_img

    def _split_image(self, image, overlay=20):
        """
        Splits the image in parts
        :param image: np array
        :param overlay: int < 100 number of overlay pixels between images
        :return: array of tuples: (x_coord, y_coord, image_slice), no_blocks_height, no_blocks_width
        """
        logging.info("Splitting image")

        img_h, img_w, _ = image.shape

        # part that is not overlaid
        different_part = 224 - overlay

        # number of 224 x 224 crops that fit in the image and the number of pixels that don't fit
        number_of_224_fits_h = ((img_h - 224) // different_part) + 1
        pixels_left_h = ((img_h - 224) % different_part)

        number_of_224_fits_w = ((img_w - 224) // different_part) + 1
        pixels_left_w = ((img_w - 224) % different_part)

        desired_h = img_h + different_part - pixels_left_h
        desired_w = img_w + different_part - pixels_left_w

        # created space for another crop
        number_of_224_fits_h += 1
        number_of_224_fits_w += 1

        # create blank image with the desired sizes
        blank = np.zeros([desired_h, desired_w, 3], dtype=np.float64)

        new_img = blank
        new_img[:img_h, :img_w, :] += image

        all_slices = []

        for y_coord in range(number_of_224_fits_h):
            for x_coord in range(number_of_224_fits_w):
                my_slice = new_img[y_coord * different_part: y_coord * different_part + 224,
                           x_coord * different_part: x_coord * different_part + 224]
                all_slices.append((x_coord, y_coord, my_slice))

        return all_slices, number_of_224_fits_h, number_of_224_fits_w

    def _apply_model(self, image, model):
        """
        Applies a specific generator model on a image
        :param image: np array (shape = (224, 224, 3))
        :param model: keras model
        :return: np array (shape = (224, 224, 3)) - generated image
        """

        graph = tf.get_default_graph()
        with graph.as_default():
            generated = model.predict(np.expand_dims(image, axis=0))[0]
        generated = 127.5 * (generated + 1)

        return generated

    def _recreate_image(self, image_parts, img_h, img_w, no_blocks_h, no_blocks_w, overlay=20):
        """
        Recreates the image to the original size by merging the split parts
        :param image_parts: array of tuples: (x_coord, y_coord, image_slice (shape = (224, 224, 3)))
        :param overlay: int < 100 number of overlay pixels between images
        :return: np array
        """
        logging.info("Merging image")
        final_image = np.zeros([img_h + 300, img_w + 300, 3], dtype=np.uint8)

        different_part = 224 - overlay

        ovLap= 10

        # TODO convert to BGR


        for x_coord, y_coord, img in image_parts:
            if x_coord == 0:
                if y_coord == 0:
                    final_image[y_coord * different_part: y_coord * different_part + 224 - ovLap,
                    x_coord * different_part: x_coord * different_part + 224 - ovLap] = img[:-ovLap, :-ovLap]
                elif y_coord == no_blocks_h:
                    final_image[ovLap + y_coord * different_part: y_coord * different_part + 224,
                    x_coord * different_part: x_coord * different_part + 224 - ovLap] = img[ovLap:, :-ovLap]
                else:
                    final_image[ovLap + y_coord * different_part: y_coord * different_part + 224 - ovLap,
                    x_coord * different_part: x_coord * different_part + 224 - ovLap] = img[ovLap:-ovLap, :-ovLap]

            elif x_coord == no_blocks_w:
                if y_coord == 0:
                    final_image[y_coord * different_part: y_coord * different_part + 224 - ovLap,
                    ovLap + x_coord * different_part: x_coord * different_part + 224] = img[:-ovLap, ovLap:]
                elif y_coord == no_blocks_h:
                    final_image[ovLap + y_coord * different_part: y_coord * different_part + 224,
                    ovLap + x_coord * different_part: x_coord * different_part + 224] = img[ovLap:, ovLap:]
                else:
                    final_image[ovLap + y_coord * different_part: y_coord * different_part + 224 - ovLap,
                    ovLap + x_coord * different_part: x_coord * different_part + 224] = img[ovLap:-ovLap, ovLap:]
            else:
                if y_coord == 0:
                    final_image[y_coord * different_part: y_coord * different_part + 224 - ovLap,
                    ovLap + x_coord * different_part: x_coord * different_part + 224 - ovLap] = img[:-ovLap, ovLap: -ovLap]
                elif y_coord == no_blocks_h:
                    final_image[ovLap + y_coord * different_part: y_coord * different_part + 224,
                    ovLap + x_coord * different_part: x_coord * different_part + 224 - ovLap] = img[ovLap:, ovLap:-ovLap]
                else:
                    final_image[ovLap + y_coord * different_part: y_coord * different_part + 224 - ovLap,
                    ovLap + x_coord * different_part: x_coord * different_part + 224 - ovLap] = img[ovLap:-ovLap, ovLap:-ovLap]


        final_image = final_image[:img_h, :img_w]
        return final_image

    def process_image(self, image, model, overlay=20):
        """
        Processes an image:
            1. split into 244 x 244 images
            2. apply model on each new image
            3. reconstruct the image form the new images
        :param image: np array
        :param model: keras model
        :param overlay: 20 < int < 100 number of overlay pixels between images
        :return: np array - generated image
        """

        image = image / 127.5 - 1

        if overlay > 100:
            logging.warning("Overlay can not be greater than 100, using 100")
            overlay = 100

        if overlay < 20:
            logging.warning("Overlay can not be smaller than 20, using 20")
            overlay = 20

        img_h, img_w, _ = image.shape

        img_parts, no_blocks_h, no_blocks_w = self._split_image(image=image, overlay=overlay)
        processed_slices = []

        global_st = time.time()
        for index, (x_coord, y_coord, image_slice) in enumerate(img_parts):
            st = time.time()
            processed_slices.append((x_coord, y_coord, self._apply_model(image=image_slice, model=model)))
            logging.info("{}/{} processed in {}".format(index + 1, len(img_parts), time.time() - st))
        logging.info("Global time: {}; Mean time: {}".format(time.time() - global_st,
                                                             (time.time() - global_st) / len(img_parts)))

        new_img = self._recreate_image(image_parts=processed_slices, img_h=img_h, img_w=img_w, no_blocks_h=no_blocks_h,
                                        no_blocks_w=no_blocks_w, overlay=overlay)

        # new_img = np.clip(new_img, 5, 255)
        # new_img = self._reduce_yellow(new_img, ammount=5)

        # new_img = Image.fromarray(new_img)
        #
        # en = ImageEnhance.Contrast(new_img)
        # new_img = en.enhance(1.15)
        #
        new_img = np.asarray(new_img, "uint8")

        return new_img
