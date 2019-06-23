import logging
import os
import random
import skimage

from utils.processingEnums import Processes
from utils.image_processor import ImageProcessor

import cv2
from glob import glob
import numpy as np
from skimage.color import rgb2gray, gray2rgb


class DataLoader:
    def __init__(self, dataset_name, img_res=(224, 224), processing_type=Processes.SUPER_RESOLUTION):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.processing_type = processing_type
        self.image_processor = ImageProcessor()

    def load_data(self, batch_size=1, is_testing=False):

        path = glob(self.dataset_name + "/*.jpg")

        batch_images = np.random.choice(path, size=batch_size)

        good_imgs = []
        bad_imgs = []
        for img_path in batch_images:

            img = self.imread(img_path)
            h, w = self.img_res

            good_img = cv2.resize(img, self.img_res)
            bad_img = good_img

            if self.processing_type is Processes.SUPER_RESOLUTION:

                resize_factor = random.randint(2, 4)

                low_h, low_w = int(h / resize_factor), int(w / resize_factor)

                bad_img = cv2.resize(bad_img, (low_h, low_w))
                bad_img = cv2.resize(bad_img, self.img_res)

            elif self.processing_type is Processes.DEBLUR:

                blur_kernel_size = random.randint(1, 4) * 2 + 1

                bad_img = cv2.blur(good_img, (blur_kernel_size, blur_kernel_size))

            # elif self.processing_type is Processes.COLORIZE:
            #
            #     bad_img = rgb2gray(bad_img)
            #     bad_img = gray2rgb(bad_img)

            # elif self.processing_type is Processes.STYLE_TRANSFER:
            #
            #     bad_img = np.array(bad_img, dtype=np.uint8)
            #     bad_img = cv2.cvtColor(bad_img, cv2.COLOR_RGB2BGR)

            elif self.processing_type is Processes.DENOISE:
                noise_amount = random.randint(0, 15)
                bad_img = self.image_processor.add_noise(image=bad_img, ammount=noise_amount)

            else:
                logging.error("Not a valid processing operation")

            if not is_testing and np.random.random() < 0.5:
                good_img = np.fliplr(good_img)
                bad_img = np.fliplr(bad_img)

            good_imgs.append(np.asarray(good_img, "int16"))
            bad_imgs.append(np.asarray(bad_img, "int16"))


        imgs_hr = np.array(good_imgs)# / 127.5 - 1.
        imgs_lr = np.array(bad_imgs)# / 127.5 - 1.

        return imgs_hr, imgs_lr

    def imread(self, path):
        return cv2.imread(path).astype(np.float)

    def imread_gray(self, path):
        return cv2.imread(path, 0).astype(np.float)
