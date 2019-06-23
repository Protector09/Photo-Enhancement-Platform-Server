import unittest
from copy import deepcopy
from os.path import join

import cv2
import numpy as np
from keras.engine.saving import model_from_json

from config import ROOT_DIR
from utils.image_processor import ImageProcessor


class TestImageProcessor(unittest.TestCase):

    def test_reduce_yellow(self):
        proc = ImageProcessor()
        test_image = [
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
            ], [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
            ], [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
            ]
        ]
        test_image = np.asarray(test_image, dtype='uint8')

        new_image = proc._reduce_yellow(deepcopy(test_image), 1)

        self.assertEqual(test_image[0, 0, 0] - 1, new_image[0, 0, 0])
        self.assertEqual(test_image[0, 0, 1] - 1, new_image[0, 0, 1])
        self.assertEqual(test_image[0, 0, 2], new_image[0, 0, 2])

    def test_split_image(self):
        test_image = cv2.imread("./test_data/test_image.jpg")
        proc = ImageProcessor()
        results = proc._split_image(deepcopy(test_image), overlay=20)

        self.assertEqual(len(results), 3)
        self.assertEqual(len(results[0]), 4)
        self.assertEqual(results[0][1][0], 1)
        self.assertEqual(results[0][1][1], 0)

    def test_merge_image(self):
        test_image = cv2.imread("./test_data/test_image.jpg")
        proc = ImageProcessor()
        results = proc._split_image(deepcopy(test_image), overlay=20)

        new_image = proc._recreate_image(results[0], 224, 224, results[1], results[2])
        self.assertEqual(test_image[0, 0, 0], new_image[0, 0, 0])
        self.assertEqual(test_image[223, 223, 2], new_image[223, 223, 2])
        self.assertEqual(test_image[223, 0, 2], new_image[223, 0, 2])
        self.assertEqual(test_image[0, 223, 2], new_image[0, 223, 2])

    def test_apply_model(self):
        test_image = cv2.imread("./test_data/test_image.jpg")[:224, :224, :]
        proc = ImageProcessor()

        name = "super_resolution_model"
        with open(join(ROOT_DIR, 'GAN/Models/{}.json'.format(name)), 'r') as json_file:
            model = model_from_json(json_file.read())
        model.load_weights(join(ROOT_DIR, "GAN/Models/{}.h5".format(name)))
        model._make_predict_function()

        results = proc._apply_model(test_image, model)

        processed_test_image = cv2.imread("./test_data/test_image_model_applied.jpg")

        self.assertListEqual(list(results[0][0]), list(processed_test_image[0][0]))

    def test_image_processing(self):
        test_image = cv2.imread("./test_data/test_image.jpg")[:224, :224, :]
        proc = ImageProcessor()

        name = "super_resolution_model"
        with open(join(ROOT_DIR, 'GAN/Models/{}.json'.format(name)), 'r') as json_file:
            model = model_from_json(json_file.read())
        model.load_weights(join(ROOT_DIR, "GAN/Models/{}.h5".format(name)))
        model._make_predict_function()

        result = proc.process_image(image=test_image, model=model, overlay=20)

        processed_test_image = cv2.imread("./test_data/test_image_processed.png")

        self.assertListEqual(list(result[100][100]), list(processed_test_image[100][100]))
