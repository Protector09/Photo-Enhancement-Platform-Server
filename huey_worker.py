import argparse
import cv2
import logging

from huey.consumer_options import OptionParserHandler, ConsumerConfig
from keras.engine.saving import model_from_json

from config import huey_queue
from utils.image_processor import ImageProcessor

from keras import backend as K


def get_model_by_name(name):
    with open('./GAN/Models/{}.json'.format(name), 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights("./GAN/Models/{}.h5".format(name))
    model._make_predict_function()
    return model


def start_consumer(workers, worker_type, scheduler_interval, enable_verbose):
    parser_handler = OptionParserHandler()
    parser = parser_handler.get_option_parser()
    options, args = parser.parse_args()

    options = {k: v for k, v in options.__dict__.items()
               if v is not None}
    options["workers"] = workers
    options["worker-type"] = worker_type
    options["scheduler-interval"] = scheduler_interval

    if enable_verbose:
        options["verbose"] = ""

    config = ConsumerConfig(**options)
    config.validate()

    config.setup_logger()

    consumer = huey_queue.create_consumer(**config.values)
    consumer.run()


@huey_queue.task()
def worker_super_resolution_one(image, new_heigth=None, new_width=None, save_filename="huey_SR_test"):


    current_heigth, current_width = image.shape[:2]
    resized_image = image


    if new_heigth is not None:
        if new_width is not None:
            resized_image = cv2.resize(image, (new_width, new_heigth))
        else:
            resized_image = cv2.resize(image, (current_width, new_heigth))
    else:
        if new_width is not None:
            resized_image = cv2.resize(image, (new_width, current_heigth))

    logging.info("Image has been resized from {} to {}".format(image.shape, resized_image.shape))

    processor = ImageProcessor()

    model = get_model_by_name("super_resolution_model")

    new_image = processor.process_image(image=resized_image, model=model, overlay=20)

    cv2.imwrite("./processed_images/{}.png".format(save_filename), new_image)

    return new_image


@huey_queue.task()
def worker_general(image, model_name, save_filename="bad_filename"):
    model = get_model_by_name(model_name)

    processor = ImageProcessor()

    new_image = processor.process_image(image=image, model=model, overlay=20)
    cv2.imwrite("./processed_images/{}.png".format(save_filename), new_image)
    return new_image


@huey_queue.task()
def worker_super_resolution_batch(images):
    for image in images:
        print(image)
    return "Success"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--worker_type', type=str, default="thread")
    parser.add_argument('--scheduler_interval', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--enable_verbose', type=str, default="True")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info("Started Huey Consumer")
    start_consumer(args.workers, args.worker_type, args.scheduler_interval, args.enable_verbose)
