import argparse
import io
import logging

from flask import Flask, request, Response, send_file
from flask_cors import CORS
import cv2
import numpy as np
from tinydb import TinyDB

from huey_worker import worker_super_resolution_one, worker_super_resolution_batch, worker_general
from config import huey_queue, DB_PATH
from monetary.operation_utils import charge_opperation
from utils.modelsEnums import Models

import tensorflow as tf

app = Flask("Licenta")
CORS(app)


@app.route('/')
def index():
    return Response('Licenta Server', status=200)


@app.route("/login", methods=['POST'])
def login_user():
    data = request.get_json()

    users = db.table("User")

    # TODO: SHA256 passwords

    for user in users:
        if user['username'] == data["username"] and user['password'] == data["password"]:
            logging.info("Found User")
            return Response("True", status=200)

    return Response("False", status=200)


# TODO: USER Page? Get and Update Data


@app.route("/train", methods=['POST'])
def train():
    data = request.get_json()

    if data["username"] == "david_dan":
        pass


@app.route("/register", methods=['POST'])
def register_user():
    data = request.get_json()

    users = db.table("User")

    # TODO: SHA256 passwords

    for user in users:
        if user['username'] == data["username"]:
            logging.info("Duplicate Username")
            return Response("Duplicate Username", status=200)

    users.insert(
        {'username': data['username'],
         'password': data['password'],
         'occupation': data['occupation'],
         'credit_card_no': data['credit_card_no'],
         'subscription_type': data['subscription_type']
         })

    return Response("True", status=200)


@app.route("/deblur", methods=["POST"])
def deblur():
    """
        Deblur one image with algorithm
        Pass image directly as post ( Image size < MAX_IMAGE_SIZE)

        :return:
            Enhanced Image/ Response
        """

    global graph
    graph = tf.get_default_graph()
    with graph.as_default():

        username = request.form["username"]

        logging.info("Added usage to DB")
        charge_opperation(db_path=DB_PATH,username=username, quantity=1)

        image = request.files.get("image", "")
        in_memory_file = io.BytesIO()
        image.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        np_image = cv2.imdecode(data, color_image_flag)

        save_filename = "huey_DB_test"

        result = worker_general(np_image, "deblur_model", save_filename=save_filename)
        result.get(blocking=True)

        return send_file(
            "E:/Licenta/Licenta/processed_images/{}.png".format(save_filename),
            mimetype='image/png',
            as_attachment=True,
            attachment_filename="response.png")


@app.route("/denoise", methods=["POST"])
def denoise():
    """
        Denoise one image with algorithm
        Pass image directly as post ( Image size < MAX_IMAGE_SIZE)

        :return:
            Enhanced Image/ Response
        """

    username = request.form["username"]

    logging.info("Added usage to DB")
    charge_opperation(db_path=DB_PATH,username=username, quantity=1)

    image = request.files.get("image", "")
    in_memory_file = io.BytesIO()
    image.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    np_image = cv2.imdecode(data, color_image_flag)

    save_filename = "huey_DN_test"

    result = worker_general(np_image, "denoise_model", save_filename=save_filename)

    result.get(blocking=True)

    return send_file(
        "E:/Licenta/Licenta/processed_images/{}.png".format(save_filename),
        mimetype='image/png',
        as_attachment=True,
        attachment_filename="response.png")


# @app.route("/colorize", methods=["POST"])
# def colorize():
#     """
#         Colorize one image with algorithm
#         Pass image directly as post ( Image size < MAX_IMAGE_SIZE)
#
#         :return:
#             Enhanced Image/ Response
#         """
#
#     username = request.form["username"]
#
#     logging.info("Added usage to DB")
#     charge_opperation(db_path=DB_PATH,username=username, quantity=1)
#
#     image = request.files.get("image", "")
#     in_memory_file = io.BytesIO()
#     image.save(in_memory_file)
#     data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
#     color_image_flag = 1
#     np_image = cv2.imdecode(data, color_image_flag)
#
#     save_filename = "huey_CO_test"
#
#     result = worker_general(np_image, "colorize_model", save_filename=save_filename)
#     result.get(blocking=True)
#
#     return send_file(
#         "E:/Licenta/Licenta/processed_images/{}.png".format(save_filename),
#         mimetype='image/png',
#         as_attachment=True,
#         attachment_filename="response.png")


@app.route("/super_resolution/one", methods=["POST"])
def super_resolution_one():
    """
        Enhance one image with super resolution algorithm
        Pass image directly as post ( Image size < MAX_IMAGE_SIZE)

        :return:
            Enhanced Image/ Response
        """

    global graph
    graph = tf.get_default_graph()
    with graph.as_default():

        username = request.form["username"]

        # TODO Backup db and read backup if regular is corrupted
        logging.info("Added usage to DB")
        charge_opperation(db_path=DB_PATH,username=username, quantity=1)

        new_heigth = request.form["new_heigth"]
        new_width = request.form["new_width"]
        factor = request.form["factor"]

        if new_heigth == "None":
            new_heigth = None
        else:
            new_heigth = int(new_heigth)

        if new_width == "None":
            new_width = None
        else:
            new_width = int(new_width)

        if factor == "None":
            factor = None
        else:
            factor = float(factor)


        image = request.files.get("image", "")
        in_memory_file = io.BytesIO()
        image.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        np_image = cv2.imdecode(data, color_image_flag)

        if factor and not(new_heigth or new_width):
            h, w, _ = np_image.shape
            new_heigth = int(h * factor)
            new_width = int(w * factor)

        save_filename = "huey_SR_test"

        result = worker_super_resolution_one(np_image, new_heigth=new_heigth, new_width=new_width,
                                             save_filename=save_filename)
        result.get(blocking=True)

        return send_file(
            "E:/Licenta/Licenta/processed_images/{}.png".format(save_filename),
            mimetype='image/png',
            as_attachment=True,
            attachment_filename="response.png")


@app.route("/super_resolution/batch", methods=["POST"])
def super_resolution_one_batch():
    """
    Enhance batch of images with super resolution algorithm
    Pass images after preprocessing (from database?)

    :return:
    Fail/ Succes Response
    """
    data = request.get_json()
    print(data)  # Link to storage location... TBD

    # TODO: Run super_resolution on batch of images, worker - send updates - add to db?

    return Response("Image", status=200)

# TODO: Schedule task to process earnings once a month w Huey


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--port', type=int, default=5000)

    db = TinyDB("db.json")
    # TODO Backup db and read backup if regular is corrupted

    args = parser.parse_args()

    app.logger.info("Starting Server.")

    app.run(debug=False, host='0.0.0.0', port=args.port, threaded=False)
