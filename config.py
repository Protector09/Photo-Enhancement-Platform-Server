import os

from huey import RedisHuey

huey_queue = RedisHuey("licenta", host="localhost")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(ROOT_DIR, 'db.json')
