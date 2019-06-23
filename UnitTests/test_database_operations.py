import unittest
from copy import deepcopy
from os.path import join

import cv2
import numpy as np
from keras.engine.saving import model_from_json
from tinydb import TinyDB

from config import ROOT_DIR
from database.tiny_db_utils import create_db_and_main_tables, purge_db
from monetary.operation_utils import charge_opperation
from monetary.pricesEnum import Prices
from utils.image_processor import ImageProcessor


class TestDatabaseOperations(unittest.TestCase):

    def test_create_db(self):
        database_path = "./test_data/test_db.json"
        create_db_and_main_tables(database_path)

        db = TinyDB(database_path)
        tables = sorted(list(db.tables()))

        self.assertEqual(tables[0], "Operation")
        self.assertEqual(tables[1], "User")


    def test_purge_db(self):
        database_path = "./test_data/test_db.json"
        create_db_and_main_tables(database_path)

        purge_db(database_path)

        db = TinyDB(database_path)
        tables = sorted(list(db.tables()))

        print(tables)

        self.assertEqual(tables[0], "_default")
        self.assertEqual(len(tables), 1)

    def test_charge_operation(self):
        database_path = "./test_data/test_db.json"
        create_db_and_main_tables(database_path)
        db = TinyDB(database_path)

        users = db.table("User")
        users.insert(
            {'username': "test",
             'password': "test",
             'occupation': "test",
             'credit_card_no': "test",
             'subscription_type': "DEFAULT"
             })

        charge_opperation(database_path, "test", 1)
        operations = db.table("Operation")

        self.assertEqual(len(operations.all()),1)
